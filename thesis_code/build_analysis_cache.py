#!/usr/bin/env python3
"""Precompute radare2 analysis and PalmTree embeddings for datasets/libraries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time

from analysis_cache import CachedCodeUnitLoader, DEFAULT_CACHE_DIR, sha256_file
from archive_utils import extract_archive_members
from asm import ASM_NORMALIZATION_MODES, CodeUnit
from model import PALMTREE_POOLING_MODES


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT.parent / f"{REPO_ROOT.name}_artifacts"
)
ELF_MAGIC = b"\x7fELF"
ARCHIVE_MAGIC = b"!<arch>\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the persistent cache used by main.py. Dataset ELF files and "
            "every object member of selected static archives are cached."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Dataset directory scanned recursively for ELF files. Repeatable. "
            "If omitted, uses <artifact-root>/datasets/{libseeker,unseen}."
        ),
    )
    parser.add_argument(
        "--libraries-dir",
        action="append",
        type=Path,
        default=[],
        help="Directory scanned recursively for static archives. Repeatable.",
    )
    parser.add_argument(
        "--elf",
        action="append",
        type=Path,
        default=[],
        help="One additional ELF path. Repeatable.",
    )
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        default=[],
        help="One additional static archive path. Repeatable.",
    )
    parser.add_argument(
        "--library-matrix",
        type=Path,
        help=(
            "Use only status=selected archives from library_matrix.tsv and "
            "write experiment provenance for every archive/member."
        ),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        help=(
            "Override the build root used to resolve paths in --library-matrix. "
            "The matrix-relative dataset_path remains portable."
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--asm-model",
        type=Path,
        default=SCRIPT_DIR / "palmtree/model/transformer.ep19",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--asm-normalization",
        choices=ASM_NORMALIZATION_MODES,
        default="v2",
    )
    parser.add_argument(
        "--palmtree-pooling",
        choices=PALMTREE_POOLING_MODES,
        default="masked_mean",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many top-level ELF/archive inputs; 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and list inputs without opening Radare2 or writing cache files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first input that cannot be analyzed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N top-level inputs (default: 100).",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--start-position",
        type=int,
        default=1,
        help=(
            "Process from this 1-based position within the selected shard "
            "(default: 1). Useful for safely resuming/splitting a shard."
        ),
    )
    parser.add_argument(
        "--stop-position",
        type=int,
        default=0,
        help=(
            "Stop at this inclusive 1-based position within the selected "
            "shard; 0 means the end (default: 0)."
        ),
    )
    parser.add_argument(
        "--provenance-path",
        type=Path,
        help="Optional per-shard provenance JSONL output path.",
    )
    return parser.parse_args()


def magic(path: Path, size: int) -> bytes:
    try:
        with path.open("rb") as stream:
            return stream.read(size)
    except OSError:
        return b""


def discover_files(roots: list[Path], expected_magic: bytes) -> list[Path]:
    discovered: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input path not found: {root}")
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if path.is_file() and magic(path, len(expected_magic)) == expected_magic:
                discovered.add(path.resolve())
    return sorted(discovered)


def matrix_archives(
    matrix_path: Path, library_root: Path | None = None
) -> tuple[list[Path], dict[str, dict[str, str]]]:
    matrix_path = matrix_path.expanduser().resolve()
    build_root = (
        library_root.expanduser().resolve()
        if library_root is not None
        else matrix_path.parents[1] / "builds/libraries"
    )
    archives: list[Path] = []
    metadata: dict[str, dict[str, str]] = {}
    with matrix_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["status"] != "selected":
                continue
            # Keep the selected path/name even when it is a symlink to another
            # selected archive (e.g. ncurses narrow/wide compatibility names).
            path = Path(os.path.abspath(build_root / row["path"]))
            archives.append(path)
            metadata[path.as_posix()] = {
                key: row[key]
                for key in (
                    "archive",
                    "package",
                    "role",
                    "toolchain",
                    "optimization",
                    "project_candidates",
                )
            }
            metadata[path.as_posix()]["dataset_path"] = (
                Path("builds/libraries") / row["path"]
            ).as_posix()
    return sorted(set(archives)), metadata


def compact_provenance(path: Path) -> int:
    """Keep the newest complete record for every selected archive path."""
    if not path.is_file():
        return 0
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["path"])] = record
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for key in sorted(records):
                stream.write(
                    json.dumps(records[key], sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records)


def main() -> int:
    args = parse_args()
    dataset_dirs = list(args.dataset_dir)
    if not dataset_dirs and not args.library_matrix:
        dataset_dirs = [
            args.artifact_root / "datasets" / "libseeker",
            args.artifact_root / "datasets" / "unseen",
        ]
        dataset_dirs = [path for path in dataset_dirs if path.exists()]
        if not dataset_dirs:
            raise FileNotFoundError(
                "No default datasets found; pass one or more --dataset-dir paths"
            )

    elfs = discover_files([*dataset_dirs, *args.elf], ELF_MAGIC)
    matrix_metadata: dict[str, dict[str, str]] = {}
    matrix_selected: list[Path] = []
    if args.library_matrix:
        matrix_selected, matrix_metadata = matrix_archives(
            args.library_matrix, args.library_root
        )
    archives = sorted(
        set(
            discover_files([*args.libraries_dir, *args.archive], ARCHIVE_MAGIC)
            + matrix_selected
        )
    )
    top_level_inputs: list[tuple[str, Path]] = [
        *(("ELF", path) for path in elfs),
        *(("archive", path) for path in archives),
    ]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.shard_count > 1:
        top_level_inputs = [
            item
            for position, item in enumerate(top_level_inputs)
            if position % args.shard_count == args.shard_index
        ]
    if args.start_position < 1:
        raise ValueError("require start-position >= 1")
    if args.stop_position and args.stop_position < args.start_position:
        raise ValueError("require stop-position >= start-position, or 0")
    interval_stop = args.stop_position or len(top_level_inputs)
    top_level_inputs = top_level_inputs[
        args.start_position - 1 : interval_stop
    ]
    if args.limit > 0:
        top_level_inputs = top_level_inputs[: args.limit]

    print(f"Dataset ELF: {len(elfs)}")
    print(f"Static archives: {len(archives)}")
    print(f"Selected top-level inputs: {len(top_level_inputs)}")
    print(f"Shard: {args.shard_index}/{args.shard_count}")
    print(f"Cache directory: {args.cache_dir.expanduser().resolve()}")
    if args.dry_run:
        for kind, path in top_level_inputs:
            print(f"{kind:7} {path}")
        print("Dry run: no cache file was created.")
        return 0

    loader = CachedCodeUnitLoader(
        args.asm_model,
        device=args.device,
        pooling=args.palmtree_pooling,
        asm_normalization=args.asm_normalization,
        cache_dir=args.cache_dir,
    )
    failures: list[tuple[Path, str]] = []
    provenance_path = (
        args.provenance_path.expanduser().resolve()
        if args.provenance_path
        else args.cache_dir.expanduser().resolve() / "library_provenance.jsonl"
    )
    processed_units = 0
    started = time.time()

    def load_one(path: Path, unit_type: str) -> None:
        nonlocal processed_units
        try:
            loader.load(path, unit_type=unit_type)
            processed_units += 1
        except Exception as error:
            failures.append((path, str(error)))
            print(f"[WARN] {path}: {error}")
            if args.fail_fast:
                raise

    def write_archive_provenance(
        path: Path,
        index_path: Path,
        *,
        archive_sha256: str,
        archive_size: int,
        member_count: int,
        unique_member_cache_keys: int,
    ) -> None:
        record = {
            "schema": 1,
            **matrix_metadata.get(path.as_posix(), {}),
            "path": path.as_posix(),
            "dataset_path": matrix_metadata.get(path.as_posix(), {}).get(
                "dataset_path",
                path.relative_to(REPO_ROOT / "Dataset").as_posix()
                if (REPO_ROOT / "Dataset") in path.parents
                else path.name,
            ),
            "archive_sha256": archive_sha256,
            "archive_size": archive_size,
            "archive_index": index_path.relative_to(
                loader.cache.root
            ).as_posix(),
            "member_count": member_count,
            "unique_member_cache_keys": unique_member_cache_keys,
        }
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with provenance_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )

    for index, (kind, path) in enumerate(top_level_inputs, start=1):
        before = loader.stats
        if kind == "ELF":
            load_one(path, CodeUnit.TYPE_ELF)
        else:
            inspected = loader.inspect_archive_index(path)
            if inspected is not None:
                index_path, archive_index = inspected
                archive_index = loader.ensure_archive_function_counts(
                    index_path,
                    archive_index,
                )
                members = archive_index["members"]
                processed_units += len(members)
                write_archive_provenance(
                    path,
                    index_path,
                    archive_sha256=archive_index["archive_sha256"],
                    archive_size=archive_index["archive_size"],
                    member_count=len(members),
                    unique_member_cache_keys=len(
                        {member["cache_key"] for member in members}
                    ),
                )
                members = None
                if (
                    index == 1
                    or index % max(1, args.progress_every) == 0
                    or index == len(top_level_inputs)
                ):
                    print(
                        f"[{index}/{len(top_level_inputs)}] hit  archive {path} "
                        f"(units={processed_units}, hits={loader.stats.hits}, "
                        f"misses={loader.stats.misses}, "
                        f"writes={loader.stats.writes})",
                        flush=True,
                    )
                continue
            with tempfile.TemporaryDirectory(prefix="thesis-cache-ar.") as tmpdir:
                try:
                    members = extract_archive_members(path, Path(tmpdir))
                except Exception as error:
                    failures.append((path, str(error)))
                    print(f"[WARN] {path}: {error}")
                    if args.fail_fast:
                        raise
                    members = None
                member_records = []
                for member in members or []:
                    try:
                        identity = loader.identity(member.path, CodeUnit.TYPE_CU)
                        if not loader.cache.contains(identity):
                            loader.load(
                                member.path,
                                unit_type=CodeUnit.TYPE_CU,
                                identity=identity,
                            )
                        processed_units += 1
                        member_records.append(
                            (member.name, member.occurrence, identity)
                        )
                    except Exception as error:
                        failures.append((member.path, str(error)))
                        print(f"[WARN] {member.path}: {error}")
                        if args.fail_fast:
                            raise
                if members is not None and len(member_records) == len(members):
                    index_path = loader.store_archive_index(path, member_records)
                    archive_sha256 = sha256_file(path)
                    write_archive_provenance(
                        path,
                        index_path,
                        archive_sha256=archive_sha256,
                        archive_size=path.stat().st_size,
                        member_count=len(member_records),
                        unique_member_cache_keys=len(
                            {
                                loader.cache.key(identity)
                                for _name, _occurrence, identity in member_records
                            }
                        ),
                    )
        after = loader.stats
        status = "hit" if after.hits > before.hits and after.misses == before.misses else "done"
        if (
            index == 1
            or index % max(1, args.progress_every) == 0
            or index == len(top_level_inputs)
        ):
            print(
                f"[{index}/{len(top_level_inputs)}] {status:4} {kind:7} {path} "
                f"(units={processed_units}, hits={after.hits}, "
                f"misses={after.misses}, writes={after.writes})",
                flush=True,
            )

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - started))
    provenance_records = compact_provenance(provenance_path)
    stats = loader.stats
    print(
        f"Completed in {elapsed}: units={processed_units}, hits={stats.hits}, "
        f"misses={stats.misses}, writes={stats.writes}, failures={len(failures)}, "
        f"provenance_records={provenance_records}"
    )
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
