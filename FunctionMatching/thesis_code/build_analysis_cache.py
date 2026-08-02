#!/usr/bin/env python3
"""Precompute radare2 analysis and PalmTree embeddings for datasets/libraries."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

from analysis_cache import CachedCodeUnitLoader, DEFAULT_CACHE_DIR
from asm import ASM_NORMALIZATION_MODES, CodeUnit
from model import PALMTREE_POOLING_MODES


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
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


def extract_archive(archive: Path, destination: Path) -> list[Path]:
    result = subprocess.run(
        ["ar", "x", archive.as_posix()],
        cwd=destination,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"ar failed for {archive}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return sorted(path for path in destination.iterdir() if path.is_file())


def main() -> int:
    args = parse_args()
    dataset_dirs = list(args.dataset_dir)
    if not dataset_dirs:
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
    archives = discover_files([*args.libraries_dir, *args.archive], ARCHIVE_MAGIC)
    top_level_inputs: list[tuple[str, Path]] = [
        *(("ELF", path) for path in elfs),
        *(("archive", path) for path in archives),
    ]
    if args.limit > 0:
        top_level_inputs = top_level_inputs[: args.limit]

    print(f"Dataset ELF: {len(elfs)}")
    print(f"Static archives: {len(archives)}")
    print(f"Selected top-level inputs: {len(top_level_inputs)}")
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

    for index, (kind, path) in enumerate(top_level_inputs, start=1):
        before = loader.stats
        if kind == "ELF":
            load_one(path, CodeUnit.TYPE_ELF)
        else:
            with tempfile.TemporaryDirectory(prefix="thesis-cache-ar.") as tmpdir:
                try:
                    members = extract_archive(path, Path(tmpdir))
                except Exception as error:
                    failures.append((path, str(error)))
                    print(f"[WARN] {path}: {error}")
                    if args.fail_fast:
                        raise
                    members = []
                for member in members:
                    load_one(member, CodeUnit.TYPE_CU)
        after = loader.stats
        status = "hit" if after.hits > before.hits and after.misses == before.misses else "done"
        print(
            f"[{index}/{len(top_level_inputs)}] {status:4} {kind:7} {path} "
            f"(hits={after.hits}, misses={after.misses}, writes={after.writes})"
        )

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - started))
    stats = loader.stats
    print(
        f"Completed in {elapsed}: units={processed_units}, hits={stats.hits}, "
        f"misses={stats.misses}, writes={stats.writes}, failures={len(failures)}"
    )
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
