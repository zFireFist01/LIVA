#!/usr/bin/env python3
"""Remove ELF cache entries not referenced by the current dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

from asm import CodeUnit
from convert_analysis_cache import peek_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("Dataset/cache"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("Dataset/datasets/libseeker/manifest.csv"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion; without this flag only write the proposed audit.",
    )
    return parser.parse_args()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest = args.dataset_manifest.expanduser().resolve()
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keep_hashes = {row["binary_sha256"] for row in rows}
    if not keep_hashes:
        raise ValueError("dataset manifest contains no binary hashes")

    # Once the experiment manifest exists, retain its exact cache identities.
    # This removes stale entries produced for the same ELF hash by an older or
    # transient parser/model implementation instead of keeping every identity
    # that merely shares the binary hash.
    experiment_manifest = cache_dir / "experiment_elf_provenance.jsonl"
    keep_keys: set[str] | None = None
    if experiment_manifest.is_file():
        experiment_records = [
            json.loads(line)
            for line in experiment_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        experiment_hashes = {
            str(record["binary_sha256"]) for record in experiment_records
        }
        if experiment_hashes != keep_hashes:
            raise ValueError(
                "refusing exact-key prune: experiment/dataset ELF hashes differ"
            )
        keep_keys = {str(record["cache_key"]) for record in experiment_records}

    candidates: list[tuple[Path, dict, int, str]] = []
    for path in sorted((cache_dir / "entries").glob("*/*.pickle")):
        try:
            identity = peek_identity(path)
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        key = path.name.rsplit(".", 1)[0]
        if identity.get("unit_type") == CodeUnit.TYPE_ELF and (
            key not in keep_keys
            if keep_keys is not None
            else identity.get("binary_sha256") not in keep_hashes
        ):
            candidates.append((path, identity, size, "pickle"))
    for path in sorted((cache_dir / "entries").glob("*/*.numpy")):
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            identity = metadata["identity"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        key = path.name.rsplit(".", 1)[0]
        if identity.get("unit_type") == CodeUnit.TYPE_ELF and (
            key not in keep_keys
            if keep_keys is not None
            else identity.get("binary_sha256") not in keep_hashes
        ):
            candidates.append((path, identity, directory_size(path), "numpy"))

    report_path = cache_dir / "pruned_elf_cache.jsonl"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for path, identity, size, storage in candidates:
                stream.write(
                    json.dumps(
                        {
                            "cache_key": path.name.rsplit(".", 1)[0],
                            "binary_sha256": identity.get("binary_sha256"),
                            "binary_size": identity.get("binary_size"),
                            "cache_bytes": size,
                            "storage": storage,
                            "reason": (
                                "not_referenced_by_exact_experiment_manifest"
                                if keep_keys is not None
                                else "not_referenced_by_current_libseeker_manifest"
                            ),
                            "deleted": bool(args.apply),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)

    if args.apply:
        for path, _identity, _size, storage in candidates:
            if storage == "pickle":
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path)

    total_bytes = sum(item[2] for item in candidates)
    summary = {
        "schema": 1,
        "generated_at_unix": int(time.time()),
        "dataset_manifest": manifest.as_posix(),
        "dataset_records": len(rows),
        "dataset_unique_binary_sha256": len(keep_hashes),
        "selection": "exact_cache_key" if keep_keys is not None else "binary_sha256",
        "referenced_exact_cache_keys": (
            len(keep_keys) if keep_keys is not None else None
        ),
        "candidates": len(candidates),
        "bytes": total_bytes,
        "applied": bool(args.apply),
        "report": report_path.as_posix(),
    }
    (cache_dir / "pruned_elf_cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
