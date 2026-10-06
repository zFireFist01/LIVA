#!/usr/bin/env python3
"""Prune cache entries not used by the final library/ELF experiment manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from convert_analysis_cache import peek_identity


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("Dataset/cache"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("Dataset/datasets/libseeker/manifest.csv"),
    )
    parser.add_argument(
        "--library-matrix",
        type=Path,
        default=Path("Dataset/manifests/library_matrix.tsv"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_manifest = args.dataset_manifest.expanduser().resolve()
    matrix = args.library_matrix.expanduser().resolve()
    provenance_path = cache_dir / "library_provenance.jsonl"

    with dataset_manifest.open(encoding="utf-8", newline="") as stream:
        elf_hashes = {
            row["binary_sha256"] for row in csv.DictReader(stream) if row["binary_sha256"]
        }
    with matrix.open(encoding="utf-8", newline="") as stream:
        selected_count = sum(
            row["status"] == "selected"
            for row in csv.DictReader(stream, delimiter="\t")
        )
    provenance = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(provenance) != selected_count:
        raise ValueError(
            f"refusing to prune: provenance={len(provenance)}, selected={selected_count}"
        )

    cu_keys: set[str] = set()
    for record in provenance:
        index_path = cache_dir / record["archive_index"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["archive_sha256"] != record["archive_sha256"]:
            raise ValueError(f"archive provenance mismatch: {index_path}")
        cu_keys.update(member["cache_key"] for member in index["members"])

    candidates: list[tuple[Path, str, dict, int]] = []
    for path in sorted((cache_dir / "entries").glob("*/*")):
        if path.suffix not in {".numpy", ".pickle"}:
            continue
        try:
            if path.suffix == ".numpy":
                metadata = json.loads(
                    (path / "metadata.json").read_text(encoding="utf-8")
                )
                identity = metadata["identity"]
            else:
                identity = peek_identity(path)
            entry_bytes = size(path)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            continue
        key = path.name.rsplit(".", 1)[0]
        unit_type = identity.get("unit_type")
        keep = (
            unit_type == "CU" and key in cu_keys
        ) or (
            unit_type == "ELF" and identity.get("binary_sha256") in elf_hashes
        )
        if not keep:
            candidates.append((path, key, identity, entry_bytes))

    report = cache_dir / "pruned_unreferenced_cache.jsonl"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for path, key, identity, entry_bytes in candidates:
            stream.write(
                json.dumps(
                    {
                        "cache_key": key,
                        "unit_type": identity.get("unit_type"),
                        "binary_sha256": identity.get("binary_sha256"),
                        "cache_bytes": entry_bytes,
                        "reason": "not_referenced_by_final_experiment_manifests",
                        "deleted": bool(args.apply),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, report)

    if args.apply:
        for path, _key, _identity, _bytes in candidates:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        # Builders are complete once full provenance exists. Lock files are
        # coordination-only and are recreated automatically on future writes.
        shutil.rmtree(cache_dir / "locks", ignore_errors=True)
        for lock_path in (cache_dir / "entries").glob(
            "*/.numpy-cache-write.lock"
        ):
            lock_path.unlink(missing_ok=True)

    summary = {
        "schema": 1,
        "applied": bool(args.apply),
        "selected_archives": selected_count,
        "referenced_cu_cache_keys": len(cu_keys),
        "referenced_elf_hashes": len(elf_hashes),
        "candidates": len(candidates),
        "bytes": sum(item[3] for item in candidates),
        "coordination_locks_removed": bool(args.apply),
        "report": report.as_posix(),
    }
    (cache_dir / "pruned_unreferenced_cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
