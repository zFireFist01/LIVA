#!/usr/bin/env python3
"""Validate the final mmap cache and write a compact reproducibility summary."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from numpy_cache import entry_size, is_complete_entry


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
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()

    counts: Counter[str] = Counter()
    bytes_by_type: Counter[str] = Counter()
    elf_hashes_in_cache: set[str] = set()
    malformed: list[str] = []
    entry_keys: set[str] = set()
    for path in sorted((cache_dir / "entries").glob("*/*.numpy")):
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            identity = metadata["identity"]
            if not is_complete_entry(path, identity):
                raise ValueError("incomplete entry")
            unit_type = str(identity["unit_type"])
            counts[unit_type] += 1
            bytes_by_type[unit_type] += entry_size(path)
            entry_keys.add(path.name.removesuffix(".numpy"))
            if unit_type == "ELF":
                elf_hashes_in_cache.add(str(identity["binary_sha256"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            malformed.append(f"{path}: {error}")

    pickles = list((cache_dir / "entries").glob("*/*.pickle"))
    temporary_entries = [
        path.as_posix()
        for path in (cache_dir / "entries").glob("*/.*")
        if path.is_dir()
    ]
    coordination_locks = list((cache_dir / "locks").rglob("*.lock")) + list(
        (cache_dir / "entries").glob("*/.numpy-cache-write.lock")
    )
    with args.dataset_manifest.expanduser().resolve().open(
        encoding="utf-8", newline=""
    ) as stream:
        dataset_rows = list(csv.DictReader(stream))
    expected_elf_hashes = {row["binary_sha256"] for row in dataset_rows}

    with args.library_matrix.expanduser().resolve().open(
        encoding="utf-8", newline=""
    ) as stream:
        selected_archives = sum(
            row["status"] == "selected"
            for row in csv.DictReader(stream, delimiter="\t")
        )
    provenance_path = cache_dir / "library_provenance.jsonl"
    provenance = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line
    ] if provenance_path.is_file() else []
    missing_indices = 0
    missing_member_entries = 0
    member_occurrences = 0
    referenced_member_keys: set[str] = set()
    for record in provenance:
        index_path = cache_dir / record["archive_index"]
        if not index_path.is_file():
            missing_indices += 1
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        member_occurrences += len(index["members"])
        for member in index["members"]:
            key = member["cache_key"]
            referenced_member_keys.add(key)
            if key not in entry_keys:
                missing_member_entries += 1

    summary = {
        "schema": 1,
        "cache_dir": cache_dir.as_posix(),
        "numpy_entries_by_type": dict(sorted(counts.items())),
        "numpy_bytes_by_type": dict(sorted(bytes_by_type.items())),
        "legacy_pickle_entries": len(pickles),
        "temporary_entries": len(temporary_entries),
        "coordination_lock_files": len(coordination_locks),
        "malformed_numpy_entries": malformed,
        "dataset_elf_records": len(dataset_rows),
        "dataset_elf_unique_hashes": len(expected_elf_hashes),
        "missing_dataset_elf_cache": len(expected_elf_hashes - elf_hashes_in_cache),
        "extra_elf_cache": len(elf_hashes_in_cache - expected_elf_hashes),
        "selected_archives": selected_archives,
        "library_provenance_records": len(provenance),
        "archive_member_occurrences": member_occurrences,
        "referenced_unique_cu_cache_keys": len(referenced_member_keys),
        "missing_archive_indices": missing_indices,
        "missing_member_cache_entries": missing_member_entries,
        "valid": (
            not pickles
            and not temporary_entries
            and not coordination_locks
            and not malformed
            and expected_elf_hashes == elf_hashes_in_cache
            and len(provenance) == selected_archives
            and missing_indices == 0
            and missing_member_entries == 0
        ),
    }
    output = cache_dir / "cache_inventory.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
