#!/usr/bin/env python3
"""Validate one 1/4 ELF shard with its complete candidate-library cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path


COMPILERS = ("gcc-11", "gcc-13", "clang-14", "clang-18")
OPTIMIZATIONS = {"O0", "O2", "O3", "Os"}
EXPECTED_RECORDS = 219 * len(OPTIMIZATIONS)


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def selected_archives(matrix: Path, library_root: Path | None = None) -> list[str]:
    build_root = library_root or matrix.parents[1] / "builds/libraries"
    with matrix.open(encoding="utf-8", newline="") as stream:
        return sorted({
            os.path.abspath(build_root / row["path"])
            for row in csv.DictReader(stream, delimiter="\t")
            if row["status"] == "selected"
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--library-matrix", type=Path, required=True)
    parser.add_argument("--library-root", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_index < 4:
        parser.error("--shard-index must be in 0..3")

    root = args.artifact_root.expanduser().resolve()
    cache = root / "cache"
    matrix = args.library_matrix.expanduser().resolve()
    library_root = (
        args.library_root.expanduser().resolve() if args.library_root else None
    )
    expected_compiler = COMPILERS[args.shard_index]
    manifest_path = root / "datasets/libseeker/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    errors: list[str] = []

    if len(records) != EXPECTED_RECORDS:
        errors.append(f"ELF records={len(records)}, expected={EXPECTED_RECORDS}")
    programs: dict[str, set[str]] = defaultdict(set)
    commands = Counter()
    coordinates = set()
    for record in records:
        program = str(record.get("program", ""))
        optimization = str(record.get("program_optimization", ""))
        command = str(
            record.get("compiler_command")
            or record.get("compiler_metadata", {}).get("command", "")
        )
        command = Path(command).name
        programs[program].add(optimization)
        commands[command] += 1
        coordinates.add((program, command, optimization))
        for field in ("binary", "ground_truth", "linker_map"):
            if not (root / record[field]).is_file():
                errors.append(f"missing {field}: {record[field]}")
    if len(programs) != 219:
        errors.append(f"programs={len(programs)}, expected=219")
    incomplete = sorted(
        program for program, optimizations in programs.items()
        if optimizations != OPTIMIZATIONS
    )
    if incomplete:
        errors.append(f"programs without four optimizations={len(incomplete)}")
    if commands != Counter({expected_compiler: EXPECTED_RECORDS}):
        errors.append(f"compiler distribution={dict(commands)}")
    if len(coordinates) != EXPECTED_RECORDS:
        errors.append(f"unique coordinates={len(coordinates)}")

    experiment_path = cache / "experiment_elf_provenance.jsonl"
    experiment = jsonl(experiment_path)
    if len(experiment) != EXPECTED_RECORDS:
        errors.append(
            f"ELF cache provenance={len(experiment)}, expected={EXPECTED_RECORDS}"
        )
    for record in experiment:
        if not (cache / record["cache_entry"]).is_dir():
            errors.append(f"missing ELF cache entry: {record['cache_entry']}")

    all_archives = selected_archives(matrix, library_root)
    # ELF work is sharded, but each local group of 876 ELF is searched against
    # the complete candidate-library catalog.
    expected_library_paths = set(all_archives)
    provenance_path = (
        cache / f"library_provenance.shard{args.shard_index}.jsonl"
    )
    library_records = jsonl(provenance_path)
    actual_library_paths = {str(record["path"]) for record in library_records}
    missing = expected_library_paths - actual_library_paths
    extra = actual_library_paths - expected_library_paths
    if missing or extra:
        errors.append(
            f"full library cache mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    for record in library_records:
        index_path = cache / record["archive_index"]
        if not index_path.is_file():
            errors.append(f"missing archive index: {record['archive_index']}")
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for member in index["members"]:
            key = member["cache_key"]
            entry = cache / "entries" / key[:2] / f"{key}.numpy"
            if not entry.is_dir():
                errors.append(f"missing CU cache entry: {key}")

    summary = {
        "schema": 1,
        "valid": not errors,
        "shard_index": args.shard_index,
        "shard_count": 4,
        "compiler": expected_compiler,
        "elf_records": len(records),
        "programs": len(programs),
        "unique_elf_hashes": len({record["binary_sha256"] for record in records}),
        "experiment_cache_records": len(experiment),
        "selected_library_archives_total": len(all_archives),
        "library_archives_cached": len(library_records),
        "full_library_catalog": not missing and not extra,
        "errors": errors[:100],
    }
    output = cache / "shard_inventory.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
