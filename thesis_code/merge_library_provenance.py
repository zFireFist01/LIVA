#!/usr/bin/env python3
"""Merge cache-builder shard journals and validate them against the matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("Dataset/cache"))
    parser.add_argument(
        "--library-matrix",
        type=Path,
        default=Path("Dataset/manifests/library_matrix.tsv"),
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    matrix = args.library_matrix.expanduser().resolve()
    build_root = matrix.parents[1] / "builds/libraries"
    with matrix.open(encoding="utf-8", newline="") as stream:
        selected = {
            os.path.abspath(build_root / row["path"])
            for row in csv.DictReader(stream, delimiter="\t")
            if row["status"] == "selected"
        }

    records: dict[str, dict] = {}
    inputs = sorted(cache_dir.glob("library_provenance.shard*.jsonl"))
    if not inputs:
        raise FileNotFoundError("no library_provenance.shard*.jsonl inputs")
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[str(record["path"])] = record

    missing = sorted(selected - records.keys())
    extra = sorted(records.keys() - selected)
    if missing or extra:
        raise ValueError(
            f"provenance/matrix mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    for record in records.values():
        record["dataset_path"] = Path(record["path"]).relative_to(
            matrix.parent.parent
        ).as_posix()
        index = cache_dir / record["archive_index"]
        if not index.is_file():
            raise FileNotFoundError(f"missing archive index: {index}")

    output = cache_dir / "library_provenance.jsonl"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for key in sorted(records):
                stream.write(
                    json.dumps(records[key], sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema": 1,
        "records": len(records),
        "member_occurrences": sum(int(r["member_count"]) for r in records.values()),
        "roles": dict(sorted(Counter(r["role"] for r in records.values()).items())),
        "toolchains": dict(
            sorted(Counter(r["toolchain"] for r in records.values()).items())
        ),
        "optimizations": dict(
            sorted(Counter(r["optimization"] for r in records.values()).items())
        ),
        "input_shards": [path.as_posix() for path in inputs],
        "output": output.as_posix(),
    }
    (cache_dir / "library_provenance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
