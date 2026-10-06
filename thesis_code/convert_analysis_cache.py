#!/usr/bin/env python3
"""Convert trusted legacy CodeUnit pickle entries to NumPy mmap directories.

Conversion is intentionally sequential and transactional.  At most one pickle
and one new entry coexist; with ``--delete-pickle`` the legacy file is removed
only after every matching feature has been compared with the mmap reload.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import pickle
import pickletools
import time
from typing import Any

from analysis_cache import CACHE_FORMAT_VERSION
from asm import CodeUnit
from numpy_cache import entry_size, read_entry, validate_matching_equivalence, write_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("Dataset/cache"))
    parser.add_argument(
        "--types",
        default="CU,ELF",
        help="Comma-separated unit types to convert: CU, ELF, or both.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 converts all entries")
    parser.add_argument(
        "--delete-pickle",
        action="store_true",
        help="delete each legacy pickle after successful full verification",
    )
    parser.add_argument(
        "--progress-every", type=int, default=100, help="progress interval in entries"
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=250,
        help="full cyclic-GC interval for CU entries; ELF entries are collected immediately",
    )
    parser.add_argument(
        "--binary-sha-manifest",
        type=Path,
        help=(
            "Optional CSV/TSV containing binary_sha256; only matching binary "
            "identities are converted (useful for retaining the current dataset only)."
        ),
    )
    return parser.parse_args()


def cache_key(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def peek_identity(path: Path, header_size: int = 4096) -> dict[str, Any]:
    data = path.open("rb").read(header_size)
    wanted = {"unit_type", "binary_sha256", "binary_size"}
    after_key: str | None = None
    values: dict[str, Any] = {}
    try:
        for _opcode, argument, _position in pickletools.genops(data):
            if after_key is not None and isinstance(argument, (str, int)):
                values[after_key] = argument
                after_key = None
                if wanted.issubset(values):
                    break
            if argument in wanted:
                after_key = str(argument)
    except (ValueError, EOFError):
        pass
    return values


def load_allowed_hashes(path: Path) -> set[str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.expanduser().resolve().open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter=delimiter)
        if not rows.fieldnames or "binary_sha256" not in rows.fieldnames:
            raise ValueError(f"binary_sha256 column missing from {path}")
        return {row["binary_sha256"] for row in rows if row["binary_sha256"]}


def append_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def convert_one(path: Path, delete_pickle: bool) -> dict[str, Any]:
    started = time.monotonic()
    legacy_bytes = path.stat().st_size
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("legacy payload is not a dictionary")
    if payload.get("cache_format") != CACHE_FORMAT_VERSION:
        raise ValueError("legacy cache format mismatch")
    identity = payload.get("identity")
    unit = payload.get("code_unit")
    if not isinstance(identity, dict) or not isinstance(unit, CodeUnit):
        raise ValueError("legacy identity or CodeUnit is invalid")
    key = cache_key(identity)
    if path.stem != key:
        raise ValueError("legacy filename does not match its content identity")

    destination = path.with_suffix(".numpy")
    write_entry(destination, identity, unit)
    _converted_identity, converted = read_entry(
        destination, expected_identity=identity
    )
    validate_matching_equivalence(unit, converted)
    functions = len(unit.functions)
    blocks = sum(len(function.blocks) for function in unit.functions)
    instructions = sum(
        len(block.instructions)
        for function in unit.functions
        for block in function.blocks
    )
    numpy_bytes = entry_size(destination)
    record = {
        "key": key,
        "unit_type": identity["unit_type"],
        "binary_sha256": identity["binary_sha256"],
        "binary_size": identity["binary_size"],
        "legacy_bytes": legacy_bytes,
        "numpy_bytes": numpy_bytes,
        "functions": functions,
        "blocks": blocks,
        "instructions": instructions,
        "rodata_bytes": len(unit.rodata_bytes),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "verified": True,
        "legacy_deleted": bool(delete_pickle),
    }
    if delete_pickle:
        path.unlink()
    return record


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    selected_types = {value.strip() for value in args.types.split(",") if value.strip()}
    invalid_types = selected_types - {CodeUnit.TYPE_CU, CodeUnit.TYPE_ELF}
    if invalid_types:
        raise ValueError(f"invalid unit types: {sorted(invalid_types)}")
    entries = cache_dir / "entries"
    legacy_paths = sorted(entries.glob("*/*.pickle"))
    allowed_hashes = (
        load_allowed_hashes(args.binary_sha_manifest)
        if args.binary_sha_manifest
        else None
    )
    candidates = []
    for path in legacy_paths:
        try:
            identity = peek_identity(path)
        except FileNotFoundError:
            continue
        if identity.get("unit_type") not in selected_types:
            continue
        if (
            allowed_hashes is not None
            and identity.get("binary_sha256") not in allowed_hashes
        ):
            continue
        candidates.append(path)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    ledger = cache_dir / "conversion.jsonl"
    summary_path = cache_dir / "conversion_summary.json"
    started = time.monotonic()
    counts: Counter[str] = Counter()
    legacy_bytes = 0
    numpy_bytes = 0
    failures: list[dict[str, str]] = []
    print(
        f"Legacy entries={len(legacy_paths)}, selected={len(candidates)}, "
        f"types={','.join(sorted(selected_types))}, delete={args.delete_pickle}"
    )

    for index, path in enumerate(candidates, start=1):
        converted_type = None
        try:
            record = convert_one(path, args.delete_pickle)
            append_ledger(ledger, record)
            converted_type = record["unit_type"]
            counts[converted_type] += 1
            legacy_bytes += int(record["legacy_bytes"])
            numpy_bytes += int(record["numpy_bytes"])
        except Exception as error:
            failure = {"path": path.as_posix(), "error": str(error)}
            failures.append(failure)
            append_ledger(ledger, {"status": "failed", **failure})
            print(f"[WARN] {path}: {error}", flush=True)
        finally:
            # NetworkX objects form cycles. Small CU entries can be reclaimed
            # in batches; forcing a full collection for every tiny object file
            # dominates conversion time. Large ELF entries are collected at
            # once to keep peak RAM bounded.
            if converted_type == CodeUnit.TYPE_ELF or index % args.gc_every == 0:
                gc.collect()

        if index == 1 or index % args.progress_every == 0 or index == len(candidates):
            elapsed = time.monotonic() - started
            saved = legacy_bytes - numpy_bytes
            print(
                f"[{index}/{len(candidates)}] converted={sum(counts.values())} "
                f"failed={len(failures)} old={legacy_bytes / 2**30:.2f}GiB "
                f"new={numpy_bytes / 2**30:.2f}GiB saved={saved / 2**30:.2f}GiB "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    summary = {
        "cache_dir": cache_dir.as_posix(),
        "selected_types": sorted(selected_types),
        "binary_sha_manifest": (
            args.binary_sha_manifest.expanduser().resolve().as_posix()
            if args.binary_sha_manifest
            else None
        ),
        "delete_pickle": bool(args.delete_pickle),
        "candidates": len(candidates),
        "converted": sum(counts.values()),
        "converted_by_type": dict(counts),
        "failures": failures,
        "legacy_bytes": legacy_bytes,
        "numpy_bytes": numpy_bytes,
        "bytes_saved": legacy_bytes - numpy_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
