#!/usr/bin/env python3
"""Join dataset/ground-truth provenance with mmap ELF cache entries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from analysis_cache import CachedCodeUnitLoader
from asm import ASM_NORMALIZATION_MODES, CodeUnit
from model import PALMTREE_POOLING_MODES
from numpy_cache import is_complete_entry


SCRIPT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("Dataset/cache"))
    parser.add_argument(
        "--asm-model",
        type=Path,
        default=SCRIPT_DIR / "palmtree/model/transformer.ep19",
    )
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
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    source_manifest = dataset_root / "datasets/libseeker/manifest.json"
    records = json.loads(source_manifest.read_text(encoding="utf-8"))["records"]
    wanted_hashes = {record["binary_sha256"] for record in records}

    # Resolve the exact identity selected by the current parser/model
    # configuration.  A content hash can legitimately have stale cache entries
    # from older implementations; scanning by binary hash alone would make the
    # experiment manifest ambiguous and could select scientifically unrelated
    # features.
    loader = CachedCodeUnitLoader(
        args.asm_model,
        device="cpu",
        pooling=args.palmtree_pooling,
        asm_normalization=args.asm_normalization,
        cache_dir=cache_dir,
        write_cache=False,
    )
    if loader.cache is None:
        raise RuntimeError("cache unexpectedly disabled")

    entries: dict[str, tuple[Path, dict]] = {}
    for record in records:
        binary_sha256 = record["binary_sha256"]
        if binary_sha256 in entries:
            continue
        binary = dataset_root / record["binary"]
        if not binary.is_file():
            raise FileNotFoundError(f"missing experiment binary: {binary}")
        identity = loader.identity(binary, CodeUnit.TYPE_ELF)
        if identity["binary_sha256"] != binary_sha256:
            raise ValueError(f"binary hash mismatch: {binary}")
        cache_path = loader.cache.entry_path(identity)
        if not is_complete_entry(cache_path, identity):
            continue
        metadata = json.loads(
            (cache_path / "metadata.json").read_text(encoding="utf-8")
        )
        entries[binary_sha256] = (cache_path, metadata)

    missing_cache = sorted(wanted_hashes - entries.keys())
    if missing_cache:
        raise ValueError(f"current ELF entries missing from cache: {len(missing_cache)}")

    output = cache_dir / "experiment_elf_provenance.jsonl"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in sorted(
            records,
            key=lambda value: (
                value["program"],
                value["compiler"],
                value["program_optimization"],
            ),
        ):
            cache_path, metadata = entries[record["binary_sha256"]]
            binary = dataset_root / record["binary"]
            ground_truth = dataset_root / record["ground_truth"]
            linker_map = dataset_root / record["linker_map"]
            if not all(path.is_file() for path in (binary, ground_truth, linker_map)):
                raise FileNotFoundError(f"missing experiment artifact for {binary}")
            if binary.stat().st_size != int(metadata["identity"]["binary_size"]):
                raise ValueError(f"binary/cache size mismatch: {binary}")
            if sha256(binary) != record["binary_sha256"]:
                raise ValueError(f"binary hash mismatch: {binary}")
            linker_map_sha256 = sha256(linker_map)
            expected_map_sha256 = record.get("linker_map_sha256")
            if expected_map_sha256 and linker_map_sha256 != expected_map_sha256:
                raise ValueError(f"linker-map hash mismatch: {linker_map}")
            cache_key = cache_path.name.removesuffix(".numpy")
            experiment_record = {
                "schema": 1,
                "program": record["program"],
                "source_project": record["source_project"],
                "source_version": record["source_version"],
                "compiler": record["compiler"],
                "compiler_family": record["compiler_family"],
                "program_optimization": record["program_optimization"],
                "program_optimization_flags": record.get(
                    "program_optimization_flags", {}
                ),
                "library_optimizations": record.get("library_optimizations", {}),
                "library_versions": record.get("library_versions", {}),
                "library_version_roles": record.get(
                    "library_version_roles", {}
                ),
                "link_type": record.get("link_type"),
                "binary": record["binary"],
                "binary_sha256": record["binary_sha256"],
                "ground_truth": record["ground_truth"],
                "ground_truth_sha256": sha256(ground_truth),
                "ground_truth_complete": bool(record["ground_truth_complete"]),
                "linker_map": record["linker_map"],
                "linker_map_sha256": linker_map_sha256,
                "cache_key": cache_key,
                "cache_entry": cache_path.relative_to(cache_dir).as_posix(),
                "cache_identity": metadata["identity"],
                "cache_counts": metadata["counts"],
                "cache_matching_features": metadata["matching_features"],
            }
            stream.write(
                json.dumps(experiment_record, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, output)

    summary = {
        "schema": 1,
        "records": len(records),
        "unique_binary_hashes": len(wanted_hashes),
        "programs": len({record["program"] for record in records}),
        "compilers": dict(sorted(Counter(r["compiler"] for r in records).items())),
        "optimizations": dict(
            sorted(Counter(r["program_optimization"] for r in records).items())
        ),
        "ground_truth_complete": sum(
            bool(record["ground_truth_complete"]) for record in records
        ),
        "source_manifest": source_manifest.as_posix(),
        "output": output.as_posix(),
    }
    (cache_dir / "experiment_elf_provenance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
