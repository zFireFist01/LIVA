#!/usr/bin/env python3
"""Create portable LibSeeker experiment archives for PC2, PC3 and PC4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PC_TO_SHARD = {2: 1, 3: 2, 4: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc", type=int, choices=PC_TO_SHARD, required=True)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def selected_libraries(repository: Path) -> list[Path]:
    matrix = repository / "Dataset/manifests/library_matrix.tsv"
    library_root = repository / "Dataset/builds/libraries"
    with matrix.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    paths = [library_root / row["path"] for row in rows if row["status"] == "selected"]
    if len(paths) != 4472 or len(set(paths)) != 4472:
        raise RuntimeError(
            f"expected 4472 distinct selected libraries, found {len(set(paths))}"
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing selected library: {missing[0]}")
    return sorted(paths)


def validate_shard(shard_root: Path, expected_shard: int) -> dict[str, object]:
    inventory_path = shard_root / "cache/shard_inventory.json"
    with inventory_path.open(encoding="utf-8") as handle:
        inventory = json.load(handle)
    if inventory.get("valid") is not True or inventory.get("errors") != []:
        raise RuntimeError(f"shard is not valid: {inventory_path}")
    if inventory.get("shard_index") != expected_shard:
        raise RuntimeError(f"unexpected shard index in {inventory_path}")
    if inventory.get("elf_records") != 876:
        raise RuntimeError(f"unexpected ELF count in {inventory_path}")
    if inventory.get("library_archives_cached") != 4472:
        raise RuntimeError(f"unexpected library count in {inventory_path}")
    return inventory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    shard = PC_TO_SHARD[args.pc]
    shard_root = repository / f"Dataset/shards/libseeker-shard-{shard}"
    inventory = validate_shard(shard_root, shard)
    libraries = selected_libraries(repository)

    exports = repository / "Dataset/exports"
    exports.mkdir(parents=True, exist_ok=True)
    archive = exports / f"libseeker-PC{args.pc}-{inventory['compiler']}-experiments.tar.zst"
    temporary = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists() and not args.replace:
        raise FileExistsError(f"output already exists (use --replace): {archive}")
    temporary.unlink(missing_ok=True)

    optional_docs = (
        repository / "Dataset/SHARDED_LIBSEEKER.md",
        repository / "Dataset/exports/README-EXPERIMENT-PACKAGE.md",
    )
    inputs: list[Path] = [
        shard_root / "cache",
        shard_root / "datasets",
        shard_root / "ground_truth",
        shard_root / "pipeline.exit",
        repository / "Dataset/manifests",
        repository / "Dataset/scripts",
        repository / "Dataset/CACHE_MMAP.md",
        *(path for path in optional_docs if path.is_file()),
        repository / "Dataset/exports/python-environment.txt",
        repository / "thesis_code",
        *sorted((shard_root / "builds").rglob("build-info.json")),
        *libraries,
    ]
    relative_inputs = [str(path.relative_to(repository)) for path in inputs]
    prefix = f"LIVA-PC{args.pc}/"
    tar_command = [
        "tar",
        "--create",
        "--file=-",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--checkpoint=100000",
        f"--checkpoint-action=echo=PC{args.pc}: archived %u entries",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=locks",
        "--transform=s|^Dataset/exports/README-EXPERIMENT-PACKAGE.md$|README.md|",
        "--transform=s|^Dataset/exports/python-environment.txt$|python-environment.txt|",
        # Prefix archive member names, not relative symbolic-link targets.
        # Rewriting targets breaks aliases such as libncurses.a -> libncursesw.a.
        f"--transform=flags=rh;s|^|{prefix}|",
        "--",
        *relative_inputs,
    ]
    zstd_command = [
        "zstd",
        f"-{args.compression_level}",
        f"-T{args.threads}",
        "-f",
        "-o",
        str(temporary),
        "-",
    ]
    print(
        f"Creating {archive.name}: PC{args.pc}, shard {shard}, "
        f"{inventory['compiler']}, 876 ELF, 4472 libraries",
        flush=True,
    )
    with subprocess.Popen(tar_command, stdout=subprocess.PIPE) as tar_process:
        assert tar_process.stdout is not None
        zstd_result = subprocess.run(zstd_command, stdin=tar_process.stdout)
        tar_process.stdout.close()
        tar_status = tar_process.wait()
    if tar_status or zstd_result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"archive failed: tar={tar_status}, zstd={zstd_result.returncode}"
        )
    temporary.replace(archive)
    digest = sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
