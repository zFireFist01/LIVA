#!/usr/bin/env python3
"""Assemble validated static-ELF datasets from the project build trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from datetime import datetime, timezone
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
REPO_DIR = DATASET_DIR.parent
DEFAULT_ELF_ROOT = DATASET_DIR / "builds/elf_builds"
DEFAULT_TOOL_ROOT = DATASET_DIR / "builds/unseen_program_builds"
DEFAULT_REFERENCE = DATASET_DIR / "manifests/libseeker_programs.txt"
DEFAULT_OUTPUT_ROOT = DATASET_DIR / "datasets"
OPTIMIZATIONS = {"O0", "O2", "O3", "Os", "Oz", "Ofast"}

# LibSeeker used a few output labels that differ from the upstream filename.
# A single compiled artifact can therefore intentionally populate two labels.
LIBSEEKER_ALIASES = {
    "gawk": ("gawk-5.3.2",),
    "ginstall": ("install",),
    "socat": ("socat1",),
    "wget2": ("wget2_noinstall",),
    "losetup": ("losetup.static",),
    "mount": ("mount.static",),
    "nsenter": ("nsenter.static",),
    "umount": ("umount.static",),
    "unshare": ("unshare.static",),
}

PRUNED_BUILD_DIRS = {
    ".git",
    ".deps",
    ".libs",
    "autom4te.cache",
    "build-aux",
    "doc",
    "docs",
    "gnulib-tests",
    "man",
    "po",
    "tests",
    "test-suite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect static ELF files, reject LibSeeker overlap for the unseen "
            "profile, and emit JSON/CSV provenance manifests."
        )
    )
    parser.add_argument("--profile", required=True, choices=("libseeker", "unseen"))
    parser.add_argument("--elf-build-root", type=Path, default=DEFAULT_ELF_ROOT)
    parser.add_argument("--tool-build-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--reference-names", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--optimization",
        action="append",
        help="Optimization to retain; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--compiler",
        action="append",
        help="Compiler id/prefix to retain, for example gcc or clang-22.1.6.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize binaries (default: copy).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only the selected assembled output before collecting.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if the LibSeeker inventory is incomplete, an unseen name "
            "overlaps it, or no binaries are collected."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    values = []
    for raw in args.optimization or []:
        values.extend(part.strip().removeprefix("-") for part in raw.split(","))
    args.optimizations = {value for value in values if value}
    invalid = args.optimizations - OPTIMIZATIONS
    if invalid:
        parser.error(f"invalid optimizations: {', '.join(sorted(invalid))}")

    args.compilers = tuple(args.compiler or ())
    args.elf_build_root = args.elf_build_root.resolve()
    args.tool_build_root = args.tool_build_root.resolve()
    args.reference_names = args.reference_names.resolve()
    args.output = (
        args.output.resolve()
        if args.output
        else (DEFAULT_OUTPUT_ROOT / args.profile).resolve()
    )
    return args


def load_reference_names(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"LibSeeker inventory not found: {path}")
    names = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    invalid = sorted(name for name in names if "/" in name or name in {".", ".."})
    if invalid:
        raise ValueError(f"invalid program names in {path}: {invalid}")
    return names


def compiler_selected(value: str, prefixes: tuple[str, ...]) -> bool:
    return not prefixes or any(
        value == prefix or value.startswith(f"{prefix}-")
        for prefix in prefixes
    )


def compiler_family(compiler: str) -> str:
    for family in ("gcc", "clang"):
        if compiler == family or compiler.startswith(f"{family}-"):
            return family
    return compiler


def matrix_coverage(
    records: list[dict],
    expected_names: set[str],
) -> tuple[dict[str, dict], list[str]]:
    by_cell: dict[tuple[str, str], set[str]] = {}
    for record in records:
        key = (
            compiler_family(str(record["compiler"])),
            str(record["optimization"]),
        )
        by_cell.setdefault(key, set()).add(str(record["program"]))

    cells = {}
    incomplete = []
    for (family, optimization), names in sorted(by_cell.items()):
        label = f"{family}/{optimization}"
        missing = sorted(expected_names - names)
        cells[label] = {
            "programs": len(names),
            "missing_programs": missing,
        }
        if missing:
            incomplete.append(label)
    return cells, incomplete


def optimization_selected(value: str, selected: set[str]) -> bool:
    return not selected or value in selected


def elf_link_type(path: Path) -> str | None:
    """Return static for executable ELF files without PT_INTERP."""
    try:
        with path.open("rb") as stream:
            ident = stream.read(16)
            if len(ident) != 16 or ident[:4] != b"\x7fELF":
                return None
            elf_class = ident[4]
            byte_order = ident[5]
            if elf_class not in (1, 2) or byte_order not in (1, 2):
                return None
            endian = "<" if byte_order == 1 else ">"
            header_size = 52 if elf_class == 1 else 64
            rest = stream.read(header_size - 16)
            if len(rest) != header_size - 16:
                return None
            header = ident + rest
            if elf_class == 1:
                e_type = struct.unpack_from(endian + "H", header, 16)[0]
                phoff = struct.unpack_from(endian + "I", header, 28)[0]
                phentsize = struct.unpack_from(endian + "H", header, 42)[0]
                phnum = struct.unpack_from(endian + "H", header, 44)[0]
            else:
                e_type = struct.unpack_from(endian + "H", header, 16)[0]
                phoff = struct.unpack_from(endian + "Q", header, 32)[0]
                phentsize = struct.unpack_from(endian + "H", header, 54)[0]
                phnum = struct.unpack_from(endian + "H", header, 56)[0]
            if e_type not in (2, 3) or not phoff or not phentsize:
                return None
            stream.seek(phoff)
            for _ in range(phnum):
                program_header = stream.read(phentsize)
                if len(program_header) != phentsize:
                    return None
                p_type = struct.unpack_from(endian + "I", program_header, 0)[0]
                if p_type == 3:  # PT_INTERP
                    return "dynamic"
            # ET_DYN without PT_INTERP is commonly a shared object. Only accept
            # it when the source is executable and does not look like a library.
            if e_type == 3 and (
                ".so" in path.name or not os.access(path, os.X_OK)
            ):
                return None
            return "static"
    except (OSError, struct.error):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_value_file(path: Path) -> dict[str, str]:
    result = {}
    if not path.is_file():
        return result
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def build_candidate_score(path: Path, build_dir: Path) -> tuple[int, int, str]:
    relative = path.relative_to(build_dir)
    parts = relative.parts
    penalty = len(parts)
    lowered = {part.lower() for part in parts[:-1]}
    if lowered & {"test", "tests", "gnulib-tests", "examples", "fuzz"}:
        penalty += 100
    if parts and parts[0] == "src":
        penalty -= 20
    if len(parts) == 1:
        penalty -= 10
    return penalty, len(str(relative)), str(relative)


def walk_build_binaries(build_dir: Path) -> Iterable[Path]:
    for current, directories, files in os.walk(build_dir):
        directories[:] = [
            directory
            for directory in directories
            if directory not in PRUNED_BUILD_DIRS
        ]
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if path.is_file() and os.access(path, os.X_OK):
                yield path


def local_build_info_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"ELF build root not found: {root}")
    return sorted(root.glob("*/randomized/*/build-info.json"))


def discover_libseeker(
    root: Path,
    reference_names: set[str],
    optimizations: set[str],
    compiler_prefixes: tuple[str, ...],
) -> tuple[list[dict], dict]:
    best: dict[tuple[str, str, str], tuple[tuple[int, int, str], dict]] = {}
    rejected_dynamic = 0
    variants = 0

    for info_path in local_build_info_files(root):
        info = json.loads(info_path.read_text())
        project = str(info["program"])
        compiler = str(info["compiler"])
        optimization = str(info["elf_optimization"])
        if not compiler_selected(compiler, compiler_prefixes):
            continue
        if not optimization_selected(optimization, optimizations):
            continue
        build_dir = info_path.parent / "build"
        if not build_dir.is_dir():
            continue
        variants += 1
        primary = info_path.parent / project
        primary_hash = sha256(primary) if primary.is_file() else None
        linker_map = info_path.parent / f"{project}.map"

        for candidate in walk_build_binaries(build_dir):
            labels = []
            if candidate.name in reference_names:
                labels.append(candidate.name)
            labels.extend(
                alias
                for alias in LIBSEEKER_ALIASES.get(candidate.name, ())
                if alias in reference_names
            )
            if not labels:
                continue
            link_type = elf_link_type(candidate)
            if link_type != "static":
                rejected_dynamic += 1
                continue
            digest = sha256(candidate)
            score = build_candidate_score(candidate, build_dir)
            for label in labels:
                key = (compiler, optimization, label)
                record = {
                    "program": label,
                    "source_project": project,
                    "compiler": compiler,
                    "optimization": optimization,
                    "source_binary": str(candidate.resolve()),
                    "sha256": digest,
                    "link_type": "static",
                    "linker_map": (
                        str(linker_map.resolve())
                        if linker_map.is_file() and digest == primary_hash
                        else None
                    ),
                    "candidate_archives": info.get("passed_archives", []),
                    "archive_ground_truth_scope": (
                        "exact_primary"
                        if linker_map.is_file() and digest == primary_hash
                        else "project_candidates_only"
                    ),
                }
                previous = best.get(key)
                if previous is None or score < previous[0]:
                    best[key] = (score, record)

    records = [entry[1] for entry in best.values()]
    records.sort(
        key=lambda item: (
            item["program"],
            item["compiler"],
            item["optimization"],
        )
    )
    found_names = {record["program"] for record in records}
    cells, incomplete_cells = matrix_coverage(records, reference_names)
    coverage = {
        "reference_programs": len(reference_names),
        "found_programs": len(found_names),
        "missing_programs": sorted(reference_names - found_names),
        "unexpected_programs": sorted(found_names - reference_names),
        "matrix_cells": cells,
        "incomplete_matrix_cells": incomplete_cells,
        "build_variants_scanned": variants,
        "dynamic_candidates_rejected": rejected_dynamic,
    }
    return records, coverage


def installed_program_dirs(variant_dir: Path) -> Iterable[Path]:
    install_dir = variant_dir / "install"
    for relative in ("bin", "sbin", "libexec"):
        directory = install_dir / relative
        if directory.is_dir():
            yield directory


def discover_unseen(
    root: Path,
    reference_names: set[str],
    optimizations: set[str],
    compiler_prefixes: tuple[str, ...],
) -> tuple[list[dict], dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"unseen build root not found: {root}")
    records = []
    overlaps = set()
    rejected_dynamic = 0
    seen_paths = set()

    for compiler_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        compiler = compiler_dir.name
        if not compiler_selected(compiler, compiler_prefixes):
            continue
        for project_dir in sorted(path for path in compiler_dir.iterdir() if path.is_dir()):
            for variant_dir in sorted(
                path for path in project_dir.iterdir() if path.is_dir()
            ):
                optimization = variant_dir.name
                if not optimization_selected(optimization, optimizations):
                    continue
                build_info = read_key_value_file(variant_dir / "build-info.txt")
                for program_dir in installed_program_dirs(variant_dir):
                    for candidate in sorted(program_dir.rglob("*")):
                        if not candidate.is_file() or candidate in seen_paths:
                            continue
                        seen_paths.add(candidate)
                        link_type = elf_link_type(candidate)
                        if link_type is None:
                            continue
                        if candidate.name in reference_names:
                            overlaps.add(candidate.name)
                            continue
                        if link_type != "static":
                            rejected_dynamic += 1
                            continue
                        records.append({
                            "program": candidate.name,
                            "source_project": project_dir.name,
                            "compiler": compiler,
                            "optimization": optimization,
                            "source_binary": str(candidate.resolve()),
                            "sha256": sha256(candidate),
                            "link_type": "static",
                            "linker_map": None,
                            "candidate_archives": [],
                            "archive_ground_truth_scope": "not_available",
                            "source": build_info.get("source"),
                        })

    records.sort(
        key=lambda item: (
            item["source_project"],
            item["program"],
            item["compiler"],
            item["optimization"],
        )
    )
    names = {record["program"] for record in records}
    cells, incomplete_cells = matrix_coverage(records, names)
    coverage_by_name: dict[str, dict[str, list[str]]] = {}
    for name in sorted(names):
        matching = [record for record in records if record["program"] == name]
        coverage_by_name[name] = {
            "compilers": sorted({record["compiler"] for record in matching}),
            "optimizations": sorted({
                record["optimization"] for record in matching
            }),
        }
    coverage = {
        "programs": len(names),
        "source_projects": len({record["source_project"] for record in records}),
        "overlap_with_libseeker": sorted(names & reference_names),
        "excluded_libseeker_names": sorted(overlaps),
        "dynamic_candidates_rejected": rejected_dynamic,
        "matrix_cells": cells,
        "incomplete_matrix_cells": incomplete_cells,
        "program_matrix": coverage_by_name,
    }
    return records, coverage


def safe_clean_output(output: Path, dry_run: bool) -> None:
    resolved = output.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(
            f"refusing to clean output outside {allowed}: {resolved}"
        )
    if not dry_run:
        shutil.rmtree(resolved, ignore_errors=True)


def materialize(
    records: list[dict],
    output: Path,
    copy_mode: str,
    dry_run: bool,
) -> list[dict]:
    materialized = []
    for record in records:
        source = Path(record["source_binary"])
        destination = (
            output
            / "binaries"
            / record["source_project"]
            / record["program"]
            / record["compiler"]
            / record["optimization"]
            / record["program"]
        )
        updated = dict(record)
        updated["binary"] = str(destination.resolve())
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256(destination) != record["sha256"]:
                    raise FileExistsError(
                        f"different binary already exists: {destination}"
                    )
            elif copy_mode == "hardlink":
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)
            if record.get("linker_map"):
                map_destination = destination.with_suffix(
                    destination.suffix + ".map"
                )
                shutil.copy2(record["linker_map"], map_destination)
                updated["linker_map"] = str(map_destination.resolve())
        materialized.append(updated)
    return materialized


def write_manifests(
    profile: str,
    records: list[dict],
    coverage: dict,
    reference: Path,
    output: Path,
    dry_run: bool,
) -> None:
    unique_hashes = {record["sha256"] for record in records}
    payload = {
        "schema": 1,
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_inventory": str(reference),
        "selection_policy": (
            "name_in_libseeker_inventory_and_static_elf"
            if profile == "libseeker"
            else "installed_static_elf_and_name_not_in_libseeker_inventory"
        ),
        "summary": {
            "binaries": len(records),
            "programs": len({record["program"] for record in records}),
            "source_projects": len({
                record["source_project"] for record in records
            }),
            "compilers": sorted({record["compiler"] for record in records}),
            "optimizations": sorted({
                record["optimization"] for record in records
            }),
            "unique_contents": len(unique_hashes),
        },
        "coverage": coverage,
        "records": records,
    }
    if dry_run:
        print(json.dumps({key: payload[key] for key in ("profile", "summary", "coverage")}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    csv_fields = (
        "program",
        "source_project",
        "compiler",
        "optimization",
        "link_type",
        "sha256",
        "binary",
        "source_binary",
        "linker_map",
        "archive_ground_truth_scope",
    )
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Dataset: {output}")
    print(
        f"Binaries={len(records)} "
        f"programs={payload['summary']['programs']} "
        f"unique_contents={len(unique_hashes)}"
    )


def main() -> int:
    args = parse_args()
    reference_names = load_reference_names(args.reference_names)
    if args.clean:
        safe_clean_output(args.output, args.dry_run)

    if args.profile == "libseeker":
        records, coverage = discover_libseeker(
            args.elf_build_root,
            reference_names,
            args.optimizations,
            args.compilers,
        )
    else:
        records, coverage = discover_unseen(
            args.tool_build_root,
            reference_names,
            args.optimizations,
            args.compilers,
        )

    materialized = materialize(
        records,
        args.output,
        args.copy_mode,
        args.dry_run,
    )
    write_manifests(
        args.profile,
        materialized,
        coverage,
        args.reference_names,
        args.output,
        args.dry_run,
    )

    problems = []
    if not records:
        problems.append("no static ELF files collected")
    if args.profile == "libseeker":
        if coverage["unexpected_programs"]:
            problems.append("unexpected LibSeeker program names")
        if coverage["missing_programs"]:
            problems.append(
                f"{len(coverage['missing_programs'])} LibSeeker programs missing"
            )
    elif coverage["overlap_with_libseeker"]:
        problems.append("unseen dataset overlaps LibSeeker")
    if coverage["incomplete_matrix_cells"]:
        problems.append(
            "incomplete compiler/optimization cells: "
            + ", ".join(coverage["incomplete_matrix_cells"])
        )

    for problem in problems:
        print(f"WARNING: {problem}", file=sys.stderr)
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
