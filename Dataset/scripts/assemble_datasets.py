#!/usr/bin/env python3
"""Assemble the two ELF datasets and exact linker-map ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ground_truth import build_ground_truth, sha256


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
REPO_DIR = DATASET_DIR.parent
DEFAULT_ARTIFACT_ROOT = DATASET_DIR
DEFAULT_LIBSEEKER_BUILD_ROOT = DEFAULT_ARTIFACT_ROOT / "builds/programs"
DEFAULT_UNSEEN_BUILD_ROOT = DEFAULT_LIBSEEKER_BUILD_ROOT
DEFAULT_INVENTORY = DATASET_DIR / "manifests/libseeker_inventory.json"
DEFAULT_UNSEEN_INVENTORY = DATASET_DIR / "manifests/unseen_glibc_inventory.json"
DEFAULT_PLAN = DATASET_DIR / "manifests/dataset_plan.json"
DEFAULT_SOURCE_MANIFEST = DATASET_DIR / "manifests/source_manifest.json"

PROJECT_SOURCE_NAMES = {
    "bash": "bash-5.3-beta",
    "coreutils": "coreutils-9.6",
    "gawk": "gawk-5.3.2",
    "gzip": "gzip-1.13",
    "gnuchess": "gnuchess-6.2.11",
    "grep": "grep-3.11",
    "inetutils": "inetutils-2.6",
    "less": "less-668",
    "make": "make-4.4.1",
    "nano": "nano-8.3",
    "openssh": "openssh-portable-V_10_0_P2",
    "rsync": "rsync-3.4.1",
    "sed": "sed-4.9",
    "socat": "socat-1.8.0.3",
    "tar": "tar-1.35",
    "util-linux": "util-linux-v2.39.3",
    "vim": "vim-v9.1.1151",
    "wget2": "wget2-2.2.0",
}

PRUNED_DIRECTORIES = {
    ".git", ".deps", ".dirstamp", "autom4te.cache", "doc", "docs",
    "fuzz", "gnulib-tests", "man", "po", "test", "tests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("libseeker", "unseen"), required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="copy",
        help=(
            "Materialize an immutable snapshot with regular copies (default); "
            "hard links are available only as an explicit space-saving option."
        ),
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit-programs", type=int,
        help="Assemble only the first N inventory programs (smoke tests).",
    )
    parser.add_argument(
        "--compiler",
        action="append",
        help=(
            "Restrict the matrix to one or more compiler commands from the plan. "
            "This is used to materialize independently mergeable shards."
        ),
    )
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.plan = args.plan.resolve()
    if args.profile == "libseeker":
        args.build_root = (args.build_root or DEFAULT_LIBSEEKER_BUILD_ROOT).resolve()
        args.inventory = (args.inventory or DEFAULT_INVENTORY).resolve()
    else:
        args.build_root = (args.build_root or DEFAULT_UNSEEN_BUILD_ROOT).resolve()
        args.inventory = (args.inventory or DEFAULT_UNSEEN_INVENTORY).resolve()
    repo_root = REPO_DIR.resolve()
    dataset_root = DATASET_DIR.resolve()
    artifact_root = args.artifact_root

    inside_repository = (
        artifact_root == repo_root
        or repo_root in artifact_root.parents
    )
    inside_dataset = (
        artifact_root == dataset_root
        or dataset_root in artifact_root.parents
    )

    if (
        artifact_root in {
            Path("/").resolve(),
            Path.home().resolve(),
            repo_root,
        }
        or artifact_root in repo_root.parents
        or (inside_repository and not inside_dataset)
    ):
        parser.error(
            "--artifact-root must be Dataset, a directory below Dataset, "
            f"or a dedicated directory outside the repository: {artifact_root}"
        )
    return args


def compiler_family(value: str) -> str:
    name = Path(value).name
    if name.startswith("gcc") or name.startswith("g++"):
        return "gcc"
    if name.startswith("clang"):
        return "clang"
    return name.split("-", 1)[0]


def compiler_command(info: dict[str, Any]) -> str:
    metadata = info.get("compiler_metadata")
    if isinstance(metadata, dict):
        command = metadata.get("command") or metadata.get("path")
        if isinstance(command, str) and command:
            return Path(command).name
    compiler = str(info.get("compiler", ""))
    name = Path(compiler).name
    match = re.match(r"^(gcc|clang)-(\d+)-\d+(?:\.\d+)+$", name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return name


def normalized_compiler(info: Any) -> tuple[str, str, dict[str, Any]]:
    """Return a path-safe compiler id, family, and preserved metadata."""
    if isinstance(info, dict):
        details = dict(info)
        command = str(details.get("command") or details.get("path") or "")
        family = compiler_family(command)
        version_text = str(details.get("version", ""))
        version_match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", version_text)
        identifier = f"{family}-{version_match.group(1)}" if version_match else family
        return identifier, family, details
    value = str(info or "")
    family = compiler_family(value)
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "-", Path(value).name or family).strip("-")
    return safe or family, family, {"command": value}


def elf_link_type(path: Path) -> str | None:
    """Return ``static``/``dynamic`` for executable ELF files."""
    try:
        with path.open("rb") as stream:
            ident = stream.read(16)
            if len(ident) != 16 or ident[:4] != b"\x7fELF":
                return None
            elf_class, byte_order = ident[4], ident[5]
            if elf_class not in (1, 2) or byte_order not in (1, 2):
                return None
            endian = "<" if byte_order == 1 else ">"
            header_size = 52 if elf_class == 1 else 64
            header = ident + stream.read(header_size - 16)
            if len(header) != header_size:
                return None
            e_type = struct.unpack_from(endian + "H", header, 16)[0]
            if elf_class == 1:
                phoff = struct.unpack_from(endian + "I", header, 28)[0]
                phentsize = struct.unpack_from(endian + "H", header, 42)[0]
                phnum = struct.unpack_from(endian + "H", header, 44)[0]
            else:
                phoff = struct.unpack_from(endian + "Q", header, 32)[0]
                phentsize = struct.unpack_from(endian + "H", header, 54)[0]
                phnum = struct.unpack_from(endian + "H", header, 56)[0]
            if e_type not in (2, 3) or not phoff or not phentsize:
                return None
            dynamic_segments: list[tuple[int, int]] = []
            stream.seek(phoff)
            for _ in range(phnum):
                program_header = stream.read(phentsize)
                if len(program_header) != phentsize:
                    return None
                program_type = struct.unpack_from(endian + "I", program_header, 0)[0]
                if program_type == 3:
                    return "dynamic"
                if program_type == 2:
                    if elf_class == 1:
                        offset = struct.unpack_from(endian + "I", program_header, 4)[0]
                        size = struct.unpack_from(endian + "I", program_header, 16)[0]
                    else:
                        offset = struct.unpack_from(endian + "Q", program_header, 8)[0]
                        size = struct.unpack_from(endian + "Q", program_header, 32)[0]
                    dynamic_segments.append((offset, size))
            dynamic_entry_size = 8 if elf_class == 1 else 16
            dynamic_tag_format = endian + ("i" if elf_class == 1 else "q")
            for offset, size in dynamic_segments:
                stream.seek(offset)
                for _ in range(size // dynamic_entry_size):
                    entry = stream.read(dynamic_entry_size)
                    if len(entry) != dynamic_entry_size:
                        return None
                    tag = struct.unpack_from(dynamic_tag_format, entry, 0)[0]
                    if tag == 0:
                        break
                    if tag == 1:  # DT_NEEDED
                        return "dynamic"
            if e_type == 3 and (".so" in path.name or not os.access(path, os.X_OK)):
                return None
            return "static"
    except (OSError, struct.error):
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_plan(path: Path, profile: str) -> dict[str, Any]:
    payload = read_json(path)
    return payload["datasets"][profile]


def restrict_plan_compilers(
    plan: dict[str, Any], compilers: list[str] | None
) -> dict[str, Any]:
    if not compilers:
        return plan
    requested = set(compilers)
    available = {
        str(cell.get("compiler"))
        for cell in plan.get("matrix", [])
        if isinstance(cell, dict)
    }
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"compiler not present in dataset plan: {unknown}")
    restricted = dict(plan)
    restricted["matrix"] = [
        dict(cell)
        for cell in plan.get("matrix", [])
        if str(cell.get("compiler")) in requested
    ]
    restricted["matrix_scope"] = {
        "kind": "compiler_subset",
        "compilers": sorted(requested),
    }
    return restricted


def load_libseeker_inventory(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = read_json(path)
    source_entries = {
        entry["name"]: entry
        for entry in read_json(DEFAULT_SOURCE_MANIFEST)["entries"]
    }
    records: list[dict[str, Any]] = []
    for project in payload["projects"]:
        reproduction = source_entries[PROJECT_SOURCE_NAMES[project["id"]]]
        metadata = project.get("program_metadata", {})
        for name in project["programs"]:
            extra = metadata.get(name, {})
            records.append({
                "program": name,
                "source_basename": extra.get("source_basename", name),
                "alias_of": extra.get("alias_of"),
                "source_project": project["id"],
                "source_project_name": project["source_project"],
                "source_version": project["source_version"],
                "source_url": reproduction["url"],
                "source_name": reproduction["name"],
                "source_sha256": reproduction.get("sha256"),
                "source_revision": reproduction.get("revision"),
                "original_libseeker_url": project["source_url"],
                "compiler_support": list(project["compiler_support"]),
            })
    if limit is not None:
        allowed = {entry["program"] for entry in records[:limit]}
        records = [entry for entry in records if entry["program"] in allowed]
    return records


def load_unseen_inventory(path: Path, limit: int | None) -> list[str]:
    names = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return names[:limit] if limit is not None else names


def expected_cells(
    plan: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> set[tuple[str, str, str]]:
    cells = set()
    ordered_names = [entry["program"] for entry in inventory]
    by_name = {entry["program"]: entry for entry in inventory}
    for cell in plan["matrix"]:
        compiler = str(cell["compiler"])
        family = str(cell.get("compiler_family") or compiler_family(compiler))
        count = min(int(cell.get("program_count", len(ordered_names))), len(ordered_names))
        offset = int(cell.get("cohort_offset", 0)) % max(1, len(ordered_names))
        selected = (
            [ordered_names[(offset + index) % len(ordered_names)] for index in range(count)]
            if ordered_names else []
        )
        for name in selected:
            if family in by_name[name]["compiler_support"]:
                cells.add((name, compiler, cell["program_optimization"]))
    return cells


def walk_executables(root: Path) -> Iterable[Path]:
    for current, directories, files in os.walk(root):
        directories[:] = [
            name for name in directories if name not in PRUNED_DIRECTORIES
        ]
        base = Path(current)
        for name in files:
            path = base / name
            if path.is_file() and os.access(path, os.X_OK):
                yield path


def candidate_score(path: Path, build_dir: Path) -> tuple[int, int, str]:
    relative = path.relative_to(build_dir)
    parts = relative.parts
    penalty = len(parts)
    lowered = {part.lower() for part in parts[:-1]}
    if lowered & PRUNED_DIRECTORIES:
        penalty += 1000
    if parts and parts[0] == "src":
        penalty -= 20
    if ".libs" in parts:
        penalty -= 5
    if len(parts) == 1:
        penalty -= 10
    return penalty, len(str(relative)), str(relative)


def adjacent_map(binary: Path) -> Path | None:
    candidate = Path(f"{binary}.map")
    return candidate if candidate.is_file() else None


def adjacent_link_json(binary: Path) -> Path | None:
    candidate = Path(f"{binary}.link.json")
    return candidate if candidate.is_file() else None


def linker_artifacts(binary: Path) -> tuple[Path | None, Path | None]:
    """Resolve the wrapper sidecar and its map, including explicit -Map paths."""
    link_json = adjacent_link_json(binary)
    linker_map = adjacent_map(binary)
    if linker_map is None and link_json is not None:
        try:
            payload = read_json(link_json)
            value = payload.get("map") or payload.get("linker_map")
            candidate = Path(str(value)) if value else None
            if candidate is not None and candidate.is_file():
                linker_map = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return linker_map, link_json


def discover_libseeker(
    root: Path,
    inventory: list[dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in inventory:
        by_project[entry["source_project"]].append(entry)
    wanted = expected_cells(plan, inventory)
    best: dict[tuple[str, str, str], tuple[tuple[int, int, str], dict[str, Any]]] = {}
    diagnostics: list[str] = []

    info_files = sorted(root.glob("*/*/build-info.json"))
    if not info_files:
        return [], [f"no build-info.json found below {root}"]

    # randomized_matrix.json is the authoritative resume snapshot.  Old
    # compiler-version directories intentionally remain in the build cache,
    # but must not compete with the variants selected by the latest matrix.
    matrix_path = root / "randomized_matrix.json"
    allowed_variants: set[tuple[str, str, str, str]] | None = None
    if matrix_path.is_file():
        try:
            matrix_payload = read_json(matrix_path)
            matrix_cases = matrix_payload.get("cases", [])
            if not isinstance(matrix_cases, list):
                raise ValueError("cases is not a list")
            allowed_variants = {
                (
                    str(case["program"]),
                    str(case["compiler"]),
                    str(case["elf_optimization"]),
                    str(case["variant"]),
                )
                for case in matrix_cases
                if isinstance(case, dict)
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            return [], [f"invalid authoritative matrix {matrix_path}: {error}"]

    for info_path in info_files:
        info = read_json(info_path)
        project = str(info["program"])
        if project not in by_project:
            continue
        family = compiler_family(str(info["compiler"]))
        command = compiler_command(info)
        optimization = str(info["elf_optimization"])
        variant = str(info.get("variant", info_path.parent.name))
        if allowed_variants is not None and (
            project,
            str(info["compiler"]),
            optimization,
            variant,
        ) not in allowed_variants:
            continue
        project_inventory = [
            entry for entry in by_project[project]
            if (entry["program"], command, optimization) in wanted
        ]
        if not project_inventory:
            continue
        build_dir = info_path.parent / "build"
        if not build_dir.is_dir():
            diagnostics.append(f"missing build directory: {build_dir}")
            continue

        candidates = []
        map_by_hash: dict[str, tuple[Path, Path, Path | None]] = {}
        for binary in walk_executables(build_dir):
            if elf_link_type(binary) != "static":
                continue
            digest = sha256(binary)
            map_path, link_json = linker_artifacts(binary)
            candidates.append((binary, digest, map_path, link_json))
            if map_path is not None:
                map_by_hash.setdefault(digest, (binary, map_path, link_json))

        for entry in project_inventory:
            basename = entry["source_basename"]
            matching = [item for item in candidates if item[0].name == basename]
            for binary, digest, map_path, link_json in matching:
                if map_path is None and digest in map_by_hash:
                    _, map_path, fallback_json = map_by_hash[digest]
                    link_json = link_json or fallback_json
                if map_path is None:
                    continue
                key = (entry["program"], command, optimization)
                record = {
                    **entry,
                    "compiler": str(info["compiler"]),
                    "compiler_metadata": info.get("compiler_metadata"),
                    "compiler_family": family,
                    "compiler_command": command,
                    "program_optimization": optimization,
                    "program_optimization_flags": info.get(
                        "program_optimization_flags"
                    ),
                    "source_binary": str(binary.resolve()),
                    "source_map": str(map_path.resolve()),
                    "source_link_json": str(link_json.resolve()) if link_json else None,
                    "sha256": digest,
                    "link_type": "static",
                    "library_optimizations": info.get("seeded_library_selection", {}),
                    "library_versions": info.get(
                        "seeded_library_version_selection", {}
                    ),
                    "library_version_roles": info.get(
                        "seeded_library_version_roles", {}
                    ),
                    "archive_metadata": info.get("passed_archives", []),
                    "build_info": str(info_path.resolve()),
                    "build_dir": str(build_dir.resolve()),
                }
                score = candidate_score(binary, build_dir)
                previous = best.get(key)
                if previous is None or score < previous[0]:
                    best[key] = (score, record)

    records = [value[1] for value in best.values()]
    records.sort(key=lambda item: (
        item["program"], item["compiler_command"], item["program_optimization"]
    ))
    missing = sorted(wanted - set(best))
    diagnostics.extend(
        f"missing cell {program}/{compiler}/{optimization}"
        for program, compiler, optimization in missing
    )
    return records, diagnostics


def discover_unseen(
    root: Path,
    names: list[str],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read normalized per-command metadata emitted by the Toybox builder."""
    expected = {
        (name, cell["compiler"], cell["program_optimization"], cell["library_optimization"])
        for name in names for cell in plan["matrix"]
    }
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    diagnostics: list[str] = []
    metadata_files = sorted(root.rglob("binary-info/*.json"))
    if not metadata_files:
        metadata_files = sorted(root.rglob("*.binary.json"))
    if not metadata_files:
        metadata_files = sorted(root.rglob("binary-info.json"))
    for path in metadata_files:
        info = read_json(path)
        if info.get("kind") not in (None, "toybox_binary_build"):
            continue
        program = str(info.get("program") or info.get("command"))
        compiler, inferred_family, compiler_metadata = normalized_compiler(
            info.get("compiler", "")
        )
        family = str(info.get("compiler_family") or inferred_family)
        command = Path(str(compiler_metadata.get("command") or "")).name
        program_opt = str(info.get("program_optimization") or info.get("optimization"))
        library_opt = str(info.get("library_optimization") or info.get("musl_optimization"))
        key = (program, command, program_opt, library_opt)
        if key not in expected:
            continue
        binary = Path(info.get("binary") or info.get("output", ""))
        linker_map = Path(info.get("linker_map") or f"{binary}.map")
        link_json_value = info.get("link_json") or info.get("link_command_json")
        link_json = Path(link_json_value) if link_json_value else adjacent_link_json(binary)
        if not binary.is_file() or elf_link_type(binary) != "static":
            diagnostics.append(f"invalid static ELF for {key}: {binary}")
            continue
        if not linker_map.is_file():
            diagnostics.append(f"missing linker map for {key}: {linker_map}")
            continue
        musl = info.get("musl") if isinstance(info.get("musl"), dict) else {}
        source = info.get("source") if isinstance(info.get("source"), dict) else {}
        archive = info.get("libc_archive") or info.get("archive") or musl.get("archive")
        archive_metadata = info.get("archive_metadata") or ([{
            "library": "musl",
            "source": f"musl-{musl.get('version', '1.2.6')}",
            "source_url": (
                musl.get("source", {}).get("url")
                if isinstance(musl.get("source"), dict)
                else None
            ),
            "source_revision": (
                musl.get("source", {}).get("git_revision")
                if isinstance(musl.get("source"), dict)
                else None
            ),
            "optimization": library_opt,
            "compiler": compiler,
            "archive": archive,
        }] if archive else [])
        best[key] = {
            "program": program,
            "source_basename": program,
            "alias_of": None,
            "source_project": "toybox",
            "source_project_name": "Toybox",
            "source_version": str(info.get("source_version") or source.get("version") or "0.8.14"),
            "source_url": str(info.get("source_url") or source.get("url") or "https://landley.net/toybox/"),
            "source_name": f"toybox-{source.get('version', '0.8.14')}",
            "source_sha256": source.get("sha256"),
            "source_revision": source.get("git_revision"),
            "original_libseeker_url": None,
            "compiler": compiler,
            "compiler_metadata": compiler_metadata,
            "compiler_family": family,
            "compiler_command": command,
            "program_optimization": program_opt,
            "program_optimization_flags": info.get("program_optimize_flags"),
            "standalone_configuration": info.get(
                "standalone_configuration"
            ),
            "library_optimization": library_opt,
            "library_optimizations": {"musl": library_opt},
            "source_binary": str(binary.resolve()),
            "source_map": str(linker_map.resolve()),
            "source_link_json": str(link_json.resolve()) if link_json and link_json.is_file() else None,
            "sha256": sha256(binary),
            "link_type": "static",
            "archive_metadata": archive_metadata,
            "build_info": str(path.resolve()),
            "build_dir": str(
                info.get("build_dir")
                or source.get("isolated_build_tree")
                or binary.parent
            ),
        }
    missing = sorted(expected - set(best))
    diagnostics.extend(
        f"missing cell {program}/{compiler}/{program_opt}/lib-{library_opt}"
        for program, compiler, program_opt, library_opt in missing
    )
    records = sorted(best.values(), key=lambda item: (
        item["program"], item["compiler_command"], item["program_optimization"]
    ))
    return records, diagnostics


def safe_clean(path: Path, artifact_root: Path, dry_run: bool) -> None:
    resolved = path.resolve()
    root = artifact_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"refusing to clean outside artifact root {root}: {resolved}")
    if not dry_run:
        shutil.rmtree(resolved, ignore_errors=True)


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise FileExistsError(f"different file already exists: {destination}")
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def safe_component(value: Any, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "\x00" in component
    ):
        raise ValueError(f"unsafe {label} path component: {component!r}")
    return component


def legacy_summaries(archives: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cu_summary = {
        "method": "linker_map",
        "included_compilation_units": sum(
            int(archive.get("included_compilation_units", 0) or 0)
            for archive in archives
        ),
        "total_compilation_units": sum(
            int(archive.get("total_compilation_units", 0) or 0)
            for archive in archives
        ),
        "map_archives": sum(
            archive.get("cu_ground_truth_method") == "linker_map"
            for archive in archives
        ),
        "empty_archives": sum(
            archive.get("cu_ground_truth_method") == "none"
            for archive in archives
        ),
    }
    libraries: dict[str, dict[str, Any]] = {}
    for archive in archives:
        library = str(archive.get("library") or archive.get("archive_basename") or "")
        if not library:
            continue
        entry = libraries.setdefault(library, {
            "present": False,
            "archives": 0,
            "included_archives": 0,
            "included_compilation_units": 0,
            "total_compilation_units": 0,
            "sources": [],
            "optimizations": [],
        })
        included = int(archive.get("included_compilation_units", 0) or 0)
        entry["archives"] += 1
        entry["included_archives"] += int(included > 0)
        entry["included_compilation_units"] += included
        entry["total_compilation_units"] += int(
            archive.get("total_compilation_units", 0) or 0
        )
        entry["present"] = entry["present"] or included > 0
        for field, target in (("source", "sources"), ("optimization", "optimizations")):
            value = archive.get(field)
            if value and value not in entry[target]:
                entry[target].append(value)
    library_summary = {
        "method": "linker_map_included_compilation_units",
        "present_libraries": sorted(
            name for name, entry in libraries.items() if entry["present"]
        ),
        "absent_libraries": sorted(
            name for name, entry in libraries.items() if not entry["present"]
        ),
        "libraries": libraries,
    }
    return cu_summary, library_summary


def add_pipeline_metadata(
    payload: dict[str, Any], record: dict[str, Any], variant: str
) -> None:
    cu_summary, library_summary = legacy_summaries(payload["archives"])
    payload.update({
        "program": record["program"],
        "variant": variant,
        "compiler": record["compiler"],
        "elf_optimization": record["program_optimization"],
        "link_type": "static",
        "seeded_library_selection": record["library_optimizations"],
        "seeded_library_version_selection": record.get("library_versions", {}),
        "seeded_library_version_roles": record.get(
            "library_version_roles", {}
        ),
        "source": record.get("source_url"),
        "libs": sorted(record["library_optimizations"]),
        "cu_ground_truth": cu_summary,
        "library_ground_truth": library_summary,
    })


def render_ground_truth_text(payload: dict[str, Any], binary: Path) -> str:
    lines = [
        f"# Binary: {binary.resolve()}",
        f"# {payload.get('program')} optimization: -{payload.get('elf_optimization')}",
        f"# compiler: {payload.get('compiler')}",
        "# Source: GNU ld linker map; excluded CUs are archive members not extracted by the linker.",
        "",
    ]
    for archive in payload["archives"]:
        basename = archive.get("archive_basename") or Path(str(archive["archive"])).name
        source = archive.get("source") or archive.get("library") or "unknown"
        optimization = archive.get("optimization") or "unknown"
        lines.append(f"LIBRARY: {basename}.{source}-{optimization}")
        lines.append(f"  Optimization: {optimization}")
        lines.append(f"  Archive: {archive['archive']}")
        members = list(archive.get("included_members", []))
        lines.append(f"  Included compilation units: {len(members)}")
        for member_name in members:
            lines.append(f"    {member_name}")
        lines.append("")
    return "\n".join(lines)


def materialize_records(
    profile: str,
    records: list[dict[str, Any]],
    dataset_root: Path,
    ground_truth_root: Path,
    copy_mode: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    result = []
    artifact_root = dataset_root.parent.parent.resolve()
    def portable(path: Path) -> str:
        return str(path.resolve().relative_to(artifact_root))

    catalog_root = ground_truth_root / "archive_catalog"
    for record in records:
        program = safe_component(record["program"], "program")
        compiler = safe_component(record["compiler"], "compiler")
        variant = safe_component(record["program_optimization"], "optimization")
        if record.get("library_optimization"):
            library_optimization = safe_component(
                record["library_optimization"], "library optimization"
            )
            variant += f"__lib-{library_optimization}"
        relative = Path(
            program, compiler, variant, program
        )
        pipeline_variant = safe_component(
            f"{program}_{compiler}_{variant}", "pipeline variant"
        )
        binary_destination = dataset_root / "binaries" / relative
        gt_directory = ground_truth_root / "binaries" / relative.parent
        gt_json = gt_directory / "ground_truth.json"
        gt_text = gt_directory / "ground_truth.txt"
        map_destination = gt_directory / "linker.map"
        link_json_destination = gt_directory / "link.json"

        updated = {key: value for key, value in record.items() if key != "archive_metadata"}
        updated.update({
            "binary": portable(binary_destination),
            "ground_truth": portable(gt_json),
            "linker_map": portable(map_destination),
        })
        if not dry_run:
            materialize(Path(record["source_binary"]), binary_destination, copy_mode)
            gt_directory.mkdir(parents=True, exist_ok=True)
            materialize(Path(record["source_map"]), map_destination, copy_mode)
            link_payload: dict[str, Any] = {}
            if record.get("source_link_json"):
                source_link_json = Path(record["source_link_json"])
                materialize(source_link_json, link_json_destination, copy_mode)
                link_payload = read_json(source_link_json)
            resolution_roots = [Path(record["build_dir"])]
            if link_payload.get("cwd"):
                resolution_roots.append(Path(link_payload["cwd"]))
            build_metadata = {
                "profile": profile,
                "program": record["program"],
                "source_project": record["source_project"],
                "source_version": record["source_version"],
                "source_url": record["source_url"],
                "source_name": record.get("source_name"),
                "source_sha256": record.get("source_sha256"),
                "source_revision": record.get("source_revision"),
                "original_libseeker_url": record.get(
                    "original_libseeker_url"
                ),
                "compiler": record["compiler"],
                "compiler_family": record["compiler_family"],
                "compiler_metadata": record.get("compiler_metadata"),
                "program_optimization": record["program_optimization"],
                "program_optimization_flags": record.get(
                    "program_optimization_flags"
                ),
                "standalone_configuration": record.get(
                    "standalone_configuration"
                ),
                "library_optimizations": record["library_optimizations"],
                "library_versions": record.get("library_versions", {}),
                "library_version_roles": record.get(
                    "library_version_roles", {}
                ),
                "build_info": record["build_info"],
                "link": link_payload,
            }
            ground_truth = build_ground_truth(
                binary=binary_destination,
                linker_map=map_destination,
                archive_metadata=record["archive_metadata"],
                catalog_root=catalog_root,
                build_metadata=build_metadata,
                resolution_roots=resolution_roots,
            )
            add_pipeline_metadata(ground_truth, record, pipeline_variant)
            rendered_text = render_ground_truth_text(ground_truth, binary_destination)
            ground_truth["binary"] = portable(binary_destination)
            ground_truth["linker_map"] = portable(map_destination)
            for archive in ground_truth["archives"]:
                catalog = archive.get("archive_catalog")
                if catalog:
                    archive["archive_catalog"] = portable(Path(str(catalog)))
            gt_json.write_text(json.dumps(ground_truth, indent=2, sort_keys=True) + "\n")
            gt_text.write_text(rendered_text)
            updated["ground_truth_complete"] = ground_truth["summary"]["ground_truth_complete"]
            updated["binary_sha256"] = ground_truth["binary_sha256"]
            updated["linker_map_sha256"] = ground_truth["linker_map_sha256"]
        result.append(updated)
    return result


def deduplicate_records_by_sha256(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep one deterministic representative for each ELF SHA-256."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        groups[str(record["sha256"])].append(record)

    keep_ids: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}

    for digest, group in groups.items():
        ordered = sorted(
            group,
            key=lambda record: (
                str(record.get("program", "")),
                str(
                    record.get(
                        "compiler_command",
                        record.get("compiler", ""),
                    )
                ),
                str(record.get("program_optimization", "")),
                str(record.get("source_binary", "")),
            ),
        )

        # La selezione è deterministica, ma non favorisce sempre
        # il compilatore lessicograficamente precedente.
        chosen = ordered[int(digest[:16], 16) % len(ordered)]
        chosen_copy = dict(chosen)

        duplicates = [
            record for record in ordered
            if record is not chosen
        ]

        if duplicates:
            chosen_copy["equivalent_builds"] = [
                {
                    "program": record.get("program"),
                    "compiler": record.get("compiler"),
                    "compiler_command": record.get(
                        "compiler_command"
                    ),
                    "program_optimization": record.get(
                        "program_optimization"
                    ),
                    "library_optimizations": record.get(
                        "library_optimizations"
                    ),
                    "build_info": record.get("build_info"),
                }
                for record in duplicates
            ]
            chosen_copy["deduplicated_group_size"] = len(ordered)

        keep_ids.add(id(chosen))
        replacements[id(chosen)] = chosen_copy

    deduplicated = [
        replacements[id(record)]
        for record in records
        if id(record) in keep_ids
    ]

    return deduplicated, len(records) - len(deduplicated)


def library_version_role_counts(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        for library, role in record.get("library_version_roles", {}).items():
            library_counts = counts[str(library)]
            role = str(role)
            library_counts[role] = library_counts.get(role, 0) + 1
    return {
        library: dict(sorted(library_counts.items()))
        for library, library_counts in sorted(counts.items())
    }


def write_manifests(
    profile: str,
    records: list[dict[str, Any]],
    diagnostics: list[str],
    dataset_root: Path,
    ground_truth_root: Path,
    expected_count: int,
    candidate_count: int,
    duplicates_removed: int,
    deduplicated: bool,
    matrix_scope: dict[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    summary = {
        "binaries": len(records),
        "candidate_binaries": candidate_count,
        "duplicates_removed": duplicates_removed,
        "programs": len({record["program"] for record in records}),
        "source_projects": len({record["source_project"] for record in records}),
        "compilers": sorted({record["compiler"] for record in records}),
        "program_optimizations": sorted({record["program_optimization"] for record in records}),
        "library_optimizations": sorted({
            optimization
            for record in records
            for optimization in record["library_optimizations"].values()
        }),
        "unique_binary_hashes": len({record.get("binary_sha256", record["sha256"]) for record in records}),
        "expected_binaries": expected_count,
        "diagnostics": len(diagnostics),
        "library_version_role_counts": library_version_role_counts(records),
    }
    if deduplicated:
        summary["deduplicated_by"] = "binary_sha256"
        summary["representative_policy"] = "sha256_deterministic_index"
    if matrix_scope:
        summary["matrix_scope"] = matrix_scope
    payload = {
        "schema_version": 2,
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "diagnostics": diagnostics,
        "records": records,
    }
    if dry_run:
        print(json.dumps({"profile": profile, "summary": summary, "diagnostics": diagnostics[:20]}, indent=2))
        return payload

    dataset_root.mkdir(parents=True, exist_ok=True)
    ground_truth_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (ground_truth_root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = (
        "program", "source_project", "source_version", "compiler",
        "compiler_family", "program_optimization", "binary_sha256",
        "binary", "ground_truth", "linker_map", "ground_truth_complete",
    )
    with (dataset_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return payload


def main() -> int:
    args = parse_args()
    plan = restrict_plan_compilers(
        load_plan(args.plan, args.profile), args.compiler
    )
    dataset_root = args.artifact_root / "datasets" / args.profile
    ground_truth_root = args.artifact_root / "ground_truth" / args.profile
    # Assembly is a snapshot operation. Always remove the two profile outputs
    # first so a compiler/version change cannot leave unmanifested stale ELF/GT.
    if not args.dry_run:
        safe_clean(dataset_root, args.artifact_root, args.dry_run)
        safe_clean(ground_truth_root, args.artifact_root, args.dry_run)

    inventory = load_libseeker_inventory(args.inventory, args.limit_programs)
    records, diagnostics = discover_libseeker(args.build_root, inventory, plan)
    candidate_count = len(records)
    deduplicated = plan.get("deduplicate_binary_sha256") is True
    if deduplicated:
        records, duplicates_removed = deduplicate_records_by_sha256(records)
    else:
        duplicates_removed = 0
    expected_count = len(expected_cells(plan, inventory))

    materialized = materialize_records(
        args.profile, records, dataset_root, ground_truth_root,
        args.copy_mode, args.dry_run,
    )
    payload = write_manifests(
        args.profile, materialized, diagnostics, dataset_root,
        ground_truth_root, expected_count, candidate_count,
        duplicates_removed, deduplicated, plan.get("matrix_scope"), args.dry_run,
    )
    problems = list(diagnostics)
    if candidate_count != expected_count:
        problems.append(
            f"expected {expected_count} candidate binaries, "
            f"found {candidate_count}"
        )
    incomplete = [record["program"] for record in materialized if record.get("ground_truth_complete") is False]
    if incomplete:
        problems.append(f"{len(incomplete)} records have unresolved archive ground truth")
    print(
        f"{args.profile}: {payload['summary']['binaries']} ELF, "
        f"{payload['summary']['programs']} programs -> {dataset_root}"
    )
    for problem in problems[:50]:
        print(f"WARNING: {problem}", file=sys.stderr)
    if len(problems) > 50:
        print(f"WARNING: {len(problems) - 50} additional problems", file=sys.stderr)
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
