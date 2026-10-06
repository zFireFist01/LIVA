#!/usr/bin/env python3
"""Validate the final LibSeeker and unseen ELF dataset artifacts.

The validator is intentionally self-contained and uses only the Python
standard library.  With no ``--profile`` argument it also enforces the
cross-dataset disjointness guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ground_truth import parse_linker_map


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
REPO_DIR = DATASET_DIR.parent
DEFAULT_ARTIFACT_ROOT = REPO_DIR.parent / f"{REPO_DIR.name}_artifacts"
DEFAULT_PLAN = DATASET_DIR / "manifests/dataset_plan.json"
DEFAULT_SOURCE_MANIFEST = DATASET_DIR / "manifests/source_manifest.json"

PROJECT_SOURCE_NAMES = {
    "bash": "bash-5.3-beta", "coreutils": "coreutils-9.6",
    "gawk": "gawk-5.3.2", "gzip": "gzip-1.13",
    "gnuchess": "gnuchess-6.2.11",
    "grep": "grep-3.11", "inetutils": "inetutils-2.6",
    "less": "less-668", "make": "make-4.4.1", "nano": "nano-8.3",
    "openssh": "openssh-portable-V_10_0_P2", "rsync": "rsync-3.4.1",
    "sed": "sed-4.9", "socat": "socat-1.8.0.3", "tar": "tar-1.35",
    "util-linux": "util-linux-v2.39.3", "vim": "vim-v9.1.1151",
    "wget2": "wget2-2.2.0",
}

PROFILES = ("libseeker", "unseen")
CANONICAL_COUNTS = {
    "libseeker": (3504, 219),
    "unseen": (1200, 75),
}
GROUND_TRUTH_METHOD = "gnu_linker_map_archive_members"
GROUND_TRUTH_SCOPE = "exact"
PT_INTERP = 3
PT_DYNAMIC = 2
DT_NEEDED = 1
EM_X86_64 = 62


@dataclass
class Reporter:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class ProfileResult:
    profile: str
    records: list[dict[str, Any]]
    programs: set[str]
    hashes: dict[str, list[str]]
    matrix_keys: set[tuple[str, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"artifact tree (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help=f"dataset plan (default: {DEFAULT_PLAN})",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        help="validate one profile only; by default both are validated",
    )
    parser.add_argument(
        "--compiler",
        action="append",
        help=(
            "Validate only matrix cells for one or more compiler commands. "
            "Requires --profile and is intended for a 1/4 shard."
        ),
    )
    parser.add_argument(
        "--allow-hash-overlap",
        action="store_true",
        help="report cross-dataset SHA-256 overlap as a warning, not an error",
    )
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.plan = args.plan.resolve()
    if args.compiler and args.profile != "libseeker":
        parser.error("--compiler currently requires --profile libseeker")
    return args


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def read_json(path: Path, reporter: Reporter, label: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        reporter.error(f"{label}: file not found: {path}")
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        reporter.error(f"{label}: cannot read valid JSON from {path}: {error}")
        return None
    if not isinstance(payload, dict):
        reporter.error(f"{label}: top-level JSON value must be an object: {path}")
        return None
    return payload


def resolve_declared_path(
    value: Any,
    *,
    manifest_dir: Path,
    artifact_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = []
    if path.parts and path.parts[0] in {"datasets", "ground_truth"}:
        candidates.append(artifact_root / path)
    candidates.extend((manifest_dir / path, artifact_root / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def inspect_static_x86_64_elf(path: Path) -> tuple[bool, str]:
    """Check ELF64/x86-64 and reject any PT_INTERP program header."""
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
            if len(header) != 64 or header[:4] != b"\x7fELF":
                return False, "not an ELF file"
            if header[4] != 2:
                return False, "not ELF64"
            if header[5] == 1:
                endian = "<"
            elif header[5] == 2:
                endian = ">"
            else:
                return False, "invalid ELF byte order"
            e_type, e_machine = struct.unpack_from(endian + "HH", header, 16)
            if e_machine != EM_X86_64:
                return False, f"ELF machine is {e_machine}, expected x86-64 ({EM_X86_64})"
            if e_type not in (2, 3):
                return False, f"ELF type is {e_type}, expected ET_EXEC or ET_DYN"
            e_phoff = struct.unpack_from(endian + "Q", header, 32)[0]
            e_phentsize = struct.unpack_from(endian + "H", header, 54)[0]
            e_phnum = struct.unpack_from(endian + "H", header, 56)[0]
            if not e_phoff or not e_phnum:
                return False, "ELF has no program headers"
            if e_phnum == 0xFFFF:
                return False, "extended program-header count is unsupported"
            if e_phentsize < 56:
                return False, f"invalid ELF program-header size {e_phentsize}"
            dynamic_segments: list[tuple[int, int]] = []
            stream.seek(e_phoff)
            for _ in range(e_phnum):
                program_header = stream.read(e_phentsize)
                if len(program_header) != e_phentsize:
                    return False, "truncated ELF program-header table"
                p_type = struct.unpack_from(endian + "I", program_header, 0)[0]
                if p_type == PT_INTERP:
                    return False, "contains PT_INTERP (dynamically linked)"
                if p_type == PT_DYNAMIC:
                    p_offset = struct.unpack_from(endian + "Q", program_header, 8)[0]
                    p_filesz = struct.unpack_from(endian + "Q", program_header, 32)[0]
                    dynamic_segments.append((p_offset, p_filesz))
            for p_offset, p_filesz in dynamic_segments:
                stream.seek(p_offset)
                for _ in range(p_filesz // 16):
                    dynamic_entry = stream.read(16)
                    if len(dynamic_entry) != 16:
                        return False, "truncated PT_DYNAMIC segment"
                    dynamic_tag = struct.unpack_from(endian + "q", dynamic_entry, 0)[0]
                    if dynamic_tag == 0:
                        break
                    if dynamic_tag == DT_NEEDED:
                        return False, "contains DT_NEEDED (dynamically linked)"
    except (OSError, struct.error) as error:
        return False, f"cannot parse ELF: {error}"
    return True, "static ELF64 x86-64"


def compiler_family(record: dict[str, Any]) -> str:
    explicit = record.get("compiler_family")
    if isinstance(explicit, str) and explicit:
        return explicit
    compiler = Path(str(record.get("compiler", ""))).name
    if compiler.startswith(("gcc", "g++")):
        return "gcc"
    if compiler.startswith("clang"):
        return "clang"
    return compiler.split("-", 1)[0]


def compiler_command(record: dict[str, Any]) -> str:
    explicit = record.get("compiler_command")
    if isinstance(explicit, str) and explicit:
        return Path(explicit).name
    metadata = record.get("compiler_metadata")
    if isinstance(metadata, dict):
        command = metadata.get("command") or metadata.get("path")
        if isinstance(command, str) and command:
            return Path(command).name
    return Path(str(record.get("compiler", ""))).name


def matrix_key(profile: str, record: dict[str, Any]) -> tuple[str, ...] | None:
    program = record.get("program")
    optimization = record.get("program_optimization")
    command = compiler_command(record)
    if not all(isinstance(value, str) and value for value in (program, command, optimization)):
        return None
    return program, command, optimization


def format_key(key: tuple[str, ...]) -> str:
    return "/".join(key)


def report_set_difference(
    reporter: Reporter,
    *,
    profile: str,
    label: str,
    values: Iterable[tuple[str, ...]],
    limit: int = 20,
) -> None:
    ordered = sorted(values)
    if not ordered:
        return
    preview = ", ".join(format_key(value) for value in ordered[:limit])
    suffix = f" (+{len(ordered) - limit} more)" if len(ordered) > limit else ""
    reporter.error(f"{profile}: {label} ({len(ordered)}): {preview}{suffix}")


def load_expected_matrix(
    profile: str,
    profile_plan: dict[str, Any],
    plan_dir: Path,
    reporter: Reporter,
) -> tuple[set[str], set[tuple[str, ...]]]:
    inventory_name = profile_plan.get("inventory")
    if not isinstance(inventory_name, str) or not inventory_name:
        reporter.error(f"{profile}: plan has no inventory path")
        return set(), set()
    inventory_path = Path(inventory_name)
    if not inventory_path.is_absolute():
        inventory_path = plan_dir / inventory_path

    raw_matrix = profile_plan.get("matrix")
    if not isinstance(raw_matrix, list):
        reporter.error(f"{profile}: plan matrix must be a list")
        return set(), set()

    cell_keys = []
    for index, cell in enumerate(raw_matrix):
        if not isinstance(cell, dict):
            reporter.error(f"{profile}: plan matrix cell {index} is not an object")
            continue
        key = (cell.get("compiler"), cell.get("program_optimization"))
        if not all(isinstance(value, str) and value for value in key):
            reporter.error(f"{profile}: incomplete plan matrix cell {index}: {cell!r}")
            continue
        cell_keys.append(key)
    duplicate_cells = [key for key, count in Counter(cell_keys).items() if count > 1]
    report_set_difference(
        reporter,
        profile=profile,
        label="duplicate cells in dataset plan",
        values=duplicate_cells,
    )

    programs: set[str] = set()
    expected: set[tuple[str, ...]] = set()
    inventory = read_json(inventory_path, reporter, f"{profile} inventory")
    if inventory is None:
        return programs, expected
    projects = inventory.get("projects")
    if not isinstance(projects, list):
        reporter.error(f"{profile}: inventory projects must be a list")
        return programs, expected
    ordered_names: list[str] = []
    support_by_name: dict[str, list[str]] = {}
    for project_index, project in enumerate(projects):
        if not isinstance(project, dict):
            reporter.error(f"{profile}: inventory project {project_index} is not an object")
            continue
        names = project.get("programs")
        support = project.get("compiler_support")
        if not isinstance(names, list) or not isinstance(support, list):
            reporter.error(
                f"{profile}: inventory project {project.get('id', project_index)!r} "
                "has invalid programs/compiler_support"
            )
            continue
        for name in names:
            if not isinstance(name, str) or not name:
                reporter.error(f"{profile}: inventory contains an invalid program name")
                continue
            if name in programs:
                reporter.error(f"{profile}: duplicate inventory program {name!r}")
            programs.add(name)
            ordered_names.append(name)
            support_by_name[name] = support
    for cell in raw_matrix:
        if not isinstance(cell, dict) or not ordered_names:
            continue
        compiler = str(cell.get("compiler", ""))
        family = str(cell.get("compiler_family") or compiler.split("-", 1)[0])
        optimization = str(cell.get("program_optimization", ""))
        count = min(int(cell.get("program_count", len(ordered_names))), len(ordered_names))
        offset = int(cell.get("cohort_offset", 0)) % len(ordered_names)
        selected = [ordered_names[(offset + index) % len(ordered_names)] for index in range(count)]
        for name in selected:
            if family in support_by_name[name]:
                expected.add((name, compiler, optimization))
    return programs, expected


def validate_catalog_entry(
    *,
    profile: str,
    record_label: str,
    archive_index: int,
    archive: dict[str, Any],
    ground_truth_root: Path,
    artifact_root: Path,
    reporter: Reporter,
    json_cache: dict[Path, dict[str, Any] | None],
    hash_cache: dict[Path, str],
) -> int:
    context = f"{profile}:{record_label}: archive[{archive_index}]"
    included = archive.get("included_members")
    if not isinstance(included, list) or not all(isinstance(item, str) for item in included):
        reporter.error(f"{context}: included_members must be a list of strings")
        included = []
    if archive.get("included_member_count") != len(included):
        reporter.error(f"{context}: included_member_count does not match included_members")
    if archive.get("included_compilation_units") != len(included):
        reporter.error(f"{context}: legacy included_compilation_units is inconsistent")
    if archive.get("confirmed_compilation_units") != included:
        reporter.error(f"{context}: legacy confirmed_compilation_units is inconsistent")
    expected_method = "linker_map" if included else "none"
    if archive.get("cu_ground_truth_method") != expected_method:
        reporter.error(f"{context}: cu_ground_truth_method is inconsistent")
    if archive.get("confirmed_compilation_units_method") != expected_method:
        reporter.error(f"{context}: confirmed CU method is inconsistent")
    if archive.get("resolved") is not True:
        reporter.error(f"{context}: archive is not marked resolved")

    archive_digest = archive.get("archive_sha256")
    if not valid_sha256(archive_digest):
        reporter.error(f"{context}: invalid or missing archive_sha256")
        archive_digest = None

    catalog_path = resolve_declared_path(
        archive.get("archive_catalog"),
        manifest_dir=ground_truth_root,
        artifact_root=artifact_root,
    )
    if catalog_path is None:
        reporter.error(f"{context}: missing archive_catalog path")
        return len(included)
    if not is_below(catalog_path, ground_truth_root):
        reporter.error(f"{context}: archive catalog is outside {ground_truth_root}: {catalog_path}")
    if not catalog_path.is_file():
        reporter.error(f"{context}: archive catalog does not exist: {catalog_path}")
        return len(included)
    if archive_digest and catalog_path.name != f"{archive_digest}.json":
        reporter.error(
            f"{context}: catalog filename {catalog_path.name!r} does not match archive SHA-256"
        )

    if catalog_path not in json_cache:
        json_cache[catalog_path] = read_json(catalog_path, reporter, f"{context} catalog")
    catalog = json_cache[catalog_path]
    if catalog is None:
        return len(included)
    if catalog.get("archive_sha256") != archive_digest:
        reporter.error(f"{context}: catalog archive_sha256 does not match ground truth")
    if catalog.get("schema_version") != 4:
        reporter.error(f"{context}: archive catalog schema is not minimal occurrence-aware v4")
    members = catalog.get("members")
    if not isinstance(members, list):
        reporter.error(f"{context}: catalog members must be a list")
        members = []
    if archive.get("total_compilation_units") != len(members):
        reporter.error(f"{context}: legacy total_compilation_units is inconsistent")
    catalog_member_names = {
        item.get("name") for item in members
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    catalog_names = [
        item.get("name") for item in members
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    duplicate_catalog_names = sorted(
        name for name, count in Counter(catalog_names).items() if count > 1
    )
    included_counts = Counter(included)
    catalog_counts = Counter(catalog_names)
    over_selected = sorted(
        name for name, count in included_counts.items()
        if count > catalog_counts.get(name, 0)
    )
    if over_selected:
        reporter.error(
            f"{context}: linker map selects more member occurrences than catalog: "
            + ", ".join(over_selected[:20])
        )
    catalog_summary = catalog.get("summary")
    duplicate_metadata = (
        catalog_summary.get("duplicate_member_names", {})
        if isinstance(catalog_summary, dict)
        else {}
    )
    for name in duplicate_catalog_names:
        metadata = duplicate_metadata.get(name)
        hashes = metadata.get("object_sha256", []) if isinstance(metadata, dict) else []
        metadata_invalid = (
            not isinstance(metadata, dict)
            or metadata.get("occurrences") != catalog_names.count(name)
            or len(hashes) != catalog_names.count(name)
        )
        if metadata_invalid:
            reporter.error(f"{context}: duplicate member {name!r} metadata is invalid")
        elif included_counts.get(name, 0) and (
            metadata.get("byte_identical") is not True or len(set(hashes)) != 1
        ):
            reporter.error(f"{context}: duplicate member {name!r} is not byte-equivalent")
    if set(duplicate_metadata) != set(duplicate_catalog_names):
        reporter.error(f"{context}: duplicate-member catalog metadata is inconsistent")
    if archive.get("archive_member_names_unique") != (not duplicate_catalog_names):
        reporter.error(f"{context}: archive_member_names_unique is inconsistent")
    if archive.get("archive_member_identity_exact") is not True:
        reporter.error(f"{context}: archive member identity is not exact")
    missing_members = sorted(set(included) - catalog_member_names)
    if missing_members:
        preview = ", ".join(missing_members[:10])
        suffix = f" (+{len(missing_members) - 10} more)" if len(missing_members) > 10 else ""
        reporter.error(
            f"{context}: {len(missing_members)} included members absent from catalog: "
            f"{preview}{suffix}"
        )
    if isinstance(catalog_summary, dict) and catalog_summary.get("members") != len(members):
        reporter.error(f"{context}: catalog summary member count is inconsistent")
    if archive.get("archive_member_count") != len(members):
        reporter.error(f"{context}: archive_member_count does not match catalog")

    archive_path_value = archive.get("archive")
    if isinstance(archive_path_value, str):
        archive_path = Path(archive_path_value)
        if archive_path.is_absolute() and archive_path.is_file() and archive_digest:
            if archive_path not in hash_cache:
                hash_cache[archive_path] = sha256(archive_path)
            if hash_cache[archive_path] != archive_digest:
                reporter.error(f"{context}: source archive content does not match archive_sha256")
    return len(included)


def validate_ground_truth(
    *,
    profile: str,
    record: dict[str, Any],
    record_label: str,
    binary_path: Path,
    actual_binary_sha: str,
    dataset_manifest_dir: Path,
    ground_truth_root: Path,
    artifact_root: Path,
    reporter: Reporter,
    json_cache: dict[Path, dict[str, Any] | None],
    hash_cache: dict[Path, str],
) -> None:
    gt_path = resolve_declared_path(
        record.get("ground_truth"),
        manifest_dir=dataset_manifest_dir,
        artifact_root=artifact_root,
    )
    map_path = resolve_declared_path(
        record.get("linker_map"),
        manifest_dir=dataset_manifest_dir,
        artifact_root=artifact_root,
    )
    context = f"{profile}:{record_label}"
    if gt_path is None:
        reporter.error(f"{context}: record has no ground_truth path")
        return
    if map_path is None:
        reporter.error(f"{context}: record has no linker_map path")
        return
    if not is_below(gt_path, ground_truth_root):
        reporter.error(f"{context}: ground truth is outside {ground_truth_root}: {gt_path}")
    if not is_below(map_path, ground_truth_root):
        reporter.error(f"{context}: linker map is outside {ground_truth_root}: {map_path}")
    if not map_path.is_file():
        reporter.error(f"{context}: linker map does not exist: {map_path}")
        return
    if not gt_path.is_file():
        reporter.error(f"{context}: ground-truth JSON does not exist: {gt_path}")
        return

    if map_path not in hash_cache:
        hash_cache[map_path] = sha256(map_path)
    actual_map_sha = hash_cache[map_path]
    record_map_sha = record.get("linker_map_sha256")
    if not valid_sha256(record_map_sha):
        reporter.error(f"{context}: record has invalid or missing linker_map_sha256")
    elif record_map_sha != actual_map_sha:
        reporter.error(f"{context}: linker map SHA-256 differs from manifest")

    if gt_path not in json_cache:
        json_cache[gt_path] = read_json(gt_path, reporter, f"{context} ground truth")
    ground_truth = json_cache[gt_path]
    if ground_truth is None:
        return
    if ground_truth.get("method") != GROUND_TRUTH_METHOD:
        reporter.error(
            f"{context}: ground-truth method is {ground_truth.get('method')!r}, "
            f"expected {GROUND_TRUTH_METHOD!r}"
        )
    if ground_truth.get("scope") != GROUND_TRUTH_SCOPE:
        reporter.error(
            f"{context}: ground-truth scope is {ground_truth.get('scope')!r}, "
            f"expected {GROUND_TRUTH_SCOPE!r}"
        )
    if "inter_cu_ground_truth" in ground_truth:
        reporter.error(
            f"{context}: call-graph diagnostics must not be embedded in the "
            "library/CU ground truth"
        )
    build = ground_truth.get("build")
    if not isinstance(build, dict):
        reporter.error(f"{context}: ground-truth build provenance must be an object")
        build = {}
    build_field_map = {
        "profile": profile,
        "program": record.get("program"),
        "source_project": record.get("source_project"),
        "source_version": record.get("source_version"),
        "source_url": record.get("source_url"),
        "source_name": record.get("source_name"),
        "source_sha256": record.get("source_sha256"),
        "source_revision": record.get("source_revision"),
        "compiler": record.get("compiler"),
        "compiler_family": compiler_family(record),
        "compiler_metadata": record.get("compiler_metadata"),
        "program_optimization": record.get("program_optimization"),
        "program_optimization_flags": record.get("program_optimization_flags"),
        "library_optimizations": record.get("library_optimizations"),
        "library_versions": record.get("library_versions", {}),
        "library_version_roles": record.get("library_version_roles", {}),
    }
    for field_name, expected_value in build_field_map.items():
        if build.get(field_name) != expected_value:
            reporter.error(
                f"{context}: build.{field_name} does not match manifest record"
            )
    pipeline_metadata_fields = {
        "program": record.get("program"),
        "compiler": record.get("compiler"),
        "elf_optimization": record.get("program_optimization"),
        "link_type": "static",
        "seeded_library_selection": record.get("library_optimizations"),
        "seeded_library_version_selection": record.get("library_versions", {}),
        "seeded_library_version_roles": record.get(
            "library_version_roles", {}
        ),
    }
    for field_name, expected_value in pipeline_metadata_fields.items():
        if ground_truth.get(field_name) != expected_value:
            reporter.error(
                f"{context}: pipeline metadata field {field_name} does not match manifest"
            )
    if not isinstance(ground_truth.get("variant"), str):
        reporter.error(f"{context}: pipeline variant is missing")
    link = build.get("link")
    if not isinstance(link, dict) or not isinstance(link.get("command"), list):
        reporter.error(f"{context}: exact link-command provenance is missing")
    elif not all(link.get(field) for field in ("compiler", "cwd", "map", "output")):
        reporter.error(f"{context}: link-command provenance is incomplete")
    if ground_truth.get("binary_sha256") != actual_binary_sha:
        reporter.error(f"{context}: ground-truth binary_sha256 does not match ELF")
    if ground_truth.get("linker_map_sha256") != actual_map_sha:
        reporter.error(f"{context}: ground-truth linker_map_sha256 does not match map")

    declared_binary = resolve_declared_path(
        ground_truth.get("binary"),
        manifest_dir=gt_path.parent,
        artifact_root=artifact_root,
    )
    if declared_binary != binary_path:
        reporter.error(f"{context}: ground-truth binary path does not match manifest")
    declared_map = resolve_declared_path(
        ground_truth.get("linker_map"),
        manifest_dir=gt_path.parent,
        artifact_root=artifact_root,
    )
    if declared_map != map_path:
        reporter.error(f"{context}: ground-truth linker-map path does not match manifest")

    summary = ground_truth.get("summary")
    if not isinstance(summary, dict):
        reporter.error(f"{context}: ground-truth summary must be an object")
        summary = {}
    unresolved = summary.get("unresolved_archives")
    if unresolved != []:
        reporter.error(f"{context}: unresolved_archives must be empty, found {unresolved!r}")
    if summary.get("ground_truth_complete") is not True:
        reporter.error(f"{context}: ground_truth_complete is not true")
    if summary.get("ambiguous_archive_member_names") != []:
        reporter.error(
            f"{context}: ambiguous_archive_member_names must be empty, found "
            f"{summary.get('ambiguous_archive_member_names')!r}"
        )
    if summary.get("missing_catalog_members") != []:
        reporter.error(
            f"{context}: missing_catalog_members must be empty, found "
            f"{summary.get('missing_catalog_members')!r}"
        )
    if record.get("ground_truth_complete") is not True:
        reporter.error(f"{context}: manifest record is not marked ground_truth_complete")

    archives = ground_truth.get("archives")
    if not isinstance(archives, list):
        reporter.error(f"{context}: archives must be a list")
        archives = []
    included_total = 0
    ground_truth_map: dict[str, list[str]] = {}
    for index, archive in enumerate(archives):
        if not isinstance(archive, dict):
            reporter.error(f"{context}: archive[{index}] is not an object")
            continue
        unexpected_call_fields = {
            "inter_cu_ground_truth_method",
            "expected_inter_cu_calls",
            "expected_inter_cu_function_edges",
            "expected_inter_cu_call_relocations",
            "expected_inter_cu_cu_edges",
        } & set(archive)
        if unexpected_call_fields:
            reporter.error(
                f"{context}: archive[{index}] contains call-graph diagnostics: "
                f"{sorted(unexpected_call_fields)}"
            )
        raw_archive = archive.get("linker_map_archive")
        selected_by_linker = archive.get("selected_by_linker")
        if selected_by_linker is True:
            if not isinstance(raw_archive, str) or not raw_archive:
                reporter.error(f"{context}: selected archive has no linker_map_archive")
            elif raw_archive in ground_truth_map:
                reporter.error(f"{context}: duplicate linker_map_archive {raw_archive!r}")
            else:
                members = archive.get("included_members")
                ground_truth_map[raw_archive] = (
                    list(members) if isinstance(members, list) else []
                )
        elif selected_by_linker is False:
            if raw_archive is not None or archive.get("included_members") != []:
                reporter.error(f"{context}: negative candidate archive is inconsistent")
        else:
            reporter.error(f"{context}: selected_by_linker must be boolean")
        library = archive.get("library")
        optimization = archive.get("optimization")
        selections = record.get("library_optimizations")
        if library and isinstance(selections, dict) and library in selections:
            if optimization != selections[library]:
                reporter.error(
                    f"{context}: archive[{index}] optimization does not match "
                    f"selection for {library}"
                )
            versions = record.get("library_versions")
            if isinstance(versions, dict) and library in versions:
                if archive.get("source") != versions[library]:
                    reporter.error(
                        f"{context}: archive[{index}] source does not match "
                        f"version selection for {library}"
                    )
            roles = record.get("library_version_roles")
            if isinstance(roles, dict) and library in roles:
                if archive.get("version_role") != roles[library]:
                    reporter.error(
                        f"{context}: archive[{index}] version role does not "
                        f"match selection for {library}"
                    )
        included_total += validate_catalog_entry(
            profile=profile,
            record_label=record_label,
            archive_index=index,
            archive=archive,
            ground_truth_root=ground_truth_root,
            artifact_root=artifact_root,
            reporter=reporter,
            json_cache=json_cache,
            hash_cache=hash_cache,
        )
    if included_total <= 0:
        reporter.error(f"{context}: ground truth contains no included archive member")
    if summary.get("included_archive_members") != included_total:
        reporter.error(f"{context}: summary included_archive_members is inconsistent")
    if summary.get("candidate_archives") != len(archives):
        reporter.error(f"{context}: summary candidate archive count is inconsistent")
    if summary.get("archives_with_included_members") != sum(
        bool(entry.get("included_members"))
        for entry in archives if isinstance(entry, dict)
    ):
        reporter.error(f"{context}: summary archive count is inconsistent")
    try:
        parsed_map = parse_linker_map(map_path)
    except (OSError, ValueError) as error:
        reporter.error(f"{context}: cannot parse exact linker-map inclusions: {error}")
    else:
        if parsed_map != ground_truth_map:
            reporter.error(
                f"{context}: ground-truth archive/member pairs differ from linker map"
            )


def compare_ground_truth_manifest(
    *,
    profile: str,
    dataset_records: list[dict[str, Any]],
    ground_truth_manifest: dict[str, Any],
    reporter: Reporter,
) -> None:
    if ground_truth_manifest.get("profile") != profile:
        reporter.error(f"{profile}: ground-truth manifest has wrong profile")
    records = ground_truth_manifest.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        reporter.error(f"{profile}: ground-truth manifest records must be a list of objects")
        return
    dataset_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    ground_truth_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in dataset_records:
        key = matrix_key(profile, record)
        if key is not None and key not in dataset_by_key:
            dataset_by_key[key] = record
    for record in records:
        key = matrix_key(profile, record)
        if key is not None and key not in ground_truth_by_key:
            ground_truth_by_key[key] = record
    if Counter(filter(None, (matrix_key(profile, item) for item in dataset_records))) != Counter(
        filter(None, (matrix_key(profile, item) for item in records))
    ):
        reporter.error(f"{profile}: dataset and ground-truth manifests have different matrix records")
        return
    for key, dataset_record in dataset_by_key.items():
        gt_record = ground_truth_by_key[key]
        for field_name in (
            "binary", "binary_sha256", "ground_truth", "linker_map", "linker_map_sha256"
        ):
            if dataset_record.get(field_name) != gt_record.get(field_name):
                reporter.error(
                    f"{profile}:{format_key(key)}: {field_name} differs between manifests"
                )


def expected_record_provenance(
    profile: str,
    profile_plan: dict[str, Any],
    plan_dir: Path,
    reporter: Reporter,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_manifest = read_json(
        DEFAULT_SOURCE_MANIFEST, reporter, "source manifest"
    )
    source_entries = {
        entry.get("name"): entry
        for entry in (source_manifest or {}).get("entries", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    expected: dict[str, dict[str, Any]] = {}
    inventory_path = plan_dir / str(profile_plan["inventory"])
    inventory = read_json(inventory_path, reporter, f"{profile} provenance inventory")
    for project in (inventory or {}).get("projects", []):
        if not isinstance(project, dict):
            continue
        project_id = str(project.get("id"))
        source_name = PROJECT_SOURCE_NAMES.get(project_id)
        source_entry = source_entries.get(source_name, {})
        metadata = project.get("program_metadata", {})
        for program in project.get("programs", []):
            extra = metadata.get(program, {}) if isinstance(metadata, dict) else {}
            expected[str(program)] = {
                "source_project": project_id,
                "source_project_name": project.get("source_project"),
                "source_version": project.get("source_version"),
                "source_name": source_name,
                "source_url": source_entry.get("url"),
                "source_sha256": source_entry.get("sha256"),
                "source_revision": source_entry.get("revision"),
                "original_libseeker_url": project.get("source_url"),
                "source_basename": extra.get("source_basename", program),
                "alias_of": extra.get("alias_of"),
            }
    return expected, source_entries


def validate_profile(
    *,
    profile: str,
    profile_plan: dict[str, Any],
    plan_dir: Path,
    artifact_root: Path,
    reporter: Reporter,
    json_cache: dict[Path, dict[str, Any] | None],
    hash_cache: dict[Path, str],
) -> ProfileResult:
    canonical_binary_count, expected_program_count = CANONICAL_COUNTS[profile]
    scoped = isinstance(profile_plan.get("matrix_scope"), dict)
    plan_binary_count = profile_plan.get("expected_binaries")
    plan_program_count = profile_plan.get("expected_programs")
    if not scoped and plan_binary_count != canonical_binary_count:
        reporter.error(
            f"{profile}: plan expected_binaries is {plan_binary_count!r}, "
            f"expected canonical value {canonical_binary_count}"
        )
    if plan_program_count != expected_program_count:
        reporter.error(
            f"{profile}: plan expected_programs is {plan_program_count!r}, "
            f"expected canonical value {expected_program_count}"
        )

    expected_programs, expected_matrix = load_expected_matrix(
        profile, profile_plan, plan_dir, reporter
    )
    expected_binary_count = len(expected_matrix)
    expected_provenance, source_entries = expected_record_provenance(
        profile, profile_plan, plan_dir, reporter
    )
    if len(expected_programs) != expected_program_count:
        reporter.error(
            f"{profile}: inventory has {len(expected_programs)} programs, "
            f"expected {expected_program_count}"
        )
    if not scoped and len(expected_matrix) != canonical_binary_count:
        reporter.error(
            f"{profile}: plan/inventory define {len(expected_matrix)} unique matrix cells, "
            f"expected {canonical_binary_count}"
        )

    dataset_root = (artifact_root / "datasets" / profile).resolve()
    ground_truth_root = (artifact_root / "ground_truth" / profile).resolve()
    dataset_manifest_path = dataset_root / "manifest.json"
    ground_truth_manifest_path = ground_truth_root / "manifest.json"
    dataset_manifest = read_json(
        dataset_manifest_path, reporter, f"{profile} dataset manifest"
    )
    ground_truth_manifest = read_json(
        ground_truth_manifest_path, reporter, f"{profile} ground-truth manifest"
    )
    if dataset_manifest is None:
        return ProfileResult(profile, [], set(), {}, set())
    if dataset_manifest.get("profile") != profile:
        reporter.error(f"{profile}: dataset manifest has wrong profile")
    records_value = dataset_manifest.get("records")
    if not isinstance(records_value, list) or not all(
        isinstance(item, dict) for item in records_value
    ):
        reporter.error(f"{profile}: dataset manifest records must be a list of objects")
        records: list[dict[str, Any]] = []
    else:
        records = records_value
    summary_value = dataset_manifest.get("summary")
    deduplicated = (
        isinstance(summary_value, dict)
        and summary_value.get("deduplicated_by")
        == "binary_sha256"
    )
    deduplication_requested = (
        profile_plan.get("deduplicate_binary_sha256") is True
    )

    if deduplication_requested != deduplicated:
        reporter.error(
            f"{profile}: deduplication policy in plan "
            "and manifest do not match"
        )

    if (
        not deduplicated
        and len(records) != expected_binary_count
    ):
        reporter.error(
            f"{profile}: manifest has {len(records)} records, "
            f"expected {expected_binary_count}"
        )

    if (
        deduplicated
        and not (0 < len(records) <= expected_binary_count)
    ):
        reporter.error(
            f"{profile}: invalid deduplicated record count "
            f"{len(records)}"
        )

    summary = dataset_manifest.get("summary")
    if not isinstance(summary, dict):
        reporter.error(f"{profile}: dataset manifest summary must be an object")
        summary = {}
    if summary.get("binaries") != len(records):
        reporter.error(f"{profile}: summary.binaries does not match record count")
    if (
        not deduplicated
        and summary.get("binaries") != expected_binary_count
    ):
        reporter.error(
            f"{profile}: summary.binaries "
            "does not match dataset plan"
        )

    if deduplicated:
        candidate_binaries = summary.get(
            "candidate_binaries"
        )
        duplicates_removed = summary.get(
            "duplicates_removed"
        )

        if candidate_binaries != expected_binary_count:
            reporter.error(
                f"{profile}: summary.candidate_binaries is "
                f"{candidate_binaries!r}, "
                f"expected {expected_binary_count}"
            )

        if (
            duplicates_removed
            != expected_binary_count - len(records)
        ):
            reporter.error(
                f"{profile}: summary.duplicates_removed "
                "is inconsistent"
            )

        if (
            summary.get("unique_binary_hashes")
            != len(records)
        ):
            reporter.error(
                f"{profile}: deduplicated summary must have "
                "one unique hash per record"
            )

    diagnostics = dataset_manifest.get("diagnostics")
    if diagnostics not in ([], None):
        reporter.error(f"{profile}: dataset manifest contains build diagnostics")
    if summary.get("diagnostics") not in (0, None):
        reporter.error(f"{profile}: summary reports build diagnostics")

    keys: list[tuple[str, ...]] = []
    programs: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    seen_binary_paths: set[Path] = set()
    seen_gt_paths: set[Path] = set()
    seen_map_paths: set[Path] = set()
    allowed_library_optimizations = set(profile_plan.get("library_optimizations", []))
    allowed_program_optimizations = {
        str(cell.get("program_optimization"))
        for cell in profile_plan.get("matrix", [])
        if isinstance(cell, dict) and cell.get("program_optimization")
    }
    allowed_compiler_commands = {
        str(cell.get("compiler"))
        for cell in profile_plan.get("matrix", [])
        if isinstance(cell, dict) and cell.get("compiler")
    }

    for index, record in enumerate(records):
        key = matrix_key(profile, record)
        if key is None:
            reporter.error(f"{profile}: record[{index}] has incomplete matrix fields")
            label = f"record[{index}]"
        else:
            keys.append(key)
            programs.add(key[0])
            label = format_key(key)
        program_name = str(record.get("program", ""))
        provenance = expected_provenance.get(program_name)
        if provenance is None:
            reporter.error(f"{profile}:{label}: no expected source provenance")
        else:
            for field_name, expected_value in provenance.items():
                if record.get(field_name) != expected_value:
                    reporter.error(
                        f"{profile}:{label}: {field_name}={record.get(field_name)!r}, "
                        f"expected {expected_value!r}"
                    )
        source_name = record.get("source_name")
        source_entry = source_entries.get(source_name)
        if not isinstance(source_entry, dict):
            reporter.error(f"{profile}:{label}: source_name is absent from source manifest")
        compiler_details = record.get("compiler_metadata")
        if not isinstance(compiler_details, dict) or not all(
            compiler_details.get(field) for field in ("command", "path", "version")
        ):
            reporter.error(f"{profile}:{label}: compiler_metadata is incomplete")
        else:
            metadata_family = compiler_family({
                "compiler": str(
                    compiler_details.get("command") or compiler_details.get("path")
                )
            })
            if metadata_family != compiler_family(record):
                reporter.error(
                    f"{profile}:{label}: compiler metadata family mismatches matrix"
                )
        command = compiler_command(record)
        if command not in allowed_compiler_commands:
            reporter.error(f"{profile}:{label}: unexpected compiler command {command!r}")
        if record.get("program_optimization") not in allowed_program_optimizations:
            reporter.error(
                f"{profile}:{label}: unsupported program optimization "
                f"{record.get('program_optimization')!r}"
            )
        if record.get("program_optimization") == "O1":
            reporter.error(f"{profile}:{label}: O1 program optimization is forbidden")
        optimization_flags = record.get("program_optimization_flags")
        if not isinstance(optimization_flags, dict) or not str(
            optimization_flags.get("CFLAGS", "")
        ).startswith(f"-{record.get('program_optimization')} "):
            reporter.error(f"{profile}:{label}: program CFLAGS do not prove optimization")
        if allowed_library_optimizations:
            selections = record.get("library_optimizations")
            if not isinstance(selections, dict) or not selections:
                reporter.error(f"{profile}:{label}: missing library_optimizations")
            else:
                invalid = sorted(set(selections.values()) - allowed_library_optimizations)
                if invalid:
                    reporter.error(
                        f"{profile}:{label}: unsupported library optimizations {invalid}"
                    )
                if "O1" in selections.values():
                    reporter.error(f"{profile}:{label}: O1 library optimization is forbidden")
            versions = record.get("library_versions")
            roles = record.get("library_version_roles")
            if profile_plan.get("library_selection") == (
                "balanced_seeded_explicit_version_role_toolchain_and_optimization_v2"
            ):
                selected_keys = set(selections) if isinstance(selections, dict) else set()
                if not isinstance(versions, dict) or set(versions) != selected_keys:
                    reporter.error(
                        f"{profile}:{label}: library_versions do not match "
                        "the selected libraries"
                    )
                allowed_roles = set(profile_plan.get("library_version_roles", []))
                if not isinstance(roles, dict) or set(roles) != selected_keys:
                    reporter.error(
                        f"{profile}:{label}: library_version_roles do not match "
                        "the selected libraries"
                    )
                elif set(roles.values()) - allowed_roles:
                    reporter.error(
                        f"{profile}:{label}: unsupported library version roles "
                        f"{sorted(set(roles.values()) - allowed_roles)}"
                    )

        binary_path = resolve_declared_path(
            record.get("binary"),
            manifest_dir=dataset_root,
            artifact_root=artifact_root,
        )
        if binary_path is None:
            reporter.error(f"{profile}:{label}: record has no binary path")
            continue
        if binary_path in seen_binary_paths:
            reporter.error(f"{profile}:{label}: binary path is reused: {binary_path}")
        seen_binary_paths.add(binary_path)
        if not is_below(binary_path, dataset_root):
            reporter.error(f"{profile}:{label}: binary is outside {dataset_root}: {binary_path}")
        if not binary_path.is_file():
            reporter.error(f"{profile}:{label}: binary does not exist: {binary_path}")
            continue
        elf_ok, elf_reason = inspect_static_x86_64_elf(binary_path)
        if not elf_ok:
            reporter.error(f"{profile}:{label}: invalid binary {binary_path}: {elf_reason}")
        if binary_path not in hash_cache:
            hash_cache[binary_path] = sha256(binary_path)
        actual_binary_sha = hash_cache[binary_path]
        manifest_binary_sha = record.get("binary_sha256")
        if not valid_sha256(manifest_binary_sha):
            reporter.error(f"{profile}:{label}: invalid or missing binary_sha256")
        elif manifest_binary_sha != actual_binary_sha:
            reporter.error(f"{profile}:{label}: binary SHA-256 differs from manifest")
        source_sha = record.get("sha256")
        if source_sha is not None and source_sha != actual_binary_sha:
            reporter.error(f"{profile}:{label}: sha256 and materialized ELF content differ")
        hashes[actual_binary_sha].append(label)

        gt_path = resolve_declared_path(
            record.get("ground_truth"),
            manifest_dir=dataset_root,
            artifact_root=artifact_root,
        )
        if gt_path is not None:
            if gt_path in seen_gt_paths:
                reporter.error(f"{profile}:{label}: ground-truth path is reused: {gt_path}")
            seen_gt_paths.add(gt_path)
        declared_map_path = resolve_declared_path(
            record.get("linker_map"),
            manifest_dir=dataset_root,
            artifact_root=artifact_root,
        )
        if declared_map_path is not None:
            if declared_map_path in seen_map_paths:
                reporter.error(f"{profile}:{label}: linker-map path is reused")
            seen_map_paths.add(declared_map_path)
        validate_ground_truth(
            profile=profile,
            record=record,
            record_label=label,
            binary_path=binary_path,
            actual_binary_sha=actual_binary_sha,
            dataset_manifest_dir=dataset_root,
            ground_truth_root=ground_truth_root,
            artifact_root=artifact_root,
            reporter=reporter,
            json_cache=json_cache,
            hash_cache=hash_cache,
        )

    if profile_plan.get("library_selection") == (
        "balanced_seeded_explicit_version_role_toolchain_and_optimization_v2"
    ):
        role_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for library, role in record.get("library_version_roles", {}).items():
                role_counts[str(library)][str(role)] += 1
        serialized_role_counts = {
            library: dict(sorted(counts.items()))
            for library, counts in sorted(role_counts.items())
        }
        if summary.get("library_version_role_counts") != serialized_role_counts:
            reporter.error(
                f"{profile}: summary.library_version_role_counts is inconsistent"
            )
        if not scoped:
            for library, counts in sorted(role_counts.items()):
                if len(counts) > 1 and max(counts.values()) - min(counts.values()) > 104:
                    reporter.error(
                        f"{profile}: library version roles are not balanced for "
                        f"{library}: {dict(sorted(counts.items()))}"
                    )

    duplicate_hashes = {
        digest: labels
        for digest, labels in hashes.items()
        if len(labels) > 1
    }

    if duplicate_hashes:
        if not profile_plan.get("allow_alias_sha256_duplicates"):
            examples = "; ".join(
                f"{digest} [{', '.join(labels[:4])}]"
                for digest, labels
                in list(sorted(duplicate_hashes.items()))[:10]
            )
            reporter.error(
                f"{profile}: {len(duplicate_hashes)} "
                f"duplicate ELF SHA-256 groups: {examples}"
            )
        else:
            records_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                digest = record.get("binary_sha256")
                if isinstance(digest, str):
                    records_by_hash[digest].append(record)
            for digest in duplicate_hashes:
                group = records_by_hash[digest]
                canonical_names = {
                    str(record.get("alias_of") or record.get("program"))
                    for record in group
                }
                coordinates = {
                    (
                        compiler_command(record),
                        str(record.get("program_optimization", "")),
                    )
                    for record in group
                }
                if len(canonical_names) != 1 or len(coordinates) != 1:
                    reporter.error(
                        f"{profile}: non-alias duplicate ELF SHA-256 {digest}: "
                        f"program roots={sorted(canonical_names)}, "
                        f"coordinates={sorted(coordinates)}"
                    )

    key_counts = Counter(keys)
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]
    report_set_difference(
        reporter,
        profile=profile,
        label="duplicate matrix records",
        values=duplicate_keys,
    )
    actual_matrix = set(keys)
    if not deduplicated:
        report_set_difference(
            reporter,
            profile=profile,
            label="missing matrix cells",
            values=expected_matrix - actual_matrix,
        )

    report_set_difference(
        reporter,
        profile=profile,
        label="unexpected matrix cells",
        values=actual_matrix - expected_matrix,
    )
    family_counts = Counter(compiler_family(record) for record in records)
    if (
        not scoped
        and not deduplicated
        and family_counts.get("gcc", 0)
        != family_counts.get("clang", 0)
    ):
        reporter.error(
            f"{profile}: GCC/Clang counts are not balanced: "
            f"{family_counts.get('gcc', 0)} != {family_counts.get('clang', 0)}"
        )
    versions_by_family: dict[str, set[str]] = defaultdict(set)
    optimizations_by_compiler: dict[str, set[str]] = defaultdict(set)
    for record in records:
        command = compiler_command(record)
        versions_by_family[compiler_family(record)].add(command)
        optimization = record.get("program_optimization")
        if isinstance(optimization, str):
            optimizations_by_compiler[command].add(optimization)
    if not deduplicated:
        required_versions: dict[str, set[str]] = defaultdict(set)
        for command in allowed_compiler_commands:
            required_versions[compiler_family({"compiler": command})].add(command)
        for family, required in required_versions.items():
            if versions_by_family[family] != required:
                reporter.error(
                    f"{profile}: expected {family} versions {sorted(required)}, found "
                    f"{sorted(versions_by_family[family])}"
                )

        required_optimizations = {
            "O0", "O2", "O3", "Os"
        }

        for command in sorted(
            allowed_compiler_commands
        ):
            if (
                optimizations_by_compiler[command]
                != required_optimizations
            ):
                reporter.error(
                    f"{profile}: compiler {command} "
                    f"has optimizations "
                    f"{sorted(optimizations_by_compiler[command])}, "
                    f"expected "
                    f"{sorted(required_optimizations)}"
                )
    missing_names = sorted(
        expected_programs - programs
    )
    extra_names = sorted(
        programs - expected_programs
    )

    if not deduplicated and missing_names:
        reporter.error(
            f"{profile}: missing program names "
            f"({len(missing_names)}): "
            + ", ".join(missing_names[:30])
        )

    if extra_names:
        reporter.error(
            f"{profile}: unexpected program names "
            f"({len(extra_names)}): "
            + ", ".join(extra_names[:30])
        )

    if (
        not deduplicated
        and len(programs) != expected_program_count
    ):
        reporter.error(
            f"{profile}: manifest has {len(programs)} "
            f"unique programs, "
            f"expected {expected_program_count}"
        )

    if summary.get("programs") != len(programs):
        reporter.error(
            f"{profile}: summary.programs "
            "does not match manifest records"
        )

    if (
        not deduplicated
        and summary.get("programs")
        != expected_program_count
    ):
        reporter.error(
            f"{profile}: summary.programs "
            "does not match dataset plan"
        )

    def compare_snapshot_files(
        label: str, expected_paths: set[Path], actual_paths: set[Path]
    ) -> None:
        missing = sorted(expected_paths - actual_paths)
        orphaned = sorted(actual_paths - expected_paths)
        if missing:
            reporter.error(
                f"{profile}: {label} missing from snapshot ({len(missing)}): "
                + ", ".join(str(path) for path in missing[:10])
            )
        if orphaned:
            reporter.error(
                f"{profile}: unmanifested/orphan {label} files ({len(orphaned)}): "
                + ", ".join(str(path) for path in orphaned[:10])
            )

    actual_binaries = {
        path.resolve() for path in (dataset_root / "binaries").rglob("*")
        if path.is_file()
    } if (dataset_root / "binaries").is_dir() else set()
    actual_gt = {
        path.resolve() for path in (ground_truth_root / "binaries").rglob("ground_truth.json")
        if path.is_file()
    } if (ground_truth_root / "binaries").is_dir() else set()
    actual_maps = {
        path.resolve() for path in (ground_truth_root / "binaries").rglob("linker.map")
        if path.is_file()
    } if (ground_truth_root / "binaries").is_dir() else set()
    compare_snapshot_files("ELF", seen_binary_paths, actual_binaries)
    compare_snapshot_files("ground-truth JSON", seen_gt_paths, actual_gt)
    compare_snapshot_files("linker map", seen_map_paths, actual_maps)

    expected_catalogs: set[Path] = set()
    for gt_path in seen_gt_paths:
        ground_truth = json_cache.get(gt_path)
        if not isinstance(ground_truth, dict):
            continue
        for archive in ground_truth.get("archives", []):
            if not isinstance(archive, dict):
                continue
            catalog = resolve_declared_path(
                archive.get("archive_catalog"),
                manifest_dir=gt_path.parent,
                artifact_root=artifact_root,
            )
            if catalog is not None:
                expected_catalogs.add(catalog)
    catalog_dir = ground_truth_root / "archive_catalog"
    actual_catalogs = {
        path.resolve() for path in catalog_dir.glob("*.json") if path.is_file()
    } if catalog_dir.is_dir() else set()
    compare_snapshot_files("archive catalog", expected_catalogs, actual_catalogs)

    if ground_truth_manifest is not None:
        compare_ground_truth_manifest(
            profile=profile,
            dataset_records=records,
            ground_truth_manifest=ground_truth_manifest,
            reporter=reporter,
        )
    return ProfileResult(profile, records, programs, dict(hashes), actual_matrix)


def validate_cross_profile(
    libseeker: ProfileResult,
    unseen: ProfileResult,
    *,
    allow_hash_overlap: bool,
    reporter: Reporter,
) -> tuple[int, int]:
    common_names = sorted(libseeker.programs & unseen.programs)
    if common_names:
        preview = ", ".join(common_names[:30])
        suffix = f" (+{len(common_names) - 30} more)" if len(common_names) > 30 else ""
        reporter.error(
            f"cross-profile: {len(common_names)} program names occur in both datasets: "
            f"{preview}{suffix}"
        )

    common_hashes = sorted(set(libseeker.hashes) & set(unseen.hashes))
    if common_hashes:
        examples = []
        for digest in common_hashes[:10]:
            left = ",".join(libseeker.hashes[digest][:3])
            right = ",".join(unseen.hashes[digest][:3])
            examples.append(f"{digest} [{left}] <-> [{right}]")
        suffix = f" (+{len(common_hashes) - 10} more)" if len(common_hashes) > 10 else ""
        message = (
            f"cross-profile: {len(common_hashes)} ELF SHA-256 values occur in both datasets: "
            + "; ".join(examples)
            + suffix
        )
        if allow_hash_overlap:
            reporter.warning(message)
        else:
            reporter.error(message)
    return len(common_names), len(common_hashes)


def print_messages(kind: str, messages: list[str], limit: int = 100) -> None:
    for message in messages[:limit]:
        print(f"{kind}: {message}", file=sys.stderr)
    if len(messages) > limit:
        print(
            f"{kind}: {len(messages) - limit} additional messages omitted",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    plan = read_json(args.plan, reporter, "dataset plan")
    if plan is None:
        print_messages("ERROR", reporter.errors)
        return 1
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict):
        reporter.error("dataset plan: datasets must be an object")
        print_messages("ERROR", reporter.errors)
        return 1

    profiles = (args.profile,) if args.profile else PROFILES
    results: dict[str, ProfileResult] = {}
    json_cache: dict[Path, dict[str, Any] | None] = {}
    hash_cache: dict[Path, str] = {}
    for profile in profiles:
        profile_plan = datasets.get(profile)
        if not isinstance(profile_plan, dict):
            reporter.error(f"dataset plan: missing profile {profile!r}")
            continue
        try:
            profile_plan = restrict_plan_compilers(profile_plan, args.compiler)
        except ValueError as error:
            reporter.error(f"{profile}: {error}")
            continue
        before = len(reporter.errors)
        result = validate_profile(
            profile=profile,
            profile_plan=profile_plan,
            plan_dir=args.plan.parent,
            artifact_root=args.artifact_root,
            reporter=reporter,
            json_cache=json_cache,
            hash_cache=hash_cache,
        )
        results[profile] = result
        profile_errors = len(reporter.errors) - before
        print(
            f"{profile}: {len(result.records)} records, {len(result.programs)} programs, "
            f"{len(result.hashes)} unique ELF hashes, {profile_errors} error(s)"
        )

    if set(PROFILES).issubset(results):
        common_names, common_hashes = validate_cross_profile(
            results["libseeker"],
            results["unseen"],
            allow_hash_overlap=args.allow_hash_overlap,
            reporter=reporter,
        )
        print(
            f"cross-profile: {common_names} shared program names, "
            f"{common_hashes} shared ELF hashes"
        )

    print_messages("WARNING", reporter.warnings)
    print_messages("ERROR", reporter.errors)
    if reporter.errors:
        print(
            f"VALIDATION FAILED: {len(reporter.errors)} error(s), "
            f"{len(reporter.warnings)} warning(s)",
            file=sys.stderr,
        )
        return 1
    print(f"VALIDATION OK: {len(reporter.warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
