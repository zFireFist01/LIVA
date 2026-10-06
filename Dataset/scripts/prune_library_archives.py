#!/usr/bin/env python3
"""Prune installed static libraries to the usable LibSeeker matrix.

The reference inventory contains 164 archive names.  Five additional public
archives are required to relink the ELF experiment, so the usable matrix
contains 169 names.  Current versions target four
compiler families and four optimization levels; the minor and major alternative
versions target GCC 13 and Clang 18 with the same four optimization levels.
The role of every source version is declared explicitly in
``library_version_roles.json``.  Missing or unsupported coordinates are
recorded, never filled with renamed copies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TOOLCHAINS = (
    "gcc-11-11.5.0",
    "gcc-13-13.3.0",
    "clang-14-14.0.6",
    "clang-18-18.1.3",
)
ALTERNATIVE_TOOLCHAINS = ("gcc-13-13.3.0", "clang-18-18.1.3")
OPTIMIZATIONS = ("O0", "O2", "O3", "Os")
VERSION_ROLES = ("current", "minor-alternative", "major-alternative")

# These names collide between projects, or are not currently installed by the
# owning project's normal `make install` target.  Their ownership is explicit
# so the selection remains stable before and after pruning.
OWNER_OVERRIDES = {
    "libcharset.a": "libiconv",
    "libcommon.a": "util-linux",
    "libpng.a": "libpng",
    "libpthread_syms.a": "glibc",
    "libsemanage.a": "selinux",
    "libsepol.a": "selinux",
    "libsupport_nonshared.a": "glibc",
    "libtestutil.a": "openssl",
    "libiconv.a": "libiconv",
    "libidn2.a": "libidn2",
    "libpsl.a": "libpsl",
    "libunistring.a": "libunistring",
    "libzstd.a": "zstd",
}

# These archives are linked by build_randomized_elf_matrix.py but are absent
# from the historical LibSeeker/LIVA name inventory.  Keeping them in the same
# versioned matrix makes ELF generation and analysis-cache provenance use one
# authoritative set of library coordinates.
EXPERIMENT_LINK_ARCHIVES = {
    "libiconv.a",
    "libidn2.a",
    "libpsl.a",
    "libunistring.a",
    "libzstd.a",
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=repository / "Dataset/builds/libraries",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=repository / "librerie_libseeker.txt",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=repository / "Dataset/manifests/source_manifest.json",
    )
    parser.add_argument(
        "--version-roles-manifest",
        type=Path,
        default=repository / "Dataset/manifests/library_version_roles.json",
        help="Explicit current/minor/major role assignment for every package.",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=repository / "Dataset/manifests/library_matrix.tsv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repository / "Dataset/manifests/library_matrix_summary.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete installed .a files outside the usable matrix",
    )
    return parser.parse_args()


def inventory_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^:]+\.a):", line)
        if match:
            names.append(match.group(1))
    if len(names) != 164 or len(set(names)) != 164:
        raise RuntimeError(
            f"expected 164 distinct archive names in {path}, found {len(set(names))}"
        )
    return sorted(set(names) | EXPERIMENT_LINK_ARCHIVES)


def installed_archives(build_root: Path) -> set[Path]:
    return {
        path
        # Include nested install directories as well.  The matcher only reads
        # install/lib/*.a, but nested test/plugin archives are auxiliary data
        # and must not survive a full prune.
        for path in build_root.glob("*/*/*/install/**/*.a")
        if path.is_file() or path.is_symlink()
    }


def version_candidates(package: dict, version: dict, current: bool) -> list[str]:
    candidates: list[str] = []
    covered_by = version.get("covered_by")
    if current and covered_by:
        candidates.append(covered_by)
    for value in (version.get("version"), version.get("label")):
        if value:
            value = value.split(":")[-1]
            candidates.append(f"{package['name']}-{value}")
    if not current and covered_by:
        candidates.append(covered_by)
    return list(dict.fromkeys(candidates))


def package_roles(packages: list[dict], manifest: dict) -> dict[str, dict]:
    """Resolve explicit semantic roles to source-project candidates.

    Version numbers are intentionally not parsed here: several packages use
    Debian revisions, beta suffixes or upstream-specific schemes.  The
    dedicated manifest therefore remains the authoritative, auditable choice.
    """
    if manifest.get("schema") != 1:
        raise ValueError("unsupported library version-role manifest schema")
    assignments = manifest.get("packages")
    if not isinstance(assignments, dict):
        raise ValueError("version-role manifest has no packages object")

    package_names = {str(package["name"]) for package in packages}
    assigned_names = set(assignments)
    if assigned_names != package_names:
        raise ValueError(
            "version-role package mismatch: "
            f"missing={sorted(package_names - assigned_names)}, "
            f"extra={sorted(assigned_names - package_names)}"
        )

    roles: dict[str, dict] = {}
    for package in packages:
        package_name = str(package["name"])
        assignment = assignments[package_name]
        if (
            not isinstance(assignment, dict)
            or set(assignment) != set(VERSION_ROLES)
        ):
            raise ValueError(
                f"{package_name}: expected exactly these roles: {VERSION_ROLES}"
            )
        versions_by_identifier: dict[str, dict] = {}
        for version in package["versions"]:
            for field in ("version", "label"):
                identifier = version.get(field)
                if identifier:
                    versions_by_identifier[str(identifier)] = version

        resolved: dict[str, list[str]] = {}
        used_versions: set[str] = set()
        for role in VERSION_ROLES:
            identifier = str(assignment[role])
            try:
                version = versions_by_identifier[identifier]
            except KeyError as error:
                raise ValueError(
                    f"{package_name}: {role} references unknown version {identifier}"
                ) from error
            canonical_version = str(version.get("version") or version.get("label"))
            if canonical_version in used_versions:
                raise ValueError(
                    f"{package_name}: version {canonical_version} has multiple roles"
                )
            used_versions.add(canonical_version)
            resolved[role] = version_candidates(
                package, version, current=role == "current"
            )
        if len(used_versions) != len(package["versions"]):
            raise ValueError(
                f"{package_name}: every source version must have exactly one role"
            )
        roles[package_name] = resolved
    return roles


def archive_owners(
    build_root: Path,
    names: list[str],
    packages: list[dict],
    roles: dict[str, dict],
) -> dict[str, str]:
    wanted = set(names)
    producers: dict[str, set[str]] = defaultdict(set)
    for package in packages:
        package_name = package["name"]
        groups = list(roles[package_name].values())
        projects = {project for group in groups for project in group}
        produced: set[str] = set()
        for toolchain in TOOLCHAINS:
            for project in projects:
                for opt in OPTIMIZATIONS:
                    install = build_root / toolchain / project / opt / "install"
                    produced.update(path.name for path in install.glob("lib*/*.a"))
        for archive_name in produced & wanted:
            producers[archive_name].add(package_name)

    owners: dict[str, str] = {}
    for archive_name in names:
        if archive_name in OWNER_OVERRIDES:
            owners[archive_name] = OWNER_OVERRIDES[archive_name]
        elif len(producers[archive_name]) == 1:
            owners[archive_name] = next(iter(producers[archive_name]))
        else:
            raise RuntimeError(
                f"cannot determine one owner for {archive_name}: "
                f"{sorted(producers[archive_name])}"
            )
    return owners


def coordinate_reason(
    build_root: Path,
    package: str,
    toolchain: str,
    projects: list[str],
    optimization: str,
) -> str:
    if package == "glibc" and toolchain.startswith("clang-"):
        return "unsupported_compiler"
    if package == "glibc" and optimization == "O0":
        return "unsupported_optimization"
    if not any((build_root / toolchain / project / optimization).is_dir() for project in projects):
        return "variant_not_built_or_unsupported"
    return "archive_not_produced_by_version"


def choose_archive(
    build_root: Path,
    toolchain: str,
    projects: list[str],
    optimization: str,
    archive_name: str,
) -> Path | None:
    for project in projects:
        for libdir in ("lib", "lib64"):
            candidate = (
                build_root
                / toolchain
                / project
                / optimization
                / "install"
                / libdir
                / archive_name
            )
            # `exists` deliberately excludes broken symbolic links.
            if candidate.exists():
                return candidate
    return None


def build_matrix(
    build_root: Path,
    names: list[str],
    owners: dict[str, str],
    roles: dict[str, dict],
) -> tuple[list[dict], set[Path]]:
    rows: list[dict] = []
    selected: set[Path] = set()
    for archive_name in names:
        package = owners[archive_name]
        package_role = roles[package]
        coordinates = [
            ("current", toolchain, package_role["current"], optimization)
            for toolchain in TOOLCHAINS
            for optimization in OPTIMIZATIONS
        ]
        coordinates.extend(
            (role, toolchain, package_role[role], optimization)
            for role in VERSION_ROLES[1:]
            for toolchain in ALTERNATIVE_TOOLCHAINS
            for optimization in OPTIMIZATIONS
        )
        if len(coordinates) != 32:
            raise AssertionError(f"invalid coordinate count for {archive_name}")

        for role, toolchain, projects, optimization in coordinates:
            chosen = choose_archive(
                build_root, toolchain, projects, optimization, archive_name
            )
            status = "selected" if chosen else "missing"
            reason = "" if chosen else coordinate_reason(
                build_root, package, toolchain, projects, optimization
            )
            if chosen:
                selected.add(chosen)
            rows.append(
                {
                    "archive": archive_name,
                    "package": package,
                    "role": role,
                    "toolchain": toolchain,
                    "optimization": optimization,
                    "project_candidates": "|".join(projects),
                    "status": status,
                    "reason": reason,
                    "path": str(chosen.relative_to(build_root)) if chosen else "",
                }
            )
    return rows, selected


def materialize_external_symlink_targets(selected: set[Path]) -> int:
    materialized = 0
    for path in sorted(selected):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if target in selected:
            continue
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(target, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        materialized += 1
    return materialized


def write_matrix(path: Path, rows: list[dict]) -> None:
    fields = (
        "archive",
        "package",
        "role",
        "toolchain",
        "optimization",
        "project_candidates",
        "status",
        "reason",
        "path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(fields)]
    lines.extend("\t".join(str(row[field]) for field in fields) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    build_root = args.build_root.resolve()
    names = inventory_names(args.inventory)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    version_roles_manifest = json.loads(
        args.version_roles_manifest.read_text(encoding="utf-8")
    )
    packages = source_manifest["historical_packages"]
    roles = package_roles(packages, version_roles_manifest)
    owners = archive_owners(build_root, names, packages, roles)
    rows, selected = build_matrix(build_root, names, owners, roles)
    installed_before = installed_archives(build_root)
    missing_rows = [row for row in rows if row["status"] == "missing"]
    to_delete = installed_before - selected
    auxiliary_names = set(path.name for path in installed_before) - set(names)
    auxiliary_paths = {path for path in installed_before if path.name in auxiliary_names}
    extra_wanted_paths = to_delete - auxiliary_paths

    materialized = 0
    deleted = 0
    if args.apply:
        materialized = materialize_external_symlink_targets(selected)
        for path in sorted(to_delete):
            path.unlink(missing_ok=True)
            deleted += 1

    write_matrix(args.matrix_output, rows)
    installed_after = installed_archives(build_root)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply,
        "requested_archive_names": len(names),
        "reference_archive_names": len(names) - len(EXPERIMENT_LINK_ARCHIVES),
        "experiment_link_archive_names": len(EXPERIMENT_LINK_ARCHIVES),
        "requested_coordinates": len(rows),
        "selected_coordinates": len(selected),
        "missing_coordinates": len(missing_rows),
        "installed_before": len(installed_before),
        "auxiliary_paths": len(auxiliary_paths),
        "extra_requested_name_paths": len(extra_wanted_paths),
        "planned_deletions": len(to_delete),
        "deleted_paths": deleted,
        "materialized_symlinks": materialized,
        "installed_after": len(installed_after),
        "missing_by_reason": dict(Counter(row["reason"] for row in missing_rows)),
        "missing_by_package": dict(Counter(row["package"] for row in missing_rows)),
        "selected_by_role": dict(
            Counter(row["role"] for row in rows if row["status"] == "selected")
        ),
        "missing_by_role": dict(Counter(row["role"] for row in missing_rows)),
        "version_roles_manifest": args.version_roles_manifest.resolve().relative_to(
            Path(__file__).resolve().parents[2]
        ).as_posix(),
    }
    # Once an applied pruning report exists, a later audit must not overwrite
    # its deletion counters with dry-run zeros.
    if args.apply or not args.summary_output.exists():
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
