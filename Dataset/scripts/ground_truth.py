#!/usr/bin/env python3
"""Exact archive/CU ground truth derived from GNU linker map files.

The per-binary ground truth stays small: it records the archive members that
the linker actually selected.  Full archive membership and defined function
symbols are normalized into a content-addressed catalog, once per archive.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_MEMBER_RE = re.compile(
    r"(?P<archive>\S+?\.a)\((?P<member>[^()\s]+)\)"
)
NM_POSIX_RE = re.compile(
    r"^(?P<archive>.+?\.a)\[(?P<member>[^\]]+)\]:\s+"
    r"(?P<symbol>\S+)\s+(?P<type>\S)(?:\s+\S+)?(?:\s+\S+)?$"
)
FUNCTION_SYMBOL_TYPES = frozenset("TtWwIi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_linker_map(path: Path) -> dict[str, list[str]]:
    """Return archive members from GNU ld's exact inclusion section."""
    included: dict[str, list[str]] = defaultdict(list)
    in_inclusion_section = False
    saw_member = False
    for raw_line in path.read_text(errors="replace").splitlines():
        if raw_line.startswith("Archive member included"):
            in_inclusion_section = True
            continue
        if not in_inclusion_section:
            continue
        if not raw_line.strip():
            if saw_member:
                break
            continue
        # Inclusion records start in column zero; the following indented line
        # only describes the file/symbol that caused the archive extraction.
        match = ARCHIVE_MEMBER_RE.match(raw_line)
        if match:
            saw_member = True
            included[match.group("archive")].append(match.group("member"))
    if not saw_member:
        raise ValueError(f"GNU linker-map inclusion section not found: {path}")
    return {
        archive: members
        for archive, members in sorted(included.items())
    }


def _metadata_archive(entry: dict[str, Any]) -> Path | None:
    value = entry.get("archive") or entry.get("path")
    return Path(value).resolve() if value else None


def resolve_archive(
    raw_archive: str,
    map_path: Path,
    metadata: Iterable[dict[str, Any]],
    resolution_roots: Iterable[Path] = (),
) -> Path | None:
    raw = Path(raw_archive)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((map_path.parent / raw, Path.cwd() / raw))
        candidates.extend(root / raw for root in resolution_roots)

    metadata_paths = [
        path for entry in metadata if (path := _metadata_archive(entry))
    ]
    metadata_matches = [
        path for path in metadata_paths
        if path.name == raw.name or str(path).endswith(raw_archive)
    ]
    if len(metadata_matches) == 1:
        candidates.extend(metadata_matches)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def archive_metadata_for_path(
    archive: Path | None,
    raw_archive: str,
    metadata: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for entry in metadata:
        candidate = _metadata_archive(entry)
        if candidate is None:
            continue
        if archive is not None and candidate == archive:
            return dict(entry)
        if candidate.name == Path(raw_archive).name:
            candidates.append(entry)
    if len(candidates) == 1:
        return dict(candidates[0])
    return {}


def archive_members(path: Path) -> list[str]:
    result = subprocess.run(
        ["ar", "t", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def duplicate_member_metadata(
    path: Path,
    name_counts: Counter[str],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], str]]:
    """Hash each duplicate archive occurrence using GNU ar's xN selector."""
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    if not duplicates:
        return {}, {}
    metadata: dict[str, dict[str, Any]] = {}
    occurrence_hashes: dict[tuple[str, int], str] = {}
    with tempfile.TemporaryDirectory(prefix="gt-ar-occurrences-") as temporary:
        work = Path(temporary)
        for name, count in sorted(duplicates.items()):
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"unsafe duplicate archive member name: {name!r}")
            hashes = []
            for occurrence in range(1, count + 1):
                subprocess.run(
                    ["ar", "xN", str(occurrence), str(path), name],
                    cwd=work,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                extracted = work / name
                digest = sha256(extracted)
                extracted.unlink()
                hashes.append(digest)
                occurrence_hashes[(name, occurrence)] = digest
            metadata[name] = {
                "occurrences": count,
                "object_sha256": hashes,
                "byte_identical": len(set(hashes)) == 1,
            }
    return metadata, occurrence_hashes


def archive_function_symbols(path: Path) -> dict[str, list[str]]:
    """List defined text/weak/ifunc symbols for each archive member."""
    result = subprocess.run(
        ["nm", "-A", "--defined-only", "--format=posix", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    symbols: dict[str, set[str]] = defaultdict(set)
    for line in result.stdout.splitlines():
        match = NM_POSIX_RE.match(line)
        if not match or match.group("type") not in FUNCTION_SYMBOL_TYPES:
            continue
        symbol = match.group("symbol")
        if symbol and not symbol.startswith(".L"):
            symbols[match.group("member")].add(symbol)
    return {member: sorted(values) for member, values in sorted(symbols.items())}


def ensure_archive_catalog(path: Path, catalog_root: Path) -> dict[str, Any]:
    digest = sha256(path)
    destination = catalog_root / f"{digest}.json"
    if destination.is_file():
        cached = json.loads(destination.read_text())
        if cached.get("schema_version") == 3:
            return cached

    members = archive_members(path)
    name_counts = Counter(members)
    duplicate_metadata, occurrence_hashes = duplicate_member_metadata(
        path, name_counts
    )
    occurrences: dict[str, int] = defaultdict(int)
    functions = archive_function_symbols(path)
    member_records = []
    for member in members:
        occurrences[member] += 1
        member_records.append({
            "name": member,
            "occurrence": occurrences[member],
            "object_sha256": occurrence_hashes.get(
                (member, occurrences[member])
            ),
            "defined_functions": functions.get(member, []),
        })
    payload = {
        "schema_version": 3,
        "archive_sha256": digest,
        "archive_basename": path.name,
        "archive_size": path.stat().st_size,
        "members": member_records,
        "summary": {
            "members": len(members),
            "members_with_functions": sum(member in functions for member in members),
            "defined_functions": sum(len(values) for values in functions.values()),
            "duplicate_member_names": duplicate_metadata,
        },
    }
    catalog_root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return payload


def resolved_archive_record(
    *,
    archive: Path,
    raw_archive: str | None,
    metadata: dict[str, Any],
    included_members: list[str],
    catalog_root: Path,
    selected_by_linker: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    catalog = ensure_archive_catalog(archive, catalog_root)
    duplicate_names = catalog["summary"].get("duplicate_member_names", {})
    selected_ambiguous = sorted(
        name for name in set(included_members) & set(duplicate_names)
        if not duplicate_names[name].get("byte_identical", False)
    )
    catalog_names = {member["name"] for member in catalog["members"]}
    selected_missing = sorted(set(included_members) - catalog_names)
    identity_exact = not selected_ambiguous and not selected_missing
    record = {
        **metadata,
        "archive": str(archive),
        "linker_map_archive": raw_archive,
        "selected_by_linker": selected_by_linker,
        "archive_basename": archive.name,
        "archive_sha256": catalog["archive_sha256"],
        "archive_catalog": str(
            (catalog_root / f"{catalog['archive_sha256']}.json").resolve()
        ),
        "resolved": True,
        "included_members": included_members,
        "included_member_count": len(included_members),
        "archive_member_count": catalog["summary"]["members"],
        "archive_member_names_unique": not duplicate_names,
        "archive_member_identity_exact": identity_exact,
        "equivalent_duplicate_members": {
            name: duplicate_names[name]
            for name in sorted(set(included_members) & set(duplicate_names))
            if duplicate_names[name].get("byte_identical", False)
        },
        # Compatibility aliases used by the existing LibSeeker consumers.
        "included_compilation_units": len(included_members),
        "total_compilation_units": catalog["summary"]["members"],
        "cu_ground_truth_method": "linker_map" if included_members else "none",
        "confirmed_compilation_units": list(included_members),
        "confirmed_compilation_units_method": (
            "linker_map" if included_members else "none"
        ),
    }
    return record, selected_ambiguous, selected_missing


def build_ground_truth(
    *,
    binary: Path,
    linker_map: Path,
    archive_metadata: list[dict[str, Any]],
    catalog_root: Path,
    build_metadata: dict[str, Any],
    resolution_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build exact per-binary library/CU ground truth."""
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not linker_map.is_file():
        raise FileNotFoundError(linker_map)

    parsed = parse_linker_map(linker_map)
    metadata_entries = [dict(entry) for entry in archive_metadata]
    archives = []
    unresolved = []
    ambiguous_member_names = []
    missing_catalog_members = []
    resolved_paths: set[Path] = set()
    for raw_archive, included_members in parsed.items():
        archive = resolve_archive(
            raw_archive,
            linker_map,
            metadata_entries,
            resolution_roots,
        )
        metadata = archive_metadata_for_path(
            archive, raw_archive, metadata_entries
        )
        if archive is None:
            unresolved.append(raw_archive)
            archives.append({
                "archive": raw_archive,
                "linker_map_archive": raw_archive,
                "selected_by_linker": True,
                "resolved": False,
                "included_members": included_members,
                "included_member_count": len(included_members),
                "included_compilation_units": len(included_members),
                "total_compilation_units": 0,
                "cu_ground_truth_method": "linker_map",
                "confirmed_compilation_units": list(included_members),
                "confirmed_compilation_units_method": "linker_map",
                **metadata,
            })
            continue
        resolved_paths.add(archive)
        record, selected_ambiguous, selected_missing = resolved_archive_record(
            archive=archive,
            raw_archive=raw_archive,
            metadata=metadata,
            included_members=included_members,
            catalog_root=catalog_root,
            selected_by_linker=True,
        )
        if selected_ambiguous:
            ambiguous_member_names.append({
                "archive": str(archive),
                "members": selected_ambiguous,
            })
        if selected_missing:
            missing_catalog_members.append({
                "archive": str(archive),
                "members": selected_missing,
            })
        archives.append(record)

    # Preserve the explicit negative universe too: archives passed to the
    # linker from which no member was extracted remain auditable candidates.
    for metadata in metadata_entries:
        archive = _metadata_archive(metadata)
        if archive is None or archive in resolved_paths:
            continue
        if not archive.is_file():
            unresolved.append(str(archive))
            archives.append({
                **metadata,
                "archive": str(archive),
                "linker_map_archive": None,
                "selected_by_linker": False,
                "resolved": False,
                "included_members": [],
                "included_member_count": 0,
                "included_compilation_units": 0,
                "total_compilation_units": 0,
                "cu_ground_truth_method": "none",
                "confirmed_compilation_units": [],
                "confirmed_compilation_units_method": "none",
            })
            continue
        resolved_paths.add(archive)
        record, _, _ = resolved_archive_record(
            archive=archive,
            raw_archive=None,
            metadata=metadata,
            included_members=[],
            catalog_root=catalog_root,
            selected_by_linker=False,
        )
        archives.append(record)

    return {
        "schema_version": 1,
        "method": "gnu_linker_map_archive_members",
        "scope": "exact",
        "binary": str(binary.resolve()),
        "binary_sha256": sha256(binary),
        "binary_size": binary.stat().st_size,
        "linker_map": str(linker_map.resolve()),
        "linker_map_sha256": sha256(linker_map),
        "build": build_metadata,
        "archives": archives,
        "summary": {
            "candidate_archives": len(archives),
            "archives_with_included_members": sum(
                bool(entry["included_members"]) for entry in archives
            ),
            "included_archive_members": sum(
                len(entry["included_members"]) for entry in archives
            ),
            "unresolved_archives": unresolved,
            "ambiguous_archive_member_names": ambiguous_member_names,
            "missing_catalog_members": missing_catalog_members,
            "ground_truth_complete": not unresolved
            and not ambiguous_member_names
            and not missing_catalog_members,
        },
    }
