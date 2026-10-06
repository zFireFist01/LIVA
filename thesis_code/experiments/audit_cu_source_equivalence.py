#!/usr/bin/env python3
"""Audit source-code equivalence of library CUs across versions.

The script maps archive member names observed in replay logs to source files in
Dataset/sources/lib_sources.  For C/C++ sources, equivalence is computed over
lexical code tokens, so comments and insignificant formatting do not make two
otherwise identical CUs different.  Ambiguous and unmapped CUs are retained
in the audit and never treated as equivalent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
SOURCE_ROOT = REPO_ROOT / "Dataset/sources/lib_sources"
VERSION_ROLES = REPO_ROOT / "Dataset/manifests/library_version_roles.json"
BUILD_SOURCE_MAP = UNIFIED / "remote_build_source_map.jsonl.gz"
ROLE_VERSIONS = json.loads(VERSION_ROLES.read_text(encoding="utf-8"))["packages"]
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
from evaluate_fixed_two_tasks import (  # noqa: E402
    BLOCK_CU_MARKER,
    LABEL_RE,
    NAME_RE,
    candidate_label_metadata,
    report_index_all,
)


ENGINE = family.engine
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".C", ".s", ".S", ".asm"
}
IGNORED_PARTS = {
    ".git", ".pc", "autom4te.cache", "test", "tests", "testing", "examples",
    "docs", "doc", "po",
}
LIBTOOL_PREFIX_RE = re.compile(r"^(?:lib)?[^-]+_(?:la|a)-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def c_family_code_tokens(text: str) -> list[str]:
    """Return C/C++ lexical material, excluding comments and whitespace.

    Newlines are retained only while inside a preprocessor directive, where
    they are semantically relevant. Backslash-newline splices are removed
    before tokenization, as required by the C translation phases.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\\\n", "", text)
    tokens: list[str] = []
    index = 0
    length = len(text)
    at_line_start = True
    in_directive = False
    identifier_pattern = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
    number_pattern = re.compile(
        r"(?:0[xX][0-9A-Fa-f](?:[0-9A-Fa-f'_]*[0-9A-Fa-f])?"
        r"|0[bB][01](?:[01'_]*[01])?"
        r"|(?:[0-9](?:[0-9'_]*[0-9])?)(?:\.[0-9'_]*)?"
        r"|\.[0-9](?:[0-9'_]*[0-9])?)"
        r"(?:[eEpP][+-]?[0-9](?:[0-9'_]*[0-9])?)?"
        r"[A-Za-z0-9_]*"
    )
    punctuators = (
        "->*", "<=>", ">>=", "<<=", "...", "##", "->", "++", "--", "<<", ">>",
        "<=", ">=", "==", "!=", "&&", "||", "*=", "/=", "%=", "+=",
        "-=", "&=", "^=", "|=", "::", ".*",
    )
    while index < length:
        char = text[index]
        if char in " \t\v\f":
            index += 1
            continue
        if char == "\n":
            if in_directive:
                tokens.append("<PP_NEWLINE>")
            at_line_start = True
            in_directive = False
            index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = length if newline < 0 else newline
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if at_line_start and char == "#":
            in_directive = True
        at_line_start = False
        if char in "\"'":
            quote = char
            end = index + 1
            while end < length:
                if text[end] == "\\":
                    end += 2
                    continue
                end += 1
                if text[end - 1] == quote:
                    break
            tokens.append(text[index:end])
            index = end
            continue
        identifier = identifier_pattern.match(text, index)
        if identifier:
            value = identifier.group(0)
            tokens.append(value)
            index = identifier.end()
            continue
        number = number_pattern.match(text, index)
        if number:
            value = number.group(0)
            tokens.append(value)
            index = number.end()
            continue
        matched = next(
            (value for value in punctuators if text.startswith(value, index)),
            None,
        )
        if matched:
            tokens.append(matched)
            index += len(matched)
        else:
            tokens.append(char)
            index += 1
    return tokens


def sha256_source_code(path: Path) -> str:
    """Hash executable source content rather than presentation metadata."""
    if path.suffix in {".c", ".cc", ".cpp", ".cxx", ".C"}:
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        payload = "\x1f".join(c_family_code_tokens(text)).encode(
            "utf-8", errors="surrogateescape"
        )
        return hashlib.sha256(payload).hexdigest()
    # Assembly comment syntax is target/toolchain dependent. Preserve the
    # conservative byte-level rule rather than deleting meaningful directives.
    return sha256_file(path)


def normalized_stem(value: str) -> str:
    stem = Path(value).name
    for suffix in (".o", ".lo", ".obj", ".c", ".cc", ".cpp", ".cxx", ".s", ".S", ".asm"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    previous = None
    while previous != stem:
        previous = stem
        stem = LIBTOOL_PREFIX_RE.sub("", stem)
    return stem.lower().replace("-", "_")


def source_index(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts[:-1]):
            continue
        result[normalized_stem(path.name)].append(path)
    return dict(result)


def source_candidates(
    index: dict[str, list[Path]], cu_name: str
) -> tuple[list[Path], str | None]:
    """Resolve common Automake target prefixes, preferring the longest stem."""
    stem = normalized_stem(cu_name)
    possible = [stem]
    for marker in ("_la_", "_a_"):
        if marker in stem:
            possible.append(stem.split(marker, 1)[1])
    parts = stem.split("_")
    possible.extend("_".join(parts[offset:]) for offset in range(1, len(parts)))
    seen: set[str] = set()
    for candidate in possible:
        if candidate in seen or len(candidate) < 2:
            continue
        seen.add(candidate)
        matches = index.get(candidate, [])
        if matches:
            rule = "exact_stem" if candidate == stem else f"suffix_stem:{candidate}"
            return matches, rule
    return [], None


def observed_members(
    feature: Path, label_meta: dict[str, dict[str, str]]
) -> dict[str, set[str]]:
    members: dict[str, set[str]] = defaultdict(set)
    with gzip.open(feature, "rb") as stream:
        for raw in stream:
            if BLOCK_CU_MARKER not in raw:
                continue
            payload = json.loads(raw)
            label = str(payload.get("library", ""))
            name = str(payload.get("name", ""))
            if not label or not name:
                continue
            if label not in label_meta:
                continue
            members[label].add(name)
    return dict(members)


def source_directories() -> dict[str, Path]:
    return {
        path.name: path
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir()
    } if SOURCE_ROOT.is_dir() else {}


def load_build_source_map(path: Path) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not path.is_file():
        return result
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            result[
                (str(row["source_build"]), normalized_stem(str(row["object"])))
            ].add(str(row["source"]))
    return dict(result)


def build_metadata_candidates(
    root: Path,
    cu_name: str,
    source_names: set[str],
) -> list[Path]:
    """Resolve dependency paths against the local copy of the source tree."""
    result: set[Path] = set()
    for raw in source_names:
        normalized = raw.replace("\\", "/")
        parts = list(Path(normalized).parts)
        if root.name in parts:
            relative = Path(*parts[parts.index(root.name) + 1 :])
        else:
            while parts and parts[0] in ("..", "."):
                parts.pop(0)
            relative = Path(*parts)
        candidate = root / relative
        if candidate.is_file() and normalized_stem(candidate.name) == normalized_stem(
            Path(raw).name
        ):
            result.add(candidate)
    return sorted(result)


def candidate_sources_for_row(
    metadata: dict[str, str], directories: dict[str, Path]
) -> Path | None:
    source = metadata["source"]
    direct = directories.get(source)
    if direct is not None:
        return direct
    package = metadata.get("package", "")
    role = metadata.get("role", "")
    version = str(ROLE_VERSIONS.get(package, {}).get(role, ""))
    if version:
        version = version.split(":", 1)[-1]
        by_role = directories.get(f"{package}-{version}")
        if by_role is not None:
            return by_role
    # Last-resort resolution is accepted only if it is genuinely unique.
    matches = [
        path for name, path in directories.items()
        if name == package or name.startswith(package + "-")
    ]
    return matches[0] if len(matches) == 1 else None


def enrich_label_metadata() -> dict[str, dict[str, str]]:
    metadata = candidate_label_metadata()
    rows = [
        row
        for row in csv.DictReader(
            (UNIFIED / "library_matrix.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
        if row["status"] == "selected" and row["path"]
    ]
    counts = Counter(row["archive"] for row in rows)
    for row in rows:
        label = row["archive"]
        if counts[row["archive"]] > 1:
            identity = "Dataset/builds/libraries/" + row["path"]
            label += "." + hashlib.sha256(identity.encode()).hexdigest()[:16]
        if label in metadata:
            metadata[label]["package"] = row["package"]
            metadata[label]["role"] = row["role"]
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=UNIFIED / "cu_source_equivalence_audit.json",
    )
    parser.add_argument(
        "--equivalence-map",
        type=Path,
        default=UNIFIED / "cu_source_equivalence_map.json",
    )
    parser.add_argument(
        "--build-source-map",
        type=Path,
        default=BUILD_SOURCE_MAP,
        help="Optional compact object-to-source metadata extracted from builds.",
    )
    args = parser.parse_args()

    label_meta = enrich_label_metadata()
    reports = report_index_all()
    members_by_label: dict[str, set[str]] = defaultdict(set)
    inventory_features: list[str] = []
    # Labels are build-specific.  One replay-complete log per compiler provides
    # the archive-member inventory for that compiler's candidate builds.
    for compiler in family.COMPILERS:
        sample = next(iter(reports[compiler].values()))
        feature = sample.with_name(
            sample.name.replace(".report.txt", ".features.jsonl.gz")
        )
        inventory_features.append(str(feature))
        for label, members in observed_members(feature, label_meta).items():
            members_by_label[label].update(members)
    directories = source_directories()
    build_map = load_build_source_map(args.build_source_map)
    indexes: dict[Path, dict[str, list[Path]]] = {}
    file_hashes: dict[Path, str] = {}
    code_hashes: dict[Path, str] = {}
    mappings: list[dict[str, Any]] = []

    # Build one row per logical family/CU/source version, deduplicating
    # compiler and optimization builds of the same source.
    requested: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    metadata_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for label, members in members_by_label.items():
        metadata = label_meta[label]
        for member in members:
            key = (
                metadata["family"],
                ENGINE.normalize_cu_name(member),
                metadata["source"],
            )
            requested[key].add(label)
            metadata_by_key[key] = metadata

    for key in sorted(requested):
        archive_family, cu_name, source = key
        metadata = metadata_by_key[key]
        root = candidate_sources_for_row(metadata, directories)
        candidates: list[Path] = []
        mapping_method: str | None = None
        build_sources: set[str] = set()
        if root is not None:
            if root not in indexes:
                indexes[root] = source_index(root)
            candidates, match_rule = source_candidates(indexes[root], cu_name)
            for source_build in {metadata["source"], root.name}:
                build_sources.update(
                    build_map.get(
                        (source_build, normalized_stem(cu_name)), set()
                    )
                )
            metadata_candidates = build_metadata_candidates(
                root, cu_name, build_sources
            )
            if metadata_candidates:
                candidates = metadata_candidates
                mapping_method = "build_dependency_metadata"
            elif candidates:
                mapping_method = "source_basename"
        else:
            match_rule = None
        hashes: dict[str, list[str]] = defaultdict(list)
        raw_hashes: dict[str, list[str]] = defaultdict(list)
        for candidate in candidates:
            if candidate not in file_hashes:
                file_hashes[candidate] = sha256_file(candidate)
                code_hashes[candidate] = sha256_source_code(candidate)
            relative_candidate = (
                str(candidate.relative_to(root)) if root else str(candidate)
            )
            hashes[code_hashes[candidate]].append(relative_candidate)
            raw_hashes[file_hashes[candidate]].append(relative_candidate)
        if root is None:
            status = "missing_source_tree"
            source_hash = None
        elif not candidates:
            status = "unmapped"
            source_hash = None
        elif len(hashes) == 1:
            status = "mapped"
            source_hash = next(iter(hashes))
        else:
            status = "ambiguous"
            source_hash = None
        mappings.append(
            {
                "family": archive_family,
                "cu": cu_name,
                "source_version": source,
                "package": metadata.get("package", ""),
                "role": metadata.get("role", ""),
                "status": status,
                # Historical compatibility field: from schema v2 this is the
                # normalized-code hash rather than the byte-for-byte file hash.
                "source_sha256": source_hash,
                "source_code_sha256": source_hash,
                "source_tree": str(root) if root else None,
                "match_rule": match_rule,
                "mapping_method": mapping_method,
                "build_metadata_sources": sorted(build_sources),
                "candidates_by_hash": dict(sorted(hashes.items())),
                "candidates_by_file_sha256": dict(sorted(raw_hashes.items())),
                "build_labels": sorted(requested[key]),
            }
        )

    by_family_cu: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mappings:
        by_family_cu[(row["family"], row["cu"])].append(row)
    equivalence_groups: list[dict[str, Any]] = []
    different_groups: list[dict[str, Any]] = []
    for (archive_family, cu_name), rows in sorted(by_family_cu.items()):
        mapped = [row for row in rows if row["status"] == "mapped"]
        versions_by_hash: dict[str, list[str]] = defaultdict(list)
        for row in mapped:
            versions_by_hash[str(row["source_sha256"])].append(
                str(row["source_version"])
            )
        exact = [
            {"source_sha256": digest, "versions": sorted(set(versions))}
            for digest, versions in sorted(versions_by_hash.items())
            if len(set(versions)) >= 2
        ]
        base = {
            "family": archive_family,
            "cu": cu_name,
            "versions_evaluated": sorted(
                {str(row["source_version"]) for row in rows}
            ),
        }
        if exact:
            equivalence_groups.append({**base, "equivalent_groups": exact})
        if len(versions_by_hash) >= 2:
            different_groups.append(
                {
                    **base,
                    "hashes": [
                        {"source_sha256": digest, "versions": sorted(set(versions))}
                        for digest, versions in sorted(versions_by_hash.items())
                    ],
                }
            )

    status_counts = Counter(row["status"] for row in mappings)
    package_coverage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in mappings:
        package_coverage[str(row["package"])][str(row["status"])] += 1
        package_coverage[str(row["package"])]["total"] += 1
    audit = {
        "schema_version": 2,
        "equivalence_rule": (
            "same logical family, normalized CU name, unambiguous mapped source "
            "file, and identical normalized-code SHA-256; C/C++ comments and "
            "insignificant formatting are ignored"
        ),
        "source_root": str(SOURCE_ROOT),
        "source_directories": len(directories),
        "build_source_map": str(args.build_source_map),
        "build_source_map_keys": len(build_map),
        "inventory_features": inventory_features,
        "summary": {
            "cu_source_version_rows": len(mappings),
            "status_counts": dict(status_counts),
            "cross_version_exact_equivalence_groups": len(equivalence_groups),
            "cross_version_different_source_groups": len(different_groups),
        },
        "package_coverage": {
            package: dict(counts)
            for package, counts in sorted(package_coverage.items())
        },
        "exact_cross_version_equivalences": equivalence_groups,
        "different_cross_version_sources": different_groups,
        "mappings": mappings,
    }
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Compact lookup consumed later by Task 2.  Only verified mappings enter.
    equivalence_map = {
        "schema_version": 2,
        "equivalence_rule": audit["equivalence_rule"],
        "identities": [
            {
                "family": row["family"],
                "cu": row["cu"],
                "source_version": row["source_version"],
                "source_sha256": row["source_sha256"],
            }
            for row in mappings
            if row["status"] == "mapped"
        ],
    }
    args.equivalence_map.write_text(
        json.dumps(equivalence_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "audit": str(args.output),
        "equivalence_map": str(args.equivalence_map),
        **audit["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
