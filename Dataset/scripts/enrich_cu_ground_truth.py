#!/usr/bin/env python3
"""Populate compilation-unit ground truth for generated static ELF cases."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GROUND_TRUTH_ROOT = PROJECT_ROOT / "GroundTruth"
ELF_BUILD_ROOT = PROJECT_ROOT / "Dataset/builds/elf_builds"

LINKER_MAP_MEMBER_RE = re.compile(
    r"(?P<archive>.+?\.a)\((?P<member>[^()]+)\)"
)
NM_ARCHIVE_RE = re.compile(
    r"^(?P<archive>.+?\.a):(?P<member>[^:]+):"
    r"(?:[0-9A-Fa-f]+)?\s*(?P<type>[A-Za-z])\s+(?P<symbol>\S+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=GROUND_TRUTH_ROOT,
        help=f"GroundTruth root to enrich (default: {GROUND_TRUTH_ROOT}).",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=ELF_BUILD_ROOT,
        help=f"ELF build root containing randomized_matrix.json (default: {ELF_BUILD_ROOT}).",
    )
    parser.add_argument(
        "--program",
        action="append",
        help="Limit enrichment to a program. Can be repeated.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Limit enrichment to a variant directory. Can be repeated.",
    )
    parser.add_argument(
        "--symbol-fallback",
        action="store_true",
        help=(
            "Infer included archive members from ELF/archive symbol tables when "
            "the linker map has no lib.a(member.o) entries. This is useful for "
            "diagnostics, but should not be used for official ground truth."
        ),
    )
    parser.add_argument(
        "--no-symbol-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-build-info",
        action="store_true",
        help="Do not update build-info.json next to generated ELF files.",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Do not update randomized_matrix.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_stdout(command: list[str]) -> str:
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0 and not process.stdout:
        return ""
    return process.stdout


def parse_linker_map(map_file: Path) -> dict[Path, set[str]]:
    included: dict[Path, set[str]] = defaultdict(set)
    if not map_file.is_file():
        return included

    in_archive_section = False
    for line in map_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Archive member included"):
            in_archive_section = True
            continue
        if in_archive_section and line.startswith("Discarded input sections"):
            break
        if not in_archive_section:
            continue

        match = LINKER_MAP_MEMBER_RE.match(line.strip())
        if not match:
            continue
        archive = Path(match.group("archive")).resolve()
        included[archive].add(match.group("member"))
    return included


def archive_members(archive: Path) -> list[str]:
    output = run_stdout(["ar", "t", str(archive)])
    return [line.strip() for line in output.splitlines() if line.strip()]


_ARCHIVE_CALL_GRAPH_CACHE: dict[Path, tuple[dict[str, Any], ...]] = {}


def _function_ranges(
    elf: ELFFile,
    symbol_table,
) -> dict[int, list[tuple[int, int, str]]]:
    functions_by_section: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for symbol in symbol_table.iter_symbols():
        section_index = symbol["st_shndx"]
        symbol_name = symbol.name
        if (
            symbol["st_info"]["type"] != "STT_FUNC"
            or not isinstance(section_index, int)
            or not symbol_name
        ):
            continue
        functions_by_section[section_index].append(
            (
                int(symbol["st_value"]),
                int(symbol["st_size"]),
                symbol_name,
            )
        )

    ranges: dict[int, list[tuple[int, int, str]]] = {}
    for section_index, functions in functions_by_section.items():
        functions.sort()
        section_size = int(elf.get_section(section_index).data_size)
        ranges[section_index] = [
            (
                start,
                start + size
                if size > 0
                else (
                    functions[index + 1][0]
                    if index + 1 < len(functions)
                    else section_size
                ),
                name,
            )
            for index, (start, size, name) in enumerate(functions)
        ]
    return ranges


def archive_inter_cu_calls(archive: Path) -> tuple[dict[str, Any], ...]:
    """Extract unique x86 direct-call edges between object members.

    Relocation symbols are used only to resolve the reference archive's
    topology. The resulting ground-truth records are later filtered to members
    positively confirmed by the final linker map.
    """
    archive = archive.resolve()
    cached = _ARCHIVE_CALL_GRAPH_CACHE.get(archive)
    if cached is not None:
        return cached

    definitions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unresolved_calls: list[tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory(prefix="cu-call-graph.") as tmpdir:
        process = subprocess.run(
            ["ar", "x", str(archive)],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode != 0:
            _ARCHIVE_CALL_GRAPH_CACHE[archive] = ()
            return ()

        for object_path in sorted(Path(tmpdir).iterdir()):
            if not object_path.is_file():
                continue
            try:
                with object_path.open("rb") as stream:
                    elf = ELFFile(stream)
                    symbol_table = elf.get_section_by_name(".symtab")
                    if symbol_table is None:
                        continue

                    function_ranges = _function_ranges(
                        elf,
                        symbol_table,
                    )
                    for symbol in symbol_table.iter_symbols():
                        section_index = symbol["st_shndx"]
                        if (
                            symbol["st_info"]["type"] == "STT_FUNC"
                            and symbol["st_info"]["bind"]
                            in {"STB_GLOBAL", "STB_WEAK"}
                            and isinstance(section_index, int)
                            and symbol.name
                        ):
                            definitions[symbol.name].append(
                                (object_path.name, symbol.name)
                            )

                    for relocation_section in elf.iter_sections():
                        if not isinstance(relocation_section, RelocationSection):
                            continue
                        target_section_index = int(
                            relocation_section["sh_info"]
                        )
                        target_section = elf.get_section(target_section_index)
                        if not (int(target_section["sh_flags"]) & 0x4):
                            continue

                        section_bytes = target_section.data()
                        relocation_symbols = elf.get_section(
                            int(relocation_section["sh_link"])
                        )
                        for relocation in relocation_section.iter_relocations():
                            offset = int(relocation["r_offset"])
                            symbol = relocation_symbols.get_symbol(
                                int(relocation["r_info_sym"])
                            )
                            if (
                                not symbol.name
                                or symbol["st_shndx"] != "SHN_UNDEF"
                                or offset < 1
                                or offset > len(section_bytes)
                                # x86/x86-64 direct near CALL rel32: E8 + imm32.
                                or section_bytes[offset - 1] != 0xE8
                            ):
                                continue
                            caller_function = next(
                                (
                                    name
                                    for start, stop, name in function_ranges.get(
                                        target_section_index,
                                        [],
                                    )
                                    if start <= offset < stop
                                ),
                                None,
                            )
                            if caller_function is not None:
                                unresolved_calls.append(
                                    (
                                        object_path.name,
                                        caller_function,
                                        symbol.name,
                                    )
                                )
            except Exception:
                # A static archive may also contain non-ELF metadata or LTO
                # members. Those members simply provide no direct-call labels.
                continue

    edge_counts: Counter[tuple[str, str, str, str]] = Counter()
    for caller_cu, caller_function, callee_symbol in unresolved_calls:
        candidates = [
            candidate
            for candidate in definitions.get(callee_symbol, [])
            if candidate[0] != caller_cu
        ]
        if len(candidates) != 1:
            continue
        callee_cu, callee_function = candidates[0]
        edge_counts[
            (
                caller_cu,
                caller_function,
                callee_cu,
                callee_function,
            )
        ] += 1

    result = tuple(
        {
            "caller_cu": caller_cu,
            "caller_function": caller_function,
            "callee_cu": callee_cu,
            "callee_function": callee_function,
            "direct_call_relocations": count,
        }
        for (
            caller_cu,
            caller_function,
            callee_cu,
            callee_function,
        ), count in sorted(edge_counts.items())
    )
    if len(_ARCHIVE_CALL_GRAPH_CACHE) >= 8:
        _ARCHIVE_CALL_GRAPH_CACHE.pop(next(iter(_ARCHIVE_CALL_GRAPH_CACHE)))
    _ARCHIVE_CALL_GRAPH_CACHE[archive] = result
    return result


def expected_inter_cu_ground_truth(
    archive: Path,
    included_members: set[str],
) -> dict[str, Any]:
    """Return expected calls whose endpoint CUs were both linked."""
    edges = [
        edge
        for edge in archive_inter_cu_calls(archive)
        if (
            edge["caller_cu"] in included_members
            and edge["callee_cu"] in included_members
        )
    ]
    cu_edge_counts: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        cu_edge_counts[(edge["caller_cu"], edge["callee_cu"])] += int(
            edge["direct_call_relocations"]
        )
    return {
        "inter_cu_ground_truth_method": (
            "archive_direct_call_relocation_between_linker_map_members"
        ),
        "expected_inter_cu_calls": edges,
        "expected_inter_cu_function_edges": len(edges),
        "expected_inter_cu_call_relocations": sum(
            int(edge["direct_call_relocations"]) for edge in edges
        ),
        "expected_inter_cu_cu_edges": [
            {
                "caller_cu": caller_cu,
                "callee_cu": callee_cu,
                "direct_call_relocations": count,
            }
            for (caller_cu, callee_cu), count in sorted(cu_edge_counts.items())
        ],
    }


def useful_symbol(symbol_type: str, symbol: str) -> bool:
    symbol_type = symbol_type.upper()
    if symbol_type in {"U", "W", "V"}:
        return False
    if not symbol or symbol.startswith("$"):
        return False
    return True


def binary_defined_symbols(binary: Path) -> set[str]:
    symbols: set[str] = set()
    for line in run_stdout(["nm", "-g", "--defined-only", str(binary)]).splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        symbol_type = parts[-2] if len(parts) >= 3 else parts[0]
        symbol = parts[-1]
        if useful_symbol(symbol_type, symbol):
            symbols.add(symbol)
    return symbols


def archive_member_symbols(archive: Path) -> dict[str, set[str]]:
    members: dict[str, set[str]] = defaultdict(set)
    for line in run_stdout(["nm", "-A", "-g", "--defined-only", str(archive)]).splitlines():
        match = NM_ARCHIVE_RE.match(line)
        if not match:
            continue
        symbol_type = match.group("type")
        symbol = match.group("symbol")
        if useful_symbol(symbol_type, symbol):
            members[match.group("member")].add(symbol)
    return members


def symbol_inferred_members(archive: Path, binary_symbols: set[str]) -> set[str]:
    member_symbols = archive_member_symbols(archive)
    symbol_counts = Counter(
        symbol for symbols in member_symbols.values() for symbol in symbols
    )
    unique_symbols = {
        symbol for symbol, count in symbol_counts.items() if count == 1
    }
    included = set()
    for member, symbols in member_symbols.items():
        if symbols & binary_symbols & unique_symbols:
            included.add(member)
    return included


def enrich_archives(
    archives: list[dict[str, Any]],
    linker_map: Path,
    binary: Path,
    use_symbol_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    map_members = parse_linker_map(linker_map)
    binary_symbols = (
        binary_defined_symbols(binary)
        if use_symbol_fallback and binary.is_file()
        else set()
    )
    stats = {
        "archives": 0,
        "map_archives": 0,
        "symbol_archives": 0,
        "empty_archives": 0,
        "included_cu": 0,
        "total_cu": 0,
    }
    enriched = []

    for metadata in archives:
        archive = Path(str(metadata["archive"])).resolve()
        members = archive_members(archive) if archive.is_file() else []
        included = set(map_members.get(archive, set()))
        method = "linker_map" if included else "none"

        if not included and use_symbol_fallback and archive.is_file():
            included = symbol_inferred_members(archive, binary_symbols)
            if included:
                method = "symbol_table"

        compilation_units = [
            {
                "compilation_unit": member,
                "included": member in included,
                "ground_truth_method": method if member in included else "not_included",
            }
            for member in members
        ]
        confirmed_compilation_units = (
            sorted(included)
            if method == "linker_map"
            else []
        )
        included_count = sum(unit["included"] for unit in compilation_units)
        total_count = len(compilation_units)
        enriched_archive = {
            **metadata,
            "archive": str(archive),
            "included_compilation_units": included_count,
            "total_compilation_units": total_count,
            "cu_ground_truth_method": method,
            # Positive-only view used by matching evaluation. Keep the full
            # member inventory above for audit, but never treat an unconfirmed
            # member as a positive CU label.
            "confirmed_compilation_units": confirmed_compilation_units,
            "confirmed_compilation_units_method": (
                "linker_map" if confirmed_compilation_units else "none"
            ),
            "compilation_units": compilation_units,
            **(
                expected_inter_cu_ground_truth(archive, included)
                if archive.is_file() and method == "linker_map"
                else {
                    "inter_cu_ground_truth_method": "none",
                    "expected_inter_cu_calls": [],
                    "expected_inter_cu_function_edges": 0,
                    "expected_inter_cu_call_relocations": 0,
                    "expected_inter_cu_cu_edges": [],
                }
            ),
        }
        enriched.append(enriched_archive)

        stats["archives"] += 1
        stats["included_cu"] += included_count
        stats["total_cu"] += total_count
        if method == "linker_map":
            stats["map_archives"] += 1
        elif method == "symbol_table":
            stats["symbol_archives"] += 1
        else:
            stats["empty_archives"] += 1

    return enriched, stats


def library_ground_truth_summary(archives: list[dict[str, Any]]) -> dict[str, Any]:
    libraries: dict[str, dict[str, Any]] = {}
    for archive in archives:
        library = str(archive.get("library", ""))
        if not library:
            continue
        entry = libraries.setdefault(
            library,
            {
                "present": False,
                "archives": 0,
                "included_archives": 0,
                "included_compilation_units": 0,
                "total_compilation_units": 0,
                "sources": [],
                "optimizations": [],
            },
        )
        included_cu = int(archive.get("included_compilation_units", 0) or 0)
        total_cu = int(archive.get("total_compilation_units", 0) or 0)
        entry["archives"] += 1
        entry["included_archives"] += int(included_cu > 0)
        entry["included_compilation_units"] += included_cu
        entry["total_compilation_units"] += total_cu
        entry["present"] = entry["present"] or included_cu > 0
        source = archive.get("source")
        optimization = archive.get("optimization")
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        if optimization and optimization not in entry["optimizations"]:
            entry["optimizations"].append(optimization)

    present = sorted(
        library for library, record in libraries.items() if record["present"]
    )
    absent = sorted(
        library for library, record in libraries.items() if not record["present"]
    )
    return {
        "method": "linker_map_included_compilation_units",
        "present_libraries": present,
        "absent_libraries": absent,
        "libraries": libraries,
    }


def inter_cu_ground_truth_summary(
    archives: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "method": "archive_direct_call_relocation_between_linker_map_members",
        "archives_with_edges": sum(
            int(archive.get("expected_inter_cu_function_edges", 0) or 0) > 0
            for archive in archives
        ),
        "function_edges": sum(
            int(archive.get("expected_inter_cu_function_edges", 0) or 0)
            for archive in archives
        ),
        "direct_call_relocations": sum(
            int(archive.get("expected_inter_cu_call_relocations", 0) or 0)
            for archive in archives
        ),
        "cu_edges": sum(
            len(archive.get("expected_inter_cu_cu_edges", []))
            for archive in archives
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ground_truth_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.ground_truth_root.glob("*/*/ground_truth.json"))
    if args.program:
        programs = set(args.program)
        paths = [path for path in paths if path.parent.parent.name in programs]
    if args.variant:
        variants = set(args.variant)
        paths = [path for path in paths if path.parent.name in variants]
    return paths


def update_build_info(payload: dict[str, Any], dry_run: bool) -> bool:
    binary = Path(str(payload.get("binary", "")))
    build_info = binary.parent / "build-info.json"
    if not build_info.is_file():
        return False
    build_payload = read_json(build_info)
    build_payload["archives"] = payload["archives"]
    build_payload["cu_ground_truth"] = payload["cu_ground_truth"]
    build_payload["library_ground_truth"] = payload["library_ground_truth"]
    build_payload["inter_cu_ground_truth"] = payload[
        "inter_cu_ground_truth"
    ]
    write_json(build_info, build_payload, dry_run)
    return True


def update_matrix(build_root: Path, cases_by_variant: dict[str, dict[str, Any]], dry_run: bool) -> bool:
    matrix_path = build_root / "randomized_matrix.json"
    if not matrix_path.is_file():
        return False
    matrix = read_json(matrix_path)
    changed = False
    for case in matrix.get("cases", []):
        variant = case.get("variant")
        if variant in cases_by_variant:
            case["archives"] = cases_by_variant[variant]["archives"]
            case["cu_ground_truth"] = cases_by_variant[variant]["cu_ground_truth"]
            case["library_ground_truth"] = cases_by_variant[variant][
                "library_ground_truth"
            ]
            case["inter_cu_ground_truth"] = cases_by_variant[variant][
                "inter_cu_ground_truth"
            ]
            changed = True
    if changed:
        write_json(matrix_path, matrix, dry_run)
    return changed


def main() -> int:
    args = parse_args()
    for command in ("ar", "nm"):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"Missing command: {command}")

    totals = defaultdict(int)
    enriched_cases: dict[str, dict[str, Any]] = {}
    paths = ground_truth_paths(args)
    use_symbol_fallback = args.symbol_fallback and not args.no_symbol_fallback

    for path in paths:
        payload = read_json(path)
        archives = payload.get("archives", [])
        linker_map = Path(str(payload.get("linker_map", "")))
        binary = Path(str(payload.get("binary", "")))
        enriched_archives, stats = enrich_archives(
            archives,
            linker_map,
            binary,
            use_symbol_fallback,
        )
        payload["archives"] = enriched_archives
        payload["cu_ground_truth"] = {
            "method": "linker_map_with_symbol_fallback"
            if use_symbol_fallback
            else "linker_map",
            "included_compilation_units": stats["included_cu"],
            "total_compilation_units": stats["total_cu"],
            "map_archives": stats["map_archives"],
            "symbol_table_archives": stats["symbol_archives"],
            "empty_archives": stats["empty_archives"],
        }
        payload["library_ground_truth"] = library_ground_truth_summary(
            enriched_archives
        )
        payload["inter_cu_ground_truth"] = inter_cu_ground_truth_summary(
            enriched_archives
        )
        write_json(path, payload, args.dry_run)
        if not args.skip_build_info:
            update_build_info(payload, args.dry_run)
        enriched_cases[str(payload.get("variant"))] = payload

        for key, value in stats.items():
            totals[key] += value
        totals["cases"] += 1
        print(
            f"{payload.get('variant')}: "
            f"cu={stats['included_cu']}/{stats['total_cu']} "
            f"map_archives={stats['map_archives']} "
            f"symbol_archives={stats['symbol_archives']} "
            f"empty_archives={stats['empty_archives']}"
        )

    matrix_updated = False
    if not args.skip_matrix:
        matrix_updated = update_matrix(args.build_root, enriched_cases, args.dry_run)

    print(
        "\nSummary: "
        f"cases={totals['cases']} archives={totals['archives']} "
        f"included_cu={totals['included_cu']} total_cu={totals['total_cu']} "
        f"map_archives={totals['map_archives']} "
        f"symbol_archives={totals['symbol_archives']} "
        f"empty_archives={totals['empty_archives']} "
        f"matrix_updated={matrix_updated}"
    )
    if args.dry_run:
        print("Dry run: no files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
