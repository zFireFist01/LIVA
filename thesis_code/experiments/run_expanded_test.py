#!/usr/bin/env python3
"""Evaluate frozen greedy-search thresholds on a larger, program-disjoint test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
TEST_DIR = THESIS_DIR / "Test"
REPO_ROOT = THESIS_DIR.parents[1]
DEFAULT_BEST = TEST_DIR / "greedy_threshold_results_9elf_v2_masked"
DEFAULT_OUTPUT = TEST_DIR / "expanded_test_12elf_v2_masked"
DEFAULT_PROGRAMS = ("bash", "less", "openssh", "rsync")
EXPECTED_OPTIMIZATIONS = {"O0", "O2", "O3"}
EXPECTED_COMPILERS = {"gcc", "clang"}
DEFAULT_VARIANTS = {
    "bash_clang-22.1.6_O0",
    "bash_gcc-16.1.1_O2",
    "bash_clang-22.1.6_O3",
    "less_gcc-16.1.1_O0",
    "less_clang-22.1.6_O2",
    "less_gcc-16.1.1_O3",
    "openssh_clang-22.1.6_O0",
    "openssh_gcc-16.1.1_O2",
    "openssh_clang-22.1.6_O3",
    "rsync_gcc-16.1.1_O0",
    "rsync_clang-22.1.6_O2",
    "rsync_gcc-16.1.1_O3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen thresholds on an expanded test without retuning them."
        )
    )
    parser.add_argument(
        "--best-results-dir",
        type=Path,
        default=DEFAULT_BEST,
        help="Directory containing best_thresholds.json from the completed greedy search.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=REPO_ROOT / "GroundTruth",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        help=(
            "Reuse the exact ELF and archive selection recorded by a previous "
            "test_selection.json. Thresholds are still loaded from "
            "--best-results-dir and are never retuned."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--program",
        action="append",
        choices=DEFAULT_PROGRAMS,
        help="Test program to include. Repeatable; defaults to all four programs.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help=(
            "Use all GCC/Clang O0/O2/O3 variants (24 ELF). By default a "
            "balanced 12-ELF subset is used."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--device",
        help="Override the device recorded by the greedy search, for example cuda:0.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def used_programs_from_report(report: dict[str, Any]) -> set[str]:
    """Return every program already used for tuning or internal evaluation."""
    splits = report.get("splits", {})
    if not isinstance(splits, dict):
        return set()
    return {
        str(program)
        for split in splits.values()
        if isinstance(split, dict)
        for program in split.get("programs", [])
    }


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.symlink_to(target)


def selected_test_cases(
    ground_truth_dir: Path,
    programs: set[str],
    all_variants: bool,
) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen_variants: set[str] = set()

    for path in sorted(ground_truth_dir.rglob("ground_truth.json")):
        payload = load_json(path)
        program = str(payload.get("program", path.parents[1].name))
        variant = str(payload.get("variant", path.parent.name))
        compiler_value = str(payload.get("compiler", ""))
        compiler = "clang" if "clang" in compiler_value else "gcc" if "gcc" in compiler_value else ""
        optimization = str(payload.get("elf_optimization", ""))
        binary_value = payload.get("binary")

        if program not in programs:
            continue
        if compiler not in EXPECTED_COMPILERS:
            continue
        if optimization not in EXPECTED_OPTIMIZATIONS:
            continue
        if not all_variants and variant not in DEFAULT_VARIANTS:
            continue
        if not binary_value:
            raise ValueError(f"Missing binary path in {path}")
        if variant in seen_variants:
            raise ValueError(f"Duplicate test variant: {variant}")

        binary = Path(str(binary_value)).resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"Test ELF not found: {binary}")
        seen_variants.add(variant)
        cases.append(
            {
                "program": program,
                "variant": variant,
                "compiler": compiler,
                "elf_optimization": optimization,
                "binary": str(binary),
                "ground_truth": str(path.resolve()),
            }
        )

    expected_count = (
        len(programs) * len(EXPECTED_COMPILERS) * len(EXPECTED_OPTIMIZATIONS)
        if all_variants
        else sum(
            variant.split("_", maxsplit=1)[0] in programs
            for variant in DEFAULT_VARIANTS
        )
    )
    if len(cases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} test ELF variants, found {len(cases)}"
        )
    return sorted(cases, key=lambda case: case["variant"])


def archive_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        *report.get("positive_archives", []),
        *report.get("negative_archives", []),
    ]
    if not records:
        raise ValueError("best_thresholds.json does not contain candidate archives")
    return sorted(records, key=lambda record: str(record["name"]))


def records_from_selection(
    selection_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    selection = load_json(selection_path)
    cases = selection.get("cases")
    archives = selection.get("archives")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No test cases recorded in {selection_path}")
    if not isinstance(archives, list) or not archives:
        raise ValueError(f"No candidate archives recorded in {selection_path}")

    normalized_cases: list[dict[str, str]] = []
    seen_variants: set[str] = set()
    for record in cases:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid test-case record in {selection_path}")
        case = {str(key): str(value) for key, value in record.items()}
        variant = case.get("variant", "")
        binary = Path(case.get("binary", "")).resolve()
        if not variant or variant in seen_variants:
            raise ValueError(f"Missing or duplicate test variant: {variant!r}")
        if not binary.is_file():
            raise FileNotFoundError(f"Test ELF not found: {binary}")
        case["binary"] = str(binary)
        seen_variants.add(variant)
        normalized_cases.append(case)

    normalized_archives: list[dict[str, Any]] = []
    seen_archives: set[str] = set()
    for record in archives:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid archive record in {selection_path}")
        archive_record = dict(record)
        name = str(archive_record.get("name", ""))
        archive = Path(str(archive_record.get("path", ""))).resolve()
        if not name or name in seen_archives:
            raise ValueError(f"Missing or duplicate archive name: {name!r}")
        if not archive.is_file():
            raise FileNotFoundError(f"Candidate archive not found: {archive}")
        archive_record["name"] = name
        archive_record["path"] = str(archive)
        seen_archives.add(name)
        normalized_archives.append(archive_record)

    return (
        sorted(normalized_cases, key=lambda case: case["variant"]),
        sorted(normalized_archives, key=lambda record: str(record["name"])),
    )


def stage_inputs(
    output_dir: Path,
    cases: list[dict[str, str]],
    archives: list[dict[str, Any]],
) -> tuple[Path, Path]:
    elf_dir = output_dir / "inputs" / "elfs"
    lib_dir = output_dir / "inputs" / "libs"
    elf_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    expected_elf_names = {case["variant"] for case in cases}
    expected_lib_names = {str(record["name"]) for record in archives}
    for path in elf_dir.iterdir():
        if path.name not in expected_elf_names:
            path.unlink()
    for path in lib_dir.iterdir():
        if path.name not in expected_lib_names:
            path.unlink()

    for case in cases:
        replace_symlink(elf_dir / case["variant"], Path(case["binary"]))
    for record in archives:
        archive = Path(str(record["path"])).resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Candidate archive not found: {archive}")
        replace_symlink(lib_dir / str(record["name"]), archive)
    return elf_dir, lib_dir


def batch_command(
    args: argparse.Namespace,
    report: dict[str, Any],
    params: dict[str, Any],
    elf_dir: Path,
    lib_dir: Path,
) -> list[str]:
    command = [
        args.python,
        str(THESIS_DIR / "run_libseeker_batch.py"),
        "--pipeline", "current",
        "--dataset-dir", str(elf_dir),
        "--libs-dir", str(lib_dir),
        "--output-dir", str(args.output_dir),
        "--ground-truth-dir", str(args.ground_truth_dir),
        "--all-elfs",
        "--all-libs",
        "--python", args.python,
        "--device", args.device or str(report.get("device", "auto")),
        "--asm-normalization", str(report.get("asm_normalization", "v2")),
        "--palmtree-pooling", str(report.get("palmtree_pooling", "masked_mean")),
        "--library-score-aggregator",
        str(params.get("library_score_aggregator", "mean")),
    ]
    mappings = {
        "block_threshold": "--block-threshold",
        "library_min_score": "--library-min-score",
        "block_coverage_mean_threshold": "--block-coverage-mean-threshold",
        "block_assignment_quality_threshold": "--block-assignment-quality-threshold",
        "block_min_assignment_ratio": "--block-min-assignment-ratio",
        "block_min_coverage_ratio": "--block-min-coverage-ratio",
        "block_locality_window_multiplier": "--block-locality-window-multiplier",
        "block_locality_window_padding": "--block-locality-window-padding",
        "block_min_call_edge_ratio": "--block-min-call-edge-ratio",
        "block_min_function_concentration": "--block-min-function-concentration",
        "block_min_function_spread": "--block-min-function-spread",
        "cu_min_function_coverage": "--cu-min-function-coverage",
        "rodata_min_bytes": "--rodata-min-bytes",
        "rodata_min_strings": "--rodata-min-strings",
        "rodata_min_ngrams": "--rodata-min-ngrams",
        "rodata_penalty_threshold": "--rodata-penalty-threshold",
        "rodata_confirm_threshold": "--rodata-confirm-threshold",
        "rodata_bonus_weight": "--rodata-bonus-weight",
        "cross_cu_call_bonus_weight": "--cross-cu-call-bonus-weight",
        "cross_cu_call_penalty_weight": "--cross-cu-call-penalty-weight",
        "cross_cu_call_saturation_edges": "--cross-cu-call-saturation-edges",
    }
    for key, option in mappings.items():
        defaults = {
            "rodata_confirm_threshold": 0.70,
            "rodata_bonus_weight": 0.30,
            "cross_cu_call_bonus_weight": 0.0,
            "cross_cu_call_penalty_weight": 0.0,
            "cross_cu_call_saturation_edges": 3,
            "cu_min_function_coverage": 0.0,
        }
        command.extend([option, str(params.get(key, defaults.get(key)))])
    if not bool(params.get("rodata_filter_enabled", 1)):
        command.append("--disable-rodata-filter")
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    args.best_results_dir = args.best_results_dir.resolve()
    args.ground_truth_dir = args.ground_truth_dir.resolve()
    if args.selection_file is not None:
        args.selection_file = args.selection_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = load_json(args.best_results_dir / "best_thresholds.json")
    params = dict(report["parameters"])
    params.setdefault("cross_cu_call_bonus_weight", 0.0)
    params.setdefault("cross_cu_call_penalty_weight", 0.0)
    params.setdefault("cross_cu_call_saturation_edges", 3)
    used_programs = used_programs_from_report(report)
    if args.selection_file is not None:
        if args.program or args.all_variants:
            raise ValueError(
                "--selection-file cannot be combined with --program or "
                "--all-variants"
            )
        cases, archives = records_from_selection(args.selection_file)
        programs = {case["program"] for case in cases}
        overlap = programs & used_programs
    else:
        requested_programs = set(args.program or DEFAULT_PROGRAMS)
        overlap = requested_programs & used_programs
        if overlap and args.program:
            raise ValueError(
                "Expanded test overlaps programs already used by the greedy search: "
                + ", ".join(sorted(overlap))
            )
        programs = requested_programs - used_programs
        if not programs:
            raise ValueError(
                "No program-disjoint expanded-test program remains after excluding: "
                + ", ".join(sorted(used_programs))
            )
        if overlap:
            print(
                "[test-skip] already used by greedy search: "
                + ", ".join(sorted(overlap))
            )
        cases = selected_test_cases(
            args.ground_truth_dir,
            programs,
            args.all_variants,
        )
        archives = archive_records(report)

    elf_dir, lib_dir = stage_inputs(args.output_dir, cases, archives)
    selection = {
        "frozen_thresholds_source": str(
            (args.best_results_dir / "best_thresholds.json").resolve()
        ),
        "best_profile_id": report.get("best_profile_id"),
        "thresholds_retuned": False,
        "selection_source": (
            str(args.selection_file) if args.selection_file is not None else None
        ),
        "program_disjoint": not overlap,
        "excluded_greedy_programs": sorted(used_programs),
        "programs": sorted(programs),
        "all_variants": args.all_variants,
        "elf_count": len(cases),
        "library_count": len(archives),
        "pair_count": len(cases) * len(archives),
        "cases": cases,
        "archives": archives,
        "parameters": params,
    }
    (args.output_dir / "test_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    command = batch_command(args, report, params, elf_dir, lib_dir)
    print(
        f"Expanded test: {len(cases)} ELF, {len(archives)} libraries, "
        f"{len(cases) * len(archives)} decisions"
    )
    print("Programs:", ", ".join(sorted(programs)))
    print("Thresholds: frozen (no test-time tuning)")
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=THESIS_DIR).returncode


if __name__ == "__main__":
    raise SystemExit(main())
