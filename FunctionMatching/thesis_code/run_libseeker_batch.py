#!/usr/bin/env python3
"""Run libseeker batch experiments with and without block matching.

The script runs main.py twice for each selected ELF:
  1. the old function-matching-only pipeline (--disable_block_matching)
  2. the current function matching + block-refinement pipeline, including a lower-score
     recovery band for CUs that should still be checked at block level

It writes raw logs plus CSV summaries that make the two modes easy to compare.
Run it from the thesis-code conda environment.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "Exploration" / "libseeker_repo" / "exp_dataset3"
DEFAULT_LIBS_DIR = REPO_ROOT / "Exploration" / "libseeker_repo" / "build_lib" / "libs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "libseeker_batch_results"

DEFAULT_ELF_PATTERNS = [
    "grep.gcc.O0",
    "grep.gcc.O2",
    "bash.gcc.O0",
    "tar.gcc.O0",
    "less.gcc.O0",
]

DEFAULT_LIBRARY_NAMES = [
    "libpcre2-8.a",
    "libz.a",
    "libbz2.a",
    "liblzma.a",
    "libreadline.a",
    "libncurses.a",
    "libssl.a",
    "libcrypto.a",
    "libacl.a",
    "libattr.a",
    "libselinux.a",
    "libcap.a",
    "libmount.a",
    "libblkid.a",
    "libuuid.a",
    "libsmartcols.a",
]

SUMMARY_FIELDS = [
    "binary",
    "mode",
    "library",
    "status",
    "score",
    "function_best",
    "block_best",
    "block_scope",
    "candidate_cu",
    "primary_candidate_cu",
    "recovery_candidate_cu",
    "matched_cu",
    "total_cu",
    "matched_functions",
    "time",
    "log_path",
]

COMPARISON_FIELDS = [
    "binary",
    "library",
    "old_status",
    "new_status",
    "old_score",
    "new_score",
    "old_matched_cu",
    "new_matched_cu",
    "old_matched_functions",
    "new_matched_functions",
    "new_candidate_cu",
    "new_primary_candidate_cu",
    "new_recovery_candidate_cu",
    "new_block_scope",
    "function_best",
    "block_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-test libseeker ELF files against static libraries.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--libs-dir", type=Path, default=DEFAULT_LIBS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--elf",
        action="append",
        default=[],
        help="ELF filename or glob. Repeatable. Defaults to a small curated set.",
    )
    parser.add_argument(
        "--lib",
        action="append",
        default=[],
        help="Library filename or glob. Repeatable. Defaults to a curated libseeker subset.",
    )
    parser.add_argument(
        "--all-elfs",
        action="store_true",
        help="Use every file in dataset-dir unless limited by --elf-limit.",
    )
    parser.add_argument(
        "--all-libs",
        action="store_true",
        help="Use every .a file in libs-dir unless limited by --lib-limit.",
    )
    parser.add_argument("--elf-limit", type=int, default=0)
    parser.add_argument("--lib-limit", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("both", "old", "block"),
        default="both",
        help="Which pipeline(s) to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a run if the expected raw log already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-run timeout in seconds. 0 disables subprocess timeout.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for main.py. Defaults to current interpreter.",
    )
    parser.add_argument("--function-threshold", type=float, default=0.8)
    parser.add_argument("--candidate-low-threshold", type=float, default=0.45)
    parser.add_argument(
        "--block-scope",
        choices=("all_cu", "function_candidates"),
        default="all_cu",
        help="Use all eligible CUs for block matching, or only CUs selected by function matching.",
    )
    parser.add_argument("--block-threshold", type=float, default=0.70)
    parser.add_argument("--block-coverage-mean-threshold", type=float, default=0.78)
    parser.add_argument("--block-assignment-threshold", type=float, default=0.75)
    parser.add_argument("--block-min-coverage-ratio", type=float, default=0.50)
    parser.add_argument("--block-locality-window-multiplier", type=float, default=3.0)
    parser.add_argument("--block-locality-window-padding", type=int, default=2)
    parser.add_argument("--block-min-edge-locality-ratio", type=float, default=0.5)
    parser.add_argument("--block-min-instructions", type=int, default=3)
    parser.add_argument("--block-min-function-concentration", type=float, default=0.45)
    parser.add_argument("--block-min-function-spread", type=float, default=0.50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected commands without running them.",
    )
    return parser.parse_args()


def resolve_patterns(base_dir: Path, patterns: list[str], default_patterns: list[str]) -> list[Path]:
    selected: list[Path] = []
    seen = set()

    for pattern in patterns or default_patterns:
        matches = sorted(base_dir.glob(pattern))
        direct = base_dir / pattern
        if not matches and direct.exists():
            matches = [direct]

        for match in matches:
            if not match.is_file():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            selected.append(resolved)

    return selected


def select_elfs(args: argparse.Namespace) -> list[Path]:
    if args.all_elfs:
        elfs = sorted(path.resolve() for path in args.dataset_dir.iterdir() if path.is_file())
    else:
        elfs = resolve_patterns(args.dataset_dir, args.elf, DEFAULT_ELF_PATTERNS)

    if args.elf_limit > 0:
        elfs = elfs[: args.elf_limit]

    return elfs


def select_libs(args: argparse.Namespace) -> list[Path]:
    if args.all_libs:
        libs = sorted(path.resolve() for path in args.libs_dir.glob("*.a") if path.is_file())
    else:
        libs = resolve_patterns(args.libs_dir, args.lib, DEFAULT_LIBRARY_NAMES)

    if args.lib_limit > 0:
        libs = libs[: args.lib_limit]

    return libs


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)


def make_libraries_dir(libraries: list[Path]) -> tempfile.TemporaryDirectory:
    tmpdir = tempfile.TemporaryDirectory(prefix="libseeker-batch-libs.")
    tmp_path = Path(tmpdir.name)

    for library in libraries:
        link_path = tmp_path / library.name
        try:
            link_path.symlink_to(library)
        except OSError:
            shutil.copy2(library, link_path)

    return tmpdir


def run_command(command: list[str], cwd: Path, log_path: Path, timeout: int, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout if timeout > 0 else None,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - started))
            log_file.write(f"\n[TIMEOUT] after {elapsed}\n")
            return 124


def main_command(
    args: argparse.Namespace,
    binary: Path,
    libraries_dir: Path,
    output_path: Path,
    mode: str,
) -> list[str]:
    command = [
        args.python,
        "main.py",
        "--hide-warnings",
        "--path_to_binary",
        binary.as_posix(),
        "--libraries_dir",
        libraries_dir.as_posix(),
        "--output",
        output_path.as_posix(),
    ]

    if mode == "old":
        command.append("--disable_block_matching")
    else:
        command.extend(
            [
                "--function_threshold",
                str(args.function_threshold),
                "--candidate_low_threshold",
                str(args.candidate_low_threshold),
                "--block_scope",
                args.block_scope,
                "--block_threshold",
                str(args.block_threshold),
                "--block_assignment_threshold",
                str(args.block_assignment_threshold),
                "--block_min_coverage_ratio",
                str(args.block_min_coverage_ratio),
                "--block_locality_window_multiplier",
                str(args.block_locality_window_multiplier),
                "--block_locality_window_padding",
                str(args.block_locality_window_padding),
                "--block_min_edge_locality_ratio",
                str(args.block_min_edge_locality_ratio),
                "--block_min_instructions",
                str(args.block_min_instructions),
                "--block_min_function_concentration",
                str(args.block_min_function_concentration),
                "--block_min_function_spread",
                str(args.block_min_function_spread),
            ]
        )
        if args.block_coverage_mean_threshold is not None:
            command.extend(
                [
                    "--block_coverage_mean_threshold",
                    str(args.block_coverage_mean_threshold),
                ]
            )

    return command


def parse_fraction(raw: str | None) -> tuple[str, str]:
    if raw is None:
        return "", ""

    if "/" not in raw:
        return raw, ""

    left, right = raw.split("/", maxsplit=1)
    return left, right


def parse_percent(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.rstrip("%")


def normalize_block_scope(raw: str | None) -> str:
    if raw == "coarse":
        return "function_candidates"
    return raw or ""


def parse_report(report_path: Path, binary: Path, mode: str, log_path: Path) -> list[dict[str, str]]:
    if not report_path.exists():
        return []

    rows = []
    line_re = re.compile(r"^(?P<status>YES \[W\]|YES|NO)\s+\|\s+(?P<fields>.+)$")

    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = line_re.match(line)
        if not match:
            continue

        values: dict[str, str] = {}
        for part in match.group("fields").split(" | "):
            if "=" not in part:
                continue
            key, value = part.split("=", maxsplit=1)
            values[key.strip()] = value.strip()

        matched_cu, total_cu = parse_fraction(values.get("matched_cu"))
        row = {
            "binary": binary.name,
            "mode": mode,
            "library": values.get("library", ""),
            "status": match.group("status").strip(),
            "score": parse_percent(values.get("score")),
            "function_best": parse_percent(
                values.get("function_best") or values.get("coarse_best")
            ),
            "block_best": parse_percent(values.get("block_best")),
            "block_scope": normalize_block_scope(
                values.get("block_scope", values.get("block_candidate_mode"))
            ),
            "candidate_cu": values.get("candidate_cu", ""),
            "primary_candidate_cu": values.get("primary_candidate_cu", ""),
            "recovery_candidate_cu": values.get("recovery_candidate_cu", ""),
            "matched_cu": matched_cu,
            "total_cu": total_cu,
            "matched_functions": values.get("matched_functions", ""),
            "time": values.get("time", ""),
            "log_path": log_path.as_posix(),
        }
        rows.append(row)

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def comparison_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    old_rows = {
        (row["binary"], row["library"]): row
        for row in summary_rows
        if row["mode"] == "old"
    }
    block_rows = {
        (row["binary"], row["library"]): row
        for row in summary_rows
        if row["mode"] == "block"
    }
    keys = sorted(set(old_rows) | set(block_rows))

    rows = []
    for key in keys:
        old = old_rows.get(key, {})
        block = block_rows.get(key, {})
        rows.append(
            {
                "binary": key[0],
                "library": key[1],
                "old_status": old.get("status", ""),
                "new_status": block.get("status", ""),
                "old_score": old.get("score", ""),
                "new_score": block.get("score", ""),
                "old_matched_cu": old.get("matched_cu", ""),
                "new_matched_cu": block.get("matched_cu", ""),
                "old_matched_functions": old.get("matched_functions", ""),
                "new_matched_functions": block.get("matched_functions", ""),
                "new_candidate_cu": block.get("candidate_cu", ""),
                "new_primary_candidate_cu": block.get("primary_candidate_cu", ""),
                "new_recovery_candidate_cu": block.get("recovery_candidate_cu", ""),
                "new_block_scope": block.get("block_scope", ""),
                "function_best": block.get("function_best") or old.get("function_best", ""),
                "block_best": block.get("block_best", ""),
            }
        )

    return rows


def run_batch(args: argparse.Namespace) -> int:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if not args.libs_dir.is_dir():
        raise FileNotFoundError(f"Libraries directory not found: {args.libs_dir}")

    elfs = select_elfs(args)
    libraries = select_libs(args)
    if not elfs:
        raise ValueError("No ELF files selected")
    if not libraries:
        raise ValueError("No libraries selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    reports_dir = args.output_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    modes = ["old", "block"] if args.mode == "both" else [args.mode]

    print(f"Selected {len(elfs)} ELF file(s)")
    print(f"Selected {len(libraries)} librar(y/ies)")
    print(f"Output directory: {args.output_dir}")

    summary_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str, int]] = []

    with make_libraries_dir(libraries) as libraries_tmp:
        libraries_dir = Path(libraries_tmp)
        for binary in elfs:
            for mode in modes:
                stem = f"{safe_name(binary)}.{mode}"
                raw_log = raw_dir / f"{stem}.raw.log"
                report = reports_dir / f"{stem}.report.txt"

                if args.resume and raw_log.exists() and report.exists():
                    print(f"[resume] {binary.name} {mode}")
                else:
                    command = main_command(args, binary, libraries_dir, report, mode)
                    returncode = run_command(
                        command,
                        cwd=SCRIPT_DIR,
                        log_path=raw_log,
                        timeout=args.timeout,
                        dry_run=args.dry_run,
                    )
                    if returncode != 0:
                        failures.append((binary.name, mode, returncode))

                summary_rows.extend(parse_report(report, binary, mode, raw_log))

    summary_csv = args.output_dir / "summary.csv"
    comparison_csv = args.output_dir / "comparison.csv"
    write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    write_csv(comparison_csv, COMPARISON_FIELDS, comparison_rows(summary_rows))

    print(f"Wrote {summary_csv}")
    print(f"Wrote {comparison_csv}")

    if failures:
        print("Failures:")
        for binary_name, mode, returncode in failures:
            print(f"  {binary_name} {mode}: return code {returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_batch(parse_args()))
