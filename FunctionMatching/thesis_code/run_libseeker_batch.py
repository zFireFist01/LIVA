#!/usr/bin/env python3
"""Compare the current block pipeline with libseeker's function matching."""

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
CURRENT_PIPELINE_DIR = SCRIPT_DIR
LIBSEEKER_PIPELINE_DIR = REPO_ROOT / "FunctionMatching" / "libseeker" / "thesis_code"
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
    "pipeline",
    "binary",
    "library",
    "status",
    "score",
    "block_best",
    "rodata_best",
    "rodata_confirmed_cu",
    "rodata_penalty_cu",
    "candidate_cu",
    "matched_cu",
    "total_cu",
    "matched_functions",
    "time",
    "total_time",
    "log_path",
]

COMPARISON_FIELDS = [
    "binary",
    "library",
    "outcome",
    "current_status",
    "libseeker_status",
    "current_score",
    "libseeker_score",
    "current_block_best",
    "current_rodata_best",
    "current_matched_cu",
    "libseeker_matched_cu",
    "current_total_cu",
    "libseeker_total_cu",
    "current_matched_functions",
    "libseeker_matched_functions",
    "current_time",
    "libseeker_time",
    "current_total_time",
    "libseeker_total_time",
    "current_log_path",
    "libseeker_log_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test static libraries with the current block pipeline, "
            "libseeker's function-matching pipeline, or both."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--libs-dir", type=Path, default=DEFAULT_LIBS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pipeline",
        "--mode",
        choices=("current", "libseeker", "both"),
        default="current",
        help=(
            "Pipeline to run. 'current' is the block/.rodata pipeline, "
            "'libseeker' is the function-matching baseline, and 'both' compares them."
        ),
    )
    parser.add_argument(
        "--libseeker-pipeline-dir",
        type=Path,
        default=LIBSEEKER_PIPELINE_DIR,
        help="Directory containing the libseeker baseline main.py.",
    )
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
        help="Python executable used for both pipelines. Defaults to the current interpreter.",
    )
    parser.add_argument("--min-cu", type=int, default=1)
    parser.add_argument("--min-functions-cu", type=int, default=1)
    parser.add_argument(
        "--libseeker-threshold",
        type=float,
        default=0.80,
        help="CU similarity threshold used by the libseeker function-matching baseline.",
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
    parser.add_argument("--disable-rodata-filter", action="store_true")
    parser.add_argument("--rodata-min-bytes", type=int, default=128)
    parser.add_argument("--rodata-min-strings", type=int, default=2)
    parser.add_argument("--rodata-min-ngrams", type=int, default=32)
    parser.add_argument("--rodata-penalty-threshold", type=float, default=0.10)
    parser.add_argument("--rodata-confirm-threshold", type=float, default=0.70)
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


def current_command(
    args: argparse.Namespace,
    binary: Path,
    libraries_dir: Path,
    output_path: Path,
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
        "--min_cu",
        str(args.min_cu),
        "--min_functions_cu",
        str(args.min_functions_cu),
    ]

    command.extend(
        [
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
            "--rodata_min_bytes",
            str(args.rodata_min_bytes),
            "--rodata_min_strings",
            str(args.rodata_min_strings),
            "--rodata_min_ngrams",
            str(args.rodata_min_ngrams),
            "--rodata_penalty_threshold",
            str(args.rodata_penalty_threshold),
            "--rodata_confirm_threshold",
            str(args.rodata_confirm_threshold),
        ]
    )
    if args.disable_rodata_filter:
        command.append("--disable_rodata_filter")
    if args.block_coverage_mean_threshold is not None:
        command.extend(
            [
                "--block_coverage_mean_threshold",
                str(args.block_coverage_mean_threshold),
            ]
        )

    return command


def libseeker_command(
    args: argparse.Namespace,
    binary: Path,
    libraries_dir: Path,
    output_path: Path,
) -> list[str]:
    return [
        args.python,
        "main.py",
        "--hide-warnings",
        "--path_to_binary",
        binary.as_posix(),
        "--libraries_dir",
        libraries_dir.as_posix(),
        "--output",
        output_path.as_posix(),
        "--threshold",
        str(args.libseeker_threshold),
        "--min_cu",
        str(args.min_cu),
        "--min_functions_cu",
        str(args.min_functions_cu),
    ]


def selected_pipelines(args: argparse.Namespace) -> list[tuple[str, Path]]:
    pipelines = {
        "current": CURRENT_PIPELINE_DIR,
        "libseeker": args.libseeker_pipeline_dir.resolve(),
    }
    if args.pipeline == "both":
        return list(pipelines.items())
    return [(args.pipeline, pipelines[args.pipeline])]


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


def parse_total_time(log_path: Path) -> str:
    if not log_path.exists():
        return ""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    completed = re.findall(r"Done processing in\s+(\d{2}:\d{2}:\d{2})", text)
    if completed:
        return completed[-1]

    timed_out = re.findall(r"\[TIMEOUT\] after\s+(\d{2}:\d{2}:\d{2})", text)
    return timed_out[-1] if timed_out else ""


def parse_report(
    pipeline: str,
    report_path: Path,
    binary: Path,
    log_path: Path,
) -> list[dict[str, str]]:
    if not report_path.exists():
        return []

    rows = []
    total_time = parse_total_time(log_path)
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
            "pipeline": pipeline,
            "binary": binary.name,
            "library": values.get("library", ""),
            "status": match.group("status").strip(),
            "score": parse_percent(values.get("score")),
            "block_best": parse_percent(values.get("block_best")),
            "rodata_best": parse_percent(values.get("rodata_best")),
            "rodata_confirmed_cu": values.get("rodata_confirmed_cu", ""),
            "rodata_penalty_cu": values.get("rodata_penalty_cu", ""),
            "candidate_cu": values.get("candidate_cu", ""),
            "matched_cu": matched_cu,
            "total_cu": total_cu,
            "matched_functions": values.get("matched_functions", ""),
            "time": values.get("time", ""),
            "total_time": total_time,
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


def detection_outcome(
    current_row: dict[str, str] | None,
    libseeker_row: dict[str, str] | None,
) -> str:
    if current_row is None:
        return "missing_current"
    if libseeker_row is None:
        return "missing_libseeker"

    current_detected = current_row["status"].startswith("YES")
    libseeker_detected = libseeker_row["status"].startswith("YES")
    if current_detected and libseeker_detected:
        return "both_detected"
    if current_detected:
        return "current_only"
    if libseeker_detected:
        return "libseeker_only"
    return "neither_detected"


def build_comparison_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed_rows = {
        (row["pipeline"], row["binary"], row["library"]): row
        for row in summary_rows
    }
    cases = sorted(
        (row["binary"], row["library"])
        for row in summary_rows
        if row["pipeline"] == "current"
    )
    cases = sorted(
        set(cases)
        | {
            (row["binary"], row["library"])
            for row in summary_rows
            if row["pipeline"] == "libseeker"
        }
    )

    comparison_rows = []
    for binary, library in cases:
        current_row = indexed_rows.get(("current", binary, library))
        libseeker_row = indexed_rows.get(("libseeker", binary, library))
        comparison_rows.append(
            {
                "binary": binary,
                "library": library,
                "outcome": detection_outcome(current_row, libseeker_row),
                "current_status": current_row["status"] if current_row else "",
                "libseeker_status": libseeker_row["status"] if libseeker_row else "",
                "current_score": current_row["score"] if current_row else "",
                "libseeker_score": libseeker_row["score"] if libseeker_row else "",
                "current_block_best": current_row["block_best"] if current_row else "",
                "current_rodata_best": current_row["rodata_best"] if current_row else "",
                "current_matched_cu": current_row["matched_cu"] if current_row else "",
                "libseeker_matched_cu": libseeker_row["matched_cu"] if libseeker_row else "",
                "current_total_cu": current_row["total_cu"] if current_row else "",
                "libseeker_total_cu": libseeker_row["total_cu"] if libseeker_row else "",
                "current_matched_functions": (
                    current_row["matched_functions"] if current_row else ""
                ),
                "libseeker_matched_functions": (
                    libseeker_row["matched_functions"] if libseeker_row else ""
                ),
                "current_time": current_row["time"] if current_row else "",
                "libseeker_time": libseeker_row["time"] if libseeker_row else "",
                "current_total_time": current_row["total_time"] if current_row else "",
                "libseeker_total_time": libseeker_row["total_time"] if libseeker_row else "",
                "current_log_path": current_row["log_path"] if current_row else "",
                "libseeker_log_path": libseeker_row["log_path"] if libseeker_row else "",
            }
        )

    return comparison_rows


def run_batch(args: argparse.Namespace) -> int:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if not args.libs_dir.is_dir():
        raise FileNotFoundError(f"Libraries directory not found: {args.libs_dir}")

    pipelines = selected_pipelines(args)
    for pipeline_name, pipeline_dir in pipelines:
        if not (pipeline_dir / "main.py").is_file():
            raise FileNotFoundError(
                f"{pipeline_name} pipeline main.py not found in: {pipeline_dir}"
            )

    elfs = select_elfs(args)
    libraries = select_libs(args)
    if not elfs:
        raise ValueError("No ELF files selected")
    if not libraries:
        raise ValueError("No libraries selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(elfs)} ELF file(s)")
    print(f"Selected {len(libraries)} librar(y/ies)")
    print(f"Pipelines: {', '.join(name for name, _ in pipelines)}")
    print(f"Output directory: {args.output_dir}")

    summary_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str, int]] = []

    with make_libraries_dir(libraries) as libraries_tmp:
        libraries_dir = Path(libraries_tmp)
        for pipeline_name, pipeline_dir in pipelines:
            raw_dir = args.output_dir / "raw" / pipeline_name
            reports_dir = args.output_dir / "reports" / pipeline_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            for binary in elfs:
                stem = safe_name(binary)
                raw_log = raw_dir / f"{stem}.raw.log"
                report = reports_dir / f"{stem}.report.txt"

                if args.resume and raw_log.exists() and report.exists():
                    print(f"[resume:{pipeline_name}] {binary.name}")
                else:
                    if pipeline_name == "current":
                        command = current_command(args, binary, libraries_dir, report)
                    else:
                        command = libseeker_command(args, binary, libraries_dir, report)

                    returncode = run_command(
                        command,
                        cwd=pipeline_dir,
                        log_path=raw_log,
                        timeout=args.timeout,
                        dry_run=args.dry_run,
                    )
                    if returncode != 0:
                        failures.append((pipeline_name, binary.name, returncode))

                summary_rows.extend(
                    parse_report(pipeline_name, report, binary, raw_log)
                )

    summary_csv = args.output_dir / "summary.csv"
    write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)

    print(f"Wrote {summary_csv}")
    if args.pipeline == "both":
        comparison_csv = args.output_dir / "comparison.csv"
        write_csv(
            comparison_csv,
            COMPARISON_FIELDS,
            build_comparison_rows(summary_rows),
        )
        print(f"Wrote {comparison_csv}")

    if failures:
        print("Failures:")
        for pipeline_name, binary_name, returncode in failures:
            print(f"  {pipeline_name} {binary_name}: return code {returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_batch(parse_args()))
