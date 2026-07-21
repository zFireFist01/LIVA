#!/usr/bin/env python3
"""Hybrid parameter search for the randomized static-ELF dataset.

Structural block-matching parameters are sampled through expensive pipeline
runs. Decision thresholds are then optimized offline with restarted greedy
coordinate search over the emitted BLOCK_CU records. By default, programs are
kept in an untouched test split plus a development set: grouped CV selects the
profile, final thresholds are retuned on all development cases, and the test is
evaluated exactly once. A legacy train/validation holdout remains available.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any

from match import LIBRARY_SCORE_AGGREGATORS, aggregate_library_score


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR / "Test"
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "GroundTruth"
DEFAULT_LIB_BUILD_ROOT = REPO_ROOT / "Dataset/builds/lib_builds"
DEFAULT_GCC_LIB_ROOT = DEFAULT_LIB_BUILD_ROOT / "gcc-16.1.1"
DEFAULT_NEGATIVE_LIB_ROOT = (
    REPO_ROOT / "Exploration/libseeker_repo/build_lib/all_libs"
)
DEFAULT_OUTPUT = TEST_DIR / "greedy_threshold_results"
OPTIMIZATIONS = ("O0", "O1", "O2", "O3", "Os")
DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES = 600_000
FEATURE_SCHEMA_VERSION = 6
DEFAULT_PROGRAMS = ("grep", "less", "sed", "gawk", "nano")
DATASET_PROGRAMS = (
    "bash",
    "gawk",
    "gnuchess",
    "grep",
    "gzip",
    "inetutils",
    "less",
    "make",
    "nano",
    "openssh",
    "rsync",
    "sed",
    "socat",
    "tar",
    "wget2",
)

PREFERRED_NEGATIVE_ARCHIVES = (
    "libpcre2-posix.a",
    "libpanelw.a",
    "libmenuw.a",
    "libformw.a",
    "libbz2.a",
    "libcap.a",
    "libattr.a",
    "libuuid.a",
    "libexpat.a",
    "liblzma.a",
    "libssl.a",
    "libcrypto.a",
    "libmount.a",
    "libblkid.a",
    "libsodium.a",
)

DECISION_DEFAULTS = {
    "library_min_score": 0.0,
    "block_coverage_mean_threshold": 0.80,
    "block_assignment_quality_threshold": 0.875,
    "block_min_assignment_ratio": 0.50,
    "block_min_coverage_ratio": 0.50,
    "block_min_call_edge_ratio": 0.50,
    "block_min_function_concentration": 0.45,
    "block_min_function_spread": 0.30,
    "rodata_filter_enabled": 1,
    "rodata_min_bytes": 512,
    "rodata_min_strings": 0,
    "rodata_min_ngrams": 32,
    # Offline grouped-CV selection for the ternary .rodata policy.  The low
    # branch remains implemented, while 0.0 avoids rejecting nonzero evidence.
    "rodata_penalty_threshold": 0.0,
    "rodata_confirm_threshold": 0.70,
    "rodata_bonus_weight": 0.30,
}

DECISION_GRIDS = {
    "library_min_score": (
        0.0,
        *tuple(round(0.60 + (index * 0.01), 3) for index in range(40)),
    ),
    "block_coverage_mean_threshold": tuple(
        round(0.60 + (index * 0.025), 3) for index in range(13)
    ),
    "block_assignment_quality_threshold": tuple(
        round(0.60 + (index * 0.025), 3) for index in range(13)
    ),
    "block_min_assignment_ratio": tuple(
        round(index * 0.05, 3) for index in range(21)
    ),
    "block_min_coverage_ratio": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "block_min_call_edge_ratio": tuple(
        round(index * 0.10, 3) for index in range(10)
    ),
    "block_min_function_concentration": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "block_min_function_spread": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "rodata_min_bytes": (0, 32, 64, 128, 256, 512),
    "rodata_min_strings": (0, 1, 2, 3, 5),
    "rodata_min_ngrams": (0, 16, 32, 64, 128),
    "rodata_penalty_threshold": (0.00, 0.05, 0.10, 0.15, 0.20, 0.30),
}

STRUCTURAL_DEFAULTS = {
    "block_threshold": 0.70,
    # Keep the search baseline generic. The production pipeline may use a
    # tuned wider window, but the greedy search should not be centered on it.
    "block_locality_window_multiplier": 3.0,
    "block_locality_window_padding": 2,
}

STRUCTURAL_GRIDS = {
    "block_threshold": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    "block_locality_window_multiplier": (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    "block_locality_window_padding": (0, 1, 2, 3, 4, 6),
}

SUMMARY_RE = re.compile(
    r"^(?:YES \[W\]|YES|NO)\s+\|\s+library=(?P<library>\S+)"
)
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect block-matching features from the randomized ELF builds, "
            "then tune the decision thresholds with greedy coordinate search."
        )
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--lib-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Compiler-specific library root. Repeatable. Defaults to "
            "gcc-16.1.1 plus any clang-* roots found under Dataset/builds/lib_builds."
        ),
    )
    parser.add_argument(
        "--negative-libs-dir",
        type=Path,
        default=DEFAULT_NEGATIVE_LIB_ROOT,
        help="libseeker_repo directory containing versioned negative archives.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260615,
        help="Controls the positive optimization mix and negative versions.",
    )
    parser.add_argument(
        "--program",
        action="append",
        choices=DATASET_PROGRAMS,
        help="Restrict the dataset to selected programs. Repeatable.",
    )
    parser.add_argument(
        "--validation-program",
        action="append",
        choices=DATASET_PROGRAMS,
        help=(
            "Assign this whole program to validation. Repeatable. When omitted, "
            "one program is selected deterministically from --seed."
        ),
    )
    parser.add_argument(
        "--test-program",
        action="append",
        choices=DATASET_PROGRAMS,
        help=(
            "Assign this whole program to test. Repeatable. When omitted, one "
            "program is selected deterministically from --seed."
        ),
    )
    parser.add_argument(
        "--test-program-count",
        type=int,
        default=2,
        help=(
            "Number of whole programs assigned automatically to the untouched "
            "test split. Ignored when --test-program is supplied. Defaults to "
            "2 so final metrics do not depend on a single program."
        ),
    )
    parser.add_argument(
        "--validation-mode",
        choices=("grouped_cv", "holdout"),
        default="grouped_cv",
        help=(
            "grouped_cv selects thresholds/profile through program-grouped "
            "cross-validation over all non-test development cases, then "
            "retunes the selected profile on the full development set. "
            "holdout preserves the legacy train/validation procedure."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of program-grouped development folds (default: 3).",
    )
    parser.add_argument(
        "--elf",
        "--variant",
        dest="elf_variants",
        action="append",
        default=[],
        help=(
            "Use this exact ELF variant, for example grep_clang-22.1.6_O0. "
            "Repeatable. With the default two-program test, exact variants "
            "must span at least four programs."
        ),
    )
    parser.add_argument(
        "--compiler",
        action="append",
        choices=("gcc", "clang"),
        help="Restrict the search to selected ELF compilers. Repeatable.",
    )
    parser.add_argument(
        "--include-glibc",
        action="store_true",
        help="Include libc.a. It is universal in this dataset and very expensive.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device forwarded to PalmTree: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--asm-normalization",
        choices=("legacy", "v2"),
        default="v2",
        help=(
            "Assembly preprocessing version used during feature collection. "
            "v2 is stripped-safe; changing it invalidates feature caches."
        ),
    )
    parser.add_argument(
        "--palmtree-pooling",
        choices=("mean", "masked_mean"),
        default="masked_mean",
        help=(
            "PalmTree instruction pooling used during feature collection. "
            "Changing it invalidates feature caches."
        ),
    )
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--structural-trials",
        type=int,
        default=19,
        help="Number of expensive structural matching profiles to evaluate.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Greedy coordinate-search restarts for each structural profile.",
    )
    parser.add_argument(
        "--train-elf-count",
        type=int,
        default=6,
        help=(
            "Maximum number of training ELF files used to tune thresholds. "
            "Defaults to 6 to "
            "keep the expensive matching stage manageable."
        ),
    )
    parser.add_argument(
        "--validation-elf-count",
        type=int,
        default=None,
        help=(
            "Optional maximum number of validation ELF variants sampled from "
            "the validation programs. By default all their variants are used."
        ),
    )
    parser.add_argument(
        "--test-elf-count",
        type=int,
        default=None,
        help=(
            "Optional maximum number of test ELF variants sampled from the test "
            "programs. By default all their variants are used."
        ),
    )
    parser.add_argument(
        "--min-test-positive-labels",
        type=int,
        default=2,
        help=(
            "Reject a final test panel with fewer positive ELF/library labels. "
            "Defaults to 2; use 1 only for deliberately small smoke tests."
        ),
    )
    parser.add_argument(
        "--positive-lib-count",
        type=int,
        default=6,
        help=(
            "Number of present libraries used by the search. The same number "
            "of absent libraries is selected from libseeker_repo."
        ),
    )
    parser.add_argument(
        "--max-positive-archive-bytes",
        type=int,
        default=DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES,
        help=(
            "Prefer positive archives no larger than this many bytes, using "
            "the smallest available compiler/optimization build. 0 disables "
            "the size filter. Defaults to 600000 to avoid very slow libraries "
            "such as libiconv.a during threshold tuning."
        ),
    )
    parser.add_argument(
        "--library-score-aggregator",
        choices=LIBRARY_SCORE_AGGREGATORS,
        default="mean",
        help=(
            "Fixed CU-to-library score aggregator for this search. Feature "
            "collections are aggregator-independent and can be reused."
        ),
    )
    parser.add_argument(
        "--objective-order",
        choices=("cu_first", "lib_first", "lib_only"),
        default="cu_first",
        help=(
            "Metric priority for greedy selection. cu_first optimizes CU "
            "F1/precision before library metrics; lib_first optimizes "
            "library F1/precision before CU metrics; lib_only never uses CU "
            "labels to break ties."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete feature collections already present in output-dir.",
    )
    parser.add_argument(
        "--reuse-feature-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Reuse compatible per-library features from a previous "
            "greedy_threshold_search output directory. Repeatable. This lets "
            "a new library selection reuse already collected profile/ELF/"
            "library records and compute only missing pairs."
        ),
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Do not run matching; use previously collected feature files.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect feature files without running the greedy search.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def archive_family(name: str) -> str:
    lowered = Path(name).name.lower()
    special = {
        "libc.a": "glibc",
        "libpcre2-8.a": "pcre2",
        "libiconv.a": "iconv",
    }
    return special.get(lowered, lowered.removesuffix(".a"))


def normalize_cu_name(name: str) -> str:
    normalized = Path(name).name
    if normalized.endswith(".o"):
        normalized = normalized[:-2]
    if normalized.endswith(".c"):
        normalized = normalized[:-2]
    prefixes = (
        "libpcre2_8_la-",
        "libpcre2_posix_la-",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


LINKER_MAP_MEMBER_RE = re.compile(
    r"(?P<archive>\S+?\.a)\((?P<member>[^()]+\.o)\)"
)


def included_cus_from_linker_map(linker_map: Path) -> dict[str, set[str]]:
    included: dict[str, set[str]] = {}
    if not linker_map.is_file():
        return included

    for match in LINKER_MAP_MEMBER_RE.finditer(
        linker_map.read_text(encoding="utf-8", errors="replace")
    ):
        family = archive_family(Path(match.group("archive")).name)
        included.setdefault(family, set()).add(
            normalize_cu_name(match.group("member"))
        )
    return included


def included_cus_from_ground_truth(data: dict[str, Any]) -> dict[str, set[str]]:
    """Return only archive members positively confirmed by a linker map."""
    included: dict[str, set[str]] = {}
    has_embedded_cu_ground_truth = False
    for archive in data.get("archives", []):
        family = archive_family(Path(str(archive["archive"])).name)
        confirmed_members = archive.get("confirmed_compilation_units")
        if confirmed_members is not None:
            has_embedded_cu_ground_truth = True
            if archive.get("confirmed_compilation_units_method") != "linker_map":
                continue
            names = {
                normalize_cu_name(str(member))
                for member in confirmed_members
            }
            if names:
                included.setdefault(family, set()).update(names)
            continue

        compilation_units = archive.get("compilation_units")
        if compilation_units is None:
            continue
        has_embedded_cu_ground_truth = True
        names = {
            normalize_cu_name(str(unit["compilation_unit"]))
            for unit in compilation_units
            if unit.get("included")
            and unit.get("ground_truth_method") == "linker_map"
        }
        if names:
            included.setdefault(family, set()).update(names)

    linker_map = data.get("linker_map")
    if linker_map and not has_embedded_cu_ground_truth:
        for family, names in included_cus_from_linker_map(Path(str(linker_map))).items():
            included.setdefault(family, set()).update(names)

    return included


def archive_is_present_in_ground_truth(archive: dict[str, Any]) -> bool:
    if "included_compilation_units" not in archive:
        return True
    return int(archive.get("included_compilation_units", 0) or 0) > 0


def resolve_lib_roots(args: argparse.Namespace) -> list[Path]:
    if args.lib_root:
        roots = [path.resolve() for path in args.lib_root]
    else:
        roots = []
        if DEFAULT_GCC_LIB_ROOT.is_dir():
            roots.append(DEFAULT_GCC_LIB_ROOT.resolve())
        roots.extend(
            path.resolve()
            for path in sorted(DEFAULT_LIB_BUILD_ROOT.glob("clang-*"))
            if path.is_dir()
        )

    if not roots:
        raise FileNotFoundError("No compiler-specific library roots found")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Library root not found: {root}")
    return roots


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.elf_variants:
        programs = set(args.program or DATASET_PROGRAMS)
        requested_variants = set(args.elf_variants)
    else:
        programs = set(args.program or DEFAULT_PROGRAMS)
        requested_variants = set()
    programs.update(args.validation_program or [])
    programs.update(args.test_program or [])
    compilers = set(args.compiler or ("gcc", "clang"))
    cases = []

    for program in sorted(programs):
        for path in sorted((args.ground_truth_dir / program).glob("*/ground_truth.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("elf_optimization"):
                continue
            if requested_variants and data.get("variant") not in requested_variants:
                continue
            compiler = str(data.get("compiler", "")).split("-", maxsplit=1)[0]
            if compiler not in compilers:
                continue
            binary = Path(str(data["binary"])).resolve()
            if not binary.is_file():
                raise FileNotFoundError(f"Ground-truth ELF not found: {binary}")
            present_archives = [
                archive
                for archive in data.get("archives", [])
                if archive_is_present_in_ground_truth(archive)
            ]
            expected = {
                archive_family(Path(str(archive["archive"])).name)
                for archive in present_archives
            }
            archive_sources: dict[str, set[str]] = {}
            for archive in present_archives:
                archive_path = Path(str(archive["archive"]))
                archive_sources.setdefault(archive_path.name, set()).add(
                    archive_path.parents[3].name
                )
            expected_cus = included_cus_from_ground_truth(data)
            cases.append(
                {
                    "variant": str(data["variant"]),
                    "program": program,
                    "compiler": compiler,
                    "elf_optimization": str(data["elf_optimization"]),
                    "binary": binary,
                    "expected": expected,
                    "expected_cus": expected_cus,
                    "archive_names": {
                        Path(str(archive["archive"])).name
                        for archive in present_archives
                    },
                    "archive_sources": archive_sources,
                }
            )

    if not cases:
        raise ValueError("No randomized ELF ground truths selected")
    if requested_variants:
        found = {case["variant"] for case in cases}
        missing = requested_variants - found
        if missing:
            raise ValueError(
                "Requested ELF variants not found after filters: "
                + ", ".join(sorted(missing))
            )
    return cases


def candidate_archive_names(
    cases: list[dict[str, Any]], include_glibc: bool
) -> list[str]:
    names = {
        archive_name
        for case in cases
        for archive_name in case["archive_names"]
    }
    if not include_glibc:
        names.discard("libc.a")
    return sorted(names)


def find_candidate_archive(
    lib_root: Path,
    source: str,
    optimization: str,
    archive_name: str,
) -> Path:
    matches = sorted(
        path.resolve()
        for path in (lib_root / source / optimization / "install/lib").glob(
            archive_name
        )
        if path.is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {archive_name} in {lib_root.name}/{source}/"
            f"{optimization}, found {len(matches)}"
        )
    return matches[0]


def preferred_sources_for_archive(
    cases: list[dict[str, Any]], archive_name: str
) -> set[str]:
    preferred: set[str] = set()
    for case in cases:
        preferred.update(case.get("archive_sources", {}).get(archive_name, set()))
    return preferred


def source_for_archive(
    lib_roots: list[Path],
    archive_name: str,
    preferred_sources: set[str] | None = None,
) -> str:
    matches = sorted(
        path.resolve()
        for lib_root in lib_roots
        for optimization in OPTIMIZATIONS
        for path in lib_root.glob(
            f"*/{optimization}/install/lib/{archive_name}"
        )
        if path.is_file()
    )
    sources = {path.parents[3].name for path in matches}
    if preferred_sources:
        preferred_matches = sources & preferred_sources
        if len(preferred_matches) == 1:
            return preferred_matches.pop()
    if len(sources) != 1:
        raise FileNotFoundError(
            f"Expected one source library for {archive_name}, found "
            f"{sorted(sources)}; preferred={sorted(preferred_sources or [])}"
        )
    return sources.pop()


def available_optimizations(
    lib_root: Path, source: str, archive_names: list[str]
) -> list[str]:
    return [
        optimization
        for optimization in OPTIMIZATIONS
        if all(
            (
                lib_root
                / source
                / optimization
                / "install/lib"
                / archive_name
            ).is_file()
            for archive_name in archive_names
        )
    ]


def available_optimizations_with_size_limit(
    args: argparse.Namespace,
    lib_root: Path,
    source: str,
    archive_names: list[str],
) -> list[str]:
    optimizations = available_optimizations(lib_root, source, archive_names)
    if args.max_positive_archive_bytes <= 0:
        return optimizations
    return [
        optimization
        for optimization in optimizations
        if all(
            archive_size(lib_root, source, optimization, archive_name)
            <= args.max_positive_archive_bytes
            for archive_name in archive_names
        )
    ]


def archive_size(lib_root: Path, source: str, optimization: str, archive_name: str) -> int:
    return (
        lib_root
        / source
        / optimization
        / "install/lib"
        / archive_name
    ).stat().st_size


def smallest_archive_size(
    lib_roots: list[Path], source: str, archive_name: str
) -> int:
    sizes = [
        archive_size(lib_root, source, optimization, archive_name)
        for lib_root in lib_roots
        for optimization in OPTIMIZATIONS
        if (
            lib_root
            / source
            / optimization
            / "install/lib"
            / archive_name
        ).is_file()
    ]
    if not sizes:
        raise FileNotFoundError(f"No build found for {archive_name}")
    return min(sizes)


def select_positive_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, str]]:
    if args.positive_lib_count < 1:
        raise ValueError("--positive-lib-count must be at least 1")

    archive_names = candidate_archive_names(cases, args.include_glibc)
    preferred_sources_by_archive = {
        archive_name: preferred_sources_for_archive(cases, archive_name)
        for archive_name in archive_names
    }
    if args.max_positive_archive_bytes > 0:
        small_archive_names = []
        skipped_archive_names = []
        for archive_name in archive_names:
            source = source_for_archive(
                args.lib_roots,
                archive_name,
                preferred_sources_by_archive.get(archive_name),
            )
            size = smallest_archive_size(args.lib_roots, source, archive_name)
            if size <= args.max_positive_archive_bytes:
                small_archive_names.append(archive_name)
            else:
                skipped_archive_names.append((archive_name, size))

        if len(small_archive_names) >= args.positive_lib_count:
            archive_names = small_archive_names
        else:
            print(
                f"[WARN] Only {len(small_archive_names)} positive libraries "
                f"fit --max-positive-archive-bytes="
                f"{args.max_positive_archive_bytes}; using them anyway"
            )
            archive_names = small_archive_names

        for archive_name, size in skipped_archive_names:
            print(
                f"[positive-skip] {archive_name} size={size} exceeds "
                f"--max-positive-archive-bytes={args.max_positive_archive_bytes}"
            )

    archives_by_source: dict[str, list[str]] = {}
    for archive_name in archive_names:
        source = source_for_archive(
            args.lib_roots,
            archive_name,
            preferred_sources_by_archive.get(archive_name),
        )
        archives_by_source.setdefault(source, []).append(archive_name)

    source_candidates = []
    for source, source_archive_names in sorted(archives_by_source.items()):
        archive_name = min(
            source_archive_names,
            key=lambda name: (
                smallest_archive_size(args.lib_roots, source, name),
                name,
            ),
        )
        source_candidates.append(
            (
                smallest_archive_size(args.lib_roots, source, archive_name),
                source,
                archive_name,
            )
        )
    source_candidates.sort()

    selected_names = []
    source_order = [source for _, source, _ in source_candidates]
    for _, _, archive_name in source_candidates:
        selected_names.append(archive_name)
        if len(selected_names) == args.positive_lib_count:
            break

    if len(selected_names) < args.positive_lib_count:
        already_selected = set(selected_names)
        remaining = [
            archive_name
            for source in source_order
            for archive_name in sorted(archives_by_source[source])
            if archive_name not in already_selected
        ]
        remaining.sort(
            key=lambda name: (
                smallest_archive_size(
                    args.lib_roots,
                    source_for_archive(
                        args.lib_roots,
                        name,
                        preferred_sources_by_archive.get(name),
                    ),
                    name,
                ),
                name,
            )
        )
        selected_names.extend(
            remaining[: args.positive_lib_count - len(selected_names)]
        )

    if len(selected_names) < args.positive_lib_count:
        print(
            f"[WARN] Requested {args.positive_lib_count} positive libraries, "
            f"but only {len(selected_names)} are available for the selected ELF set"
        )

    archives_by_source = {}
    for archive_name in selected_names:
        source = source_for_archive(
            args.lib_roots,
            archive_name,
            preferred_sources_by_archive.get(archive_name),
        )
        archives_by_source.setdefault(source, []).append(archive_name)

    sources = sorted(archives_by_source)
    optimization_cycle = list(OPTIMIZATIONS)
    rng.shuffle(optimization_cycle)
    root_cycle = list(args.lib_roots)
    rng.shuffle(root_cycle)
    selected = []

    for index, source in enumerate(sources):
        source_archives = sorted(archives_by_source[source])
        root_candidates = [
            lib_root
            for lib_root in args.lib_roots
            if available_optimizations_with_size_limit(
                args, lib_root, source, source_archives
            )
        ]
        if not root_candidates:
            raise FileNotFoundError(
                f"No compiler root has all archives for {source_archives} "
                f"within --max-positive-archive-bytes="
                f"{args.max_positive_archive_bytes}"
            )
        preferred_root = root_cycle[index % len(root_cycle)]
        lib_root = (
            preferred_root
            if preferred_root in root_candidates
            else min(root_candidates, key=lambda root: root.name)
        )
        available = available_optimizations_with_size_limit(
            args, lib_root, source, source_archives
        )
        if not available:
            raise FileNotFoundError(
                f"No common optimization available for {source_archives}"
            )
        preferred = optimization_cycle[index % len(optimization_cycle)]
        optimization = preferred if preferred in available else rng.choice(available)

        for archive_name in source_archives:
            archive = find_candidate_archive(
                lib_root, source, optimization, archive_name
            )
            selected.append(
                {
                    "name": archive_name,
                    "path": str(archive),
                    "kind": "positive",
                    "source": source,
                    "variant": f"{lib_root.name}/{optimization}",
                    "compiler_root": lib_root.name,
                    "optimization": optimization,
                    "size": archive.stat().st_size,
                }
            )
    return sorted(selected, key=lambda item: item["name"])


def versioned_archive_name(path: Path) -> str | None:
    match = re.match(r"^(?P<archive>.+\.a)\.(?P<version>.+)$", path.name)
    return match.group("archive") if match else None


def negative_archive_pool(directory: Path) -> dict[str, list[Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Negative libraries directory not found: {directory}")

    pool: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        archive_name = versioned_archive_name(path)
        if archive_name is None or path.stat().st_size < 4096:
            continue
        pool.setdefault(archive_name, []).append(path.resolve())
    return pool


def smallest_negative_size(paths: list[Path]) -> int:
    return min(path.stat().st_size for path in paths)


def select_negative_archives(
    args: argparse.Namespace,
    positive_archives: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    pool = negative_archive_pool(args.negative_libs_dir)
    positive_names = {item["name"] for item in positive_archives}
    count = len(positive_archives)
    eligible_names = [
        name
        for name in PREFERRED_NEGATIVE_ARCHIVES
        if name in pool and name not in positive_names
    ]
    eligible_names.sort(key=lambda name: (smallest_negative_size(pool[name]), name))

    if len(eligible_names) < count:
        fallback = sorted(
            name
            for name in pool
            if name not in positive_names
            and name not in eligible_names
            and not any(
                token in name
                for token in ("_pic.a", "_g.a", "nonshared", "BrokenLocale")
            )
        )
        fallback.sort(key=lambda name: (smallest_negative_size(pool[name]), name))
        eligible_names.extend(fallback[: count - len(eligible_names)])

    if len(eligible_names) < count:
        raise ValueError(
            f"Need {count} negative libraries, found {len(eligible_names)}"
        )

    selected = []
    for archive_name in eligible_names[:count]:
        candidates = sorted(pool[archive_name], key=lambda path: (path.stat().st_size, path.name))
        archive = candidates[0]
        version = archive.name.removeprefix(f"{archive_name}.")
        selected.append(
            {
                "name": archive_name,
                "path": str(archive),
                "kind": "negative",
                "source": "libseeker_repo",
                "variant": version,
                "compiler_root": "libseeker_repo",
                "optimization": "",
                "size": archive.stat().st_size,
            }
        )
    return sorted(selected, key=lambda item: item["name"])


def selection_signature(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def archive_key_matches(library_key: str, archive_name: str) -> bool:
    return library_key == archive_name or library_key.startswith(f"{archive_name}.")


def canonical_library_key(library_key: str, archive_names: list[str]) -> str:
    for archive_name in archive_names:
        if archive_key_matches(library_key, archive_name):
            return archive_name
    return library_key


def find_archive_records(
    library_records: dict[str, list[dict[str, Any]]],
    archive_name: str,
) -> tuple[bool, list[dict[str, Any]]]:
    if archive_name in library_records:
        return True, library_records[archive_name]

    merged_records = []
    found = False
    for library_key, records in library_records.items():
        if archive_key_matches(library_key, archive_name):
            found = True
            merged_records.extend(records)
    return found, merged_records


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def prepare_inputs(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, str]],
) -> tuple[Path, Path]:
    elf_dir = args.output_dir / "inputs/elfs"
    lib_dir = args.output_dir / "inputs/libs"
    elf_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    for stale in (*elf_dir.iterdir(), *lib_dir.iterdir()):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)

    for case in cases:
        replace_symlink(elf_dir / case["variant"], case["binary"])
    for archive in selected_archives:
        replace_symlink(
            lib_dir / archive["name"],
            Path(archive["path"]),
        )
    return elf_dir, lib_dir


def prepare_subset_inputs(
    args: argparse.Namespace,
    profile_id: str,
    archive_name: str,
    cases: list[dict[str, Any]],
    archive: dict[str, str],
) -> tuple[Path, Path]:
    subset_root = (
        args.output_dir
        / "inputs"
        / "missing"
        / profile_id
        / safe_path_part(archive_name)
    )
    elf_dir = subset_root / "elfs"
    lib_dir = subset_root / "libs"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    elf_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)

    for case in cases:
        replace_symlink(elf_dir / case["variant"], case["binary"])
    replace_symlink(lib_dir / archive["name"], Path(archive["path"]))
    return elf_dir, lib_dir


def balanced_sample(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("ELF counts cannot be negative")
    if count == 0:
        return []

    buckets = {
        key: sorted(value, key=lambda case: case["variant"])
        for key, value in groups.items()
    }
    rng = random.Random(seed)
    for key, bucket in buckets.items():
        local_rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        local_rng.shuffle(bucket)

    selected = []
    group_order = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        rng.shuffle(group_order)
        progressed = False
        for key in group_order:
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop())
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break

    if len(selected) < count:
        raise ValueError(f"Requested {count} ELF files, found only {len(selected)}")
    return sorted(selected, key=lambda case: case["variant"])


def select_program_splits(
    cases: list[dict[str, Any]],
    validation_programs: list[str] | None,
    test_programs: list[str] | None,
    seed: int,
    include_glibc: bool = False,
    test_program_count: int = 1,
) -> tuple[set[str], set[str], set[str]]:
    """Assign whole programs to train, validation and test deterministically.

    Automatic holdouts prefer programs whose library families also occur in
    train. This keeps both held-out splits useful for positive detection without
    allowing any ELF variant of the same program to cross split boundaries.
    """
    available_programs = {str(case["program"]) for case in cases}
    if len(available_programs) < 3:
        raise ValueError(
            "At least three programs are required for grouped "
            "train/validation/test splitting"
        )

    if test_program_count < 1:
        raise ValueError("--test-program-count must be at least 1")

    validation = set(validation_programs or [])
    test = set(test_programs or [])
    unknown = (validation | test) - available_programs
    if unknown:
        raise ValueError(
            "Validation/test programs not present after case filters: "
            + ", ".join(sorted(unknown))
        )
    overlap = validation & test
    if overlap:
        raise ValueError(
            "Programs cannot be in both validation and test: "
            + ", ".join(sorted(overlap))
        )

    unassigned = sorted(available_programs - validation - test)
    automatic_test_count = 0 if test else test_program_count
    required = int(not validation) + automatic_test_count + 1
    if len(unassigned) < required:
        raise ValueError(
            "The requested validation/test programs do not leave at least one "
            "training program"
        )

    rng = random.Random(f"{seed}:program-split")
    validation_options = [validation] if validation else [{name} for name in unassigned]
    split_candidates = []
    for validation_option in validation_options:
        test_pool = available_programs - validation_option - test
        test_options = (
            [test]
            if test
            else [
                set(names)
                for names in itertools.combinations(
                    sorted(test_pool),
                    test_program_count,
                )
            ]
        )
        for test_option in test_options:
            train_option = available_programs - validation_option - test_option
            if not train_option:
                continue
            split_candidates.append(
                (set(train_option), set(validation_option), set(test_option))
            )

    if not split_candidates:
        raise ValueError("Unable to construct disjoint train/validation/test splits")

    ignored_families = set() if include_glibc else {"glibc"}
    families_by_program: dict[str, set[str]] = {
        program: set().union(
            *(
                set(case.get("expected", set()))
                for case in cases
                if case["program"] == program
            )
        )
        - ignored_families
        for program in available_programs
    }

    def families(programs: set[str]) -> set[str]:
        return set().union(*(families_by_program[name] for name in programs))

    rng.shuffle(split_candidates)
    train, validation, test = max(
        split_candidates,
        key=lambda split: (
            min(
                len(families(split[0]) & families(split[1])),
                len(families(split[0]) & families(split[2])),
            ),
            len(families(split[0]) & families(split[1]))
            + len(families(split[0]) & families(split[2])),
        ),
    )

    if not train:
        raise ValueError("At least one training program is required")
    return train, validation, test


def split_library_label_counts(
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, int]:
    """Count positive and negative library/case labels in one split."""
    candidate_families = [archive_family(name) for name in archive_names]
    positive = sum(
        family in case["expected"]
        for case in cases
        for family in candidate_families
    )
    total = len(cases) * len(candidate_families)
    return {"positive": positive, "negative": total - positive, "total": total}


def select_split_cases(
    cases: list[dict[str, Any]],
    count: int | None,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Return a balanced deterministic subset, treating count as an upper bound."""
    if not cases:
        raise ValueError(f"No cases available for the {split_name} split")
    if count is None:
        return sorted(cases, key=lambda case: case["variant"])
    if count < 1:
        raise ValueError(f"--{split_name}-elf-count must be at least 1")
    if count >= len(cases):
        return sorted(cases, key=lambda case: case["variant"])

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((case["program"], case["compiler"]), []).append(case)

    return balanced_sample(groups, count, seed)


def build_grouped_cv_folds(
    cases: list[dict[str, Any]],
    archive_names: list[str],
    fold_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic, balanced folds without splitting a program.

    Every development case appears in validation exactly once. Candidate
    assignments are scored by positive-label and case-count balance; folds
    without a positive or a negative label are rejected because their F1 is
    not useful for model selection.
    """
    programs = sorted({str(case["program"]) for case in cases})
    if fold_count < 2:
        raise ValueError("--cv-folds must be at least 2")
    if len(programs) < 2:
        raise ValueError("Grouped cross-validation requires at least two programs")
    fold_count = min(fold_count, len(programs))
    cases_by_program = {
        program: [case for case in cases if case["program"] == program]
        for program in programs
    }
    candidate_families = {archive_family(name) for name in archive_names}

    def positive_families(selected: list[dict[str, Any]]) -> set[str]:
        return set().union(
            *(set(case.get("expected", set())) & candidate_families for case in selected)
        )

    best_buckets: list[list[str]] | None = None
    best_key: tuple[Any, ...] | None = None
    rng = random.Random(f"{seed}:grouped-cv")
    attempts = max(256, len(programs) * 64)
    for _ in range(attempts):
        order = list(programs)
        rng.shuffle(order)
        buckets: list[list[str]] = [[] for _ in range(fold_count)]
        for index, program in enumerate(order):
            if index < fold_count:
                bucket_index = index
            else:
                sizes = [
                    sum(len(cases_by_program[name]) for name in bucket)
                    for bucket in buckets
                ]
                minimum = min(sizes)
                options = [i for i, size in enumerate(sizes) if size == minimum]
                bucket_index = rng.choice(options)
            buckets[bucket_index].append(program)

        validation_sets = [
            [case for name in bucket for case in cases_by_program[name]]
            for bucket in buckets
        ]
        training_sets = [
            [case for case in cases if case["program"] not in set(bucket)]
            for bucket in buckets
        ]
        validation_counts = [
            split_library_label_counts(selected, archive_names)
            for selected in validation_sets
        ]
        training_counts = [
            split_library_label_counts(selected, archive_names)
            for selected in training_sets
        ]
        usable = all(
            counts["positive"] > 0 and counts["negative"] > 0
            for counts in (*validation_counts, *training_counts)
        )
        positive_counts = [counts["positive"] for counts in validation_counts]
        case_counts = [len(selected) for selected in validation_sets]
        family_counts = [len(positive_families(selected)) for selected in validation_sets]
        key = (
            usable,
            min(positive_counts),
            min(family_counts),
            -(max(positive_counts) - min(positive_counts)),
            -(max(case_counts) - min(case_counts)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_buckets = [sorted(bucket) for bucket in buckets]

    if best_buckets is None or not bool(best_key and best_key[0]):
        raise ValueError(
            "Unable to build grouped CV folds containing both positive and "
            "negative labels; add development programs or expand the library panel"
        )

    folds = []
    for index, validation_programs in enumerate(best_buckets):
        validation_program_set = set(validation_programs)
        validation_cases = sorted(
            [case for case in cases if case["program"] in validation_program_set],
            key=lambda case: case["variant"],
        )
        train_cases = sorted(
            [case for case in cases if case["program"] not in validation_program_set],
            key=lambda case: case["variant"],
        )
        folds.append(
            {
                "fold_id": f"fold_{index:02d}",
                "train_programs": sorted({case["program"] for case in train_cases}),
                "validation_programs": validation_programs,
                "train_cases": train_cases,
                "validation_cases": validation_cases,
                "train_labels": split_library_label_counts(train_cases, archive_names),
                "validation_labels": split_library_label_counts(
                    validation_cases,
                    archive_names,
                ),
            }
        )
    return folds


def aggregate_metrics(
    metrics_by_fold: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Pool disjoint fold predictions by summing their confusion counts."""
    if not metrics_by_fold:
        raise ValueError("Cannot aggregate an empty metric list")
    library = calculate_metrics(
        tp=sum(int(metrics["tp"]) for metrics in metrics_by_fold),
        tn=sum(int(metrics["tn"]) for metrics in metrics_by_fold),
        fp=sum(int(metrics["fp"]) for metrics in metrics_by_fold),
        fn=sum(int(metrics["fn"]) for metrics in metrics_by_fold),
    )
    return {
        **library,
        "lib_accuracy": library["accuracy"],
        "lib_precision": library["precision"],
        "lib_recall": library["recall"],
        "lib_specificity": library["specificity"],
        "lib_f1": library["f1"],
        **calculate_cu_metrics(
            sum(int(metrics["cu_tp"]) for metrics in metrics_by_fold),
            sum(int(metrics["cu_fp"]) for metrics in metrics_by_fold),
            sum(int(metrics["cu_fn"]) for metrics in metrics_by_fold),
            sum(int(metrics["cu_evaluated_pairs"]) for metrics in metrics_by_fold),
        ),
    }


def mean_fold_metrics(
    metrics_by_fold: list[dict[str, float | int]],
) -> dict[str, float]:
    """Return equally weighted mean rates, so large programs do not dominate."""
    rate_names = (
        "lib_accuracy",
        "lib_precision",
        "lib_recall",
        "lib_specificity",
        "lib_f1",
        "cu_precision",
        "cu_recall",
        "cu_f1",
    )
    return {
        name: statistics.fmean(float(metrics[name]) for metrics in metrics_by_fold)
        for name in rate_names
    }


def structural_profiles(count: int, seed: int) -> list[dict[str, float | int | str]]:
    default_tuple = tuple(STRUCTURAL_DEFAULTS.values())
    coverage_combinations = [default_tuple]
    for parameter_index, parameter in enumerate(STRUCTURAL_DEFAULTS):
        default_value = STRUCTURAL_DEFAULTS[parameter]
        for candidate in STRUCTURAL_GRIDS[parameter]:
            if candidate == default_value:
                continue
            values = list(default_tuple)
            values[parameter_index] = candidate
            coverage_combinations.append(tuple(values))

    minimum_count = len(coverage_combinations)
    if count < minimum_count:
        raise ValueError(
            f"--structural-trials must be at least {minimum_count} "
            "to cover every structural parameter value"
        )

    interaction_combinations = [
        combination
        for combination in itertools.product(
            *(STRUCTURAL_GRIDS[name] for name in STRUCTURAL_DEFAULTS)
        )
        if combination not in set(coverage_combinations)
    ]
    rng = random.Random(seed)
    rng.shuffle(interaction_combinations)
    chosen = [
        *coverage_combinations,
        *interaction_combinations[: count - minimum_count],
    ]

    profiles = []
    for index, values in enumerate(chosen):
        profile = dict(zip(STRUCTURAL_DEFAULTS, values))
        profile["profile_id"] = f"profile_{index:03d}"
        profiles.append(profile)
    return profiles


def feature_path(output_dir: Path, profile_id: str) -> Path:
    return output_dir / "features" / f"{profile_id}.json.gz"


def collection_command(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    elf_dir: Path,
    lib_dir: Path,
    batch_dir: Path,
    resume: bool,
) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "run_libseeker_batch.py"),
        "--pipeline",
        "current",
        "--dataset-dir",
        str(elf_dir),
        "--libs-dir",
        str(lib_dir),
        "--output-dir",
        str(batch_dir),
        "--all-elfs",
        "--all-libs",
        "--python",
        args.python,
        "--device",
        args.device,
        "--asm-normalization",
        args.asm_normalization,
        "--palmtree-pooling",
        args.palmtree_pooling,
        "--library-score-aggregator",
        args.library_score_aggregator,
        "--block-threshold",
        str(profile["block_threshold"]),
        "--library-min-score",
        "0",
        "--block-locality-window-multiplier",
        str(profile["block_locality_window_multiplier"]),
        "--block-locality-window-padding",
        str(profile["block_locality_window_padding"]),
        "--block-coverage-mean-threshold",
        "0",
        "--block-assignment-quality-threshold",
        "0",
        "--block-min-assignment-ratio",
        "0",
        "--block-min-coverage-ratio",
        "0",
        "--block-min-call-edge-ratio",
        "0",
        "--block-min-function-concentration",
        "0",
        "--block-min-function-spread",
        "0",
    ]
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])
    if resume:
        command.append("--resume")
    return command


def fraction_right(value: str) -> int:
    try:
        return int(value.split("/", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def parse_report(path: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    binary_path = ""
    pending: list[dict[str, Any]] = []
    libraries: dict[str, list[dict[str, Any]]] = {}
    required_fields = {
        "coverage_mean",
        "coverage_ratio",
        "assignment_quality",
        "assignment_ratio",
        "call_edge_ratio",
        "function_concentration",
        "function_spread",
        "rodata",
        "rodata_strings",
        "rodata_ngrams",
        "rodata_bytes",
    }

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Target binary: "):
            binary_path = str(Path(line.removeprefix("Target binary: ")).resolve())
            continue
        if line.lstrip().startswith("BLOCK_CU["):
            fields = dict(FIELD_RE.findall(line))
            if not required_fields.issubset(fields):
                continue
            pending.append(
                {
                    "name": fields.get("name", ""),
                    "coverage_mean": float(fields["coverage_mean"]),
                    "coverage_ratio": float(fields["coverage_ratio"]),
                    "assignment_quality": float(fields["assignment_quality"]),
                    "assignment_ratio": float(fields["assignment_ratio"]),
                    "call_edge_ratio": float(fields["call_edge_ratio"]),
                    "function_concentration": float(fields["function_concentration"]),
                    "function_spread": float(fields["function_spread"]),
                    "rodata": float(fields["rodata"]),
                    "rodata_strings": fraction_right(fields["rodata_strings"]),
                    "rodata_ngrams": fraction_right(fields["rodata_ngrams"]),
                    "rodata_bytes": int(fields["rodata_bytes"]),
                    "target_functions": int(
                        fields.get("target_functions", fields.get("matched_functions", 0))
                    ),
                }
            )
            continue
        summary = SUMMARY_RE.match(line)
        if summary:
            libraries[summary.group("library")] = pending
            pending = []

    if not binary_path:
        raise ValueError(f"Target binary missing from report: {path}")
    return binary_path, libraries


def parse_feature_jsonl(
    path: Path,
) -> tuple[str, dict[str, list[dict[str, Any]]], set[str]]:
    binary_path = ""
    libraries: dict[str, list[dict[str, Any]]] = {}
    pooling_modes: set[str] = set()
    seen_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    decoder = json.JSONDecoder()

    payloads = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        cursor = 0
        while cursor < len(line):
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            if cursor >= len(line):
                break
            try:
                payload, cursor = decoder.raw_decode(line, cursor)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON feature in {path}:{line_number} "
                    f"at column {error.colno}: {error.msg}"
                ) from error
            payloads.append(payload)

    for payload in payloads:
        if payload.get("type") != "block_cu":
            continue
        pooling_modes.add(str(payload.get("palmtree_pooling", "mean")))
        record_binary_path = str(Path(str(payload["binary_path"])).resolve())
        library = str(payload["library"])
        record_name = str(payload.get("name", ""))
        identity = (record_binary_path, library, record_name)
        previous = seen_records.get(identity)
        if previous is not None:
            if previous != payload:
                raise ValueError(
                    "Conflicting duplicate CU feature for "
                    f"binary={record_binary_path}, library={library}, "
                    f"name={record_name} in {path}"
                )
            duplicate_count += 1
            continue
        seen_records[identity] = payload

        if binary_path and record_binary_path != binary_path:
            raise ValueError(f"Mixed binary paths in feature file {path}")
        binary_path = record_binary_path
        windows = [
            {
                "coverage_mean": float(window["coverage_mean"]),
                "coverage_ratio": float(window["coverage_ratio"]),
                "assignment_quality": float(window["assignment_quality"]),
                "assignment_ratio": float(window["assignment_ratio"]),
                "call_edge_ratio": float(window["call_edge_ratio"]),
                "call_edges_evaluated": int(window["call_edges_evaluated"]),
                "call_edges_total": int(window["call_edges_total"]),
                "function_concentration": float(window["function_concentration"]),
                "function_spread": float(window["function_spread"]),
            }
            for window in payload.get("windows", [])
        ]
        libraries.setdefault(library, []).append(
            {
                "name": record_name,
                "coverage_mean": float(payload["coverage_mean"]),
                "coverage_ratio": float(payload["coverage_ratio"]),
                "assignment_quality": float(payload["assignment_quality"]),
                "assignment_ratio": float(payload["assignment_ratio"]),
                "call_edge_ratio": float(payload["call_edge_ratio"]),
                "call_edges_evaluated": int(payload["call_edges_evaluated"]),
                "call_edges_total": int(payload["call_edges_total"]),
                "function_concentration": float(payload["function_concentration"]),
                "function_spread": float(payload["function_spread"]),
                "windows": windows,
                "rodata": float(payload["rodata"]),
                "rodata_has_rodata": bool(payload["rodata_has_rodata"]),
                "rodata_string_score": float(payload["rodata_string_score"]),
                "rodata_byte_score": float(payload["rodata_byte_score"]),
                "rodata_string_informative": bool(
                    payload["rodata_string_informative"]
                ),
                "rodata_byte_informative": bool(
                    payload["rodata_byte_informative"]
                ),
                "rodata_strings": int(payload["rodata_strings"]),
                "rodata_ngrams": int(payload["rodata_ngrams"]),
                "rodata_bytes": int(payload["rodata_bytes"]),
                "target_functions": int(payload["target_functions"]),
            }
        )

    if duplicate_count:
        print(
            f"[features] recovered {path.name}: ignored "
            f"{duplicate_count} exact duplicate CU record(s)"
        )

    return binary_path, libraries, pooling_modes


def report_asm_normalization(path: Path) -> str:
    """Read preprocessing provenance; pre-v2 reports are legacy."""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^Assembly normalization:\s*(legacy|v2)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "legacy"


def report_palmtree_pooling(path: Path) -> str:
    """Read pooling provenance; reports created before schema 6 used mean."""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^PalmTree pooling:\s*(mean|masked_mean)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "mean"


def parse_feature_collection(
    profile: dict[str, float | int | str],
    reports_dir: Path,
    expected_report_count: int,
    input_signature: str,
    archive_names: list[str],
) -> dict[str, Any]:
    reports = sorted(reports_dir.glob("*.report.txt"))
    if len(reports) != expected_report_count:
        raise RuntimeError(
            f"Expected {expected_report_count} reports in {reports_dir}, "
            f"found {len(reports)}"
        )
    cases = {}
    normalization_modes: set[str] = set()
    pooling_modes: set[str] = set()
    for report in reports:
        report_normalization = report_asm_normalization(report)
        report_pooling = report_palmtree_pooling(report)
        normalization_modes.add(report_normalization)
        pooling_modes.add(report_pooling)
        binary_path, libraries = parse_report(report)
        reported_archives = {
            canonical_library_key(library_key, archive_names)
            for library_key in libraries
        }
        feature_file = report.with_name(
            report.name.replace(".report.txt", ".features.jsonl")
        )
        if not feature_file.is_file():
            raise RuntimeError(
                f"Feature schema {FEATURE_SCHEMA_VERSION} requires {feature_file}"
            )
        (
            feature_binary_path,
            feature_libraries,
            feature_pooling_modes,
        ) = parse_feature_jsonl(feature_file)
        if feature_pooling_modes and feature_pooling_modes != {report_pooling}:
            raise ValueError(
                f"Feature/report PalmTree pooling mismatch: {feature_file} "
                f"contains {sorted(feature_pooling_modes)}, report={report_pooling}"
            )
        if feature_binary_path and binary_path and feature_binary_path != binary_path:
            raise ValueError(
                f"Feature/report binary mismatch: {feature_file} vs {report}"
            )
        binary_path = feature_binary_path or binary_path
        libraries.update(feature_libraries)
        normalized_libraries: dict[str, list[dict[str, Any]]] = {}
        for library_key, records in libraries.items():
            canonical_key = canonical_library_key(library_key, archive_names)
            normalized_libraries.setdefault(canonical_key, []).extend(records)
        missing_archives = [
            archive_name
            for archive_name in archive_names
            if archive_name not in reported_archives
        ]
        if missing_archives:
            raise RuntimeError(
                f"Incomplete report {report}: missing library summaries for "
                + ", ".join(missing_archives)
            )
        cases[binary_path] = normalized_libraries
    if len(normalization_modes) != 1:
        raise RuntimeError(
            "Feature collection mixes assembly normalization modes: "
            + ", ".join(sorted(normalization_modes))
        )
    if len(pooling_modes) != 1:
        raise RuntimeError(
            "Feature collection mixes PalmTree pooling modes: "
            + ", ".join(sorted(pooling_modes))
        )
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "asm_normalization": next(iter(normalization_modes)),
        "palmtree_pooling": next(iter(pooling_modes)),
        "profile": profile,
        "input_signature": input_signature,
        "cases": cases,
    }


def write_gzip_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def structural_profile_matches(
    expected: dict[str, float | int | str],
    collection: dict[str, Any],
) -> bool:
    actual = collection.get("profile", {})
    return all(actual.get(key) == expected.get(key) for key in STRUCTURAL_DEFAULTS)


def normalize_existing_collection(
    collection: dict[str, Any],
    profile: dict[str, float | int | str],
    input_signature: str,
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, Any]:
    normalized_cases = {}
    existing_cases = collection.get("cases", {})
    for case in cases:
        binary_path = str(case["binary"])
        existing_libraries = existing_cases.get(binary_path, {})
        normalized_libraries = {}
        for archive_name in archive_names:
            found, records = find_archive_records(existing_libraries, archive_name)
            if found:
                normalized_libraries[archive_name] = records
        normalized_cases[binary_path] = normalized_libraries
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "asm_normalization": collection.get("asm_normalization", "legacy"),
        "palmtree_pooling": collection.get("palmtree_pooling", "mean"),
        "profile": profile,
        "input_signature": input_signature,
        "cases": normalized_cases,
    }


def count_cached_pairs(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> int:
    count = 0
    for case in cases:
        library_records = collection["cases"].get(str(case["binary"]), {})
        for archive_name in archive_names:
            if archive_name in library_records:
                count += 1
    return count


def reusable_feature_paths(
    args: argparse.Namespace,
    profile_id: str,
    destination: Path,
) -> list[Path]:
    paths = []
    seen = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        paths.append(resolved)

    add(destination)
    for root in args.reuse_feature_dir:
        root = root.resolve()
        if root.is_file():
            add(root)
        elif (root / "features").is_dir():
            add(root / "features" / f"{profile_id}.json.gz")
        elif root.is_dir():
            add(root / f"{profile_id}.json.gz")
    return paths


def find_partial_reuse_collection(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    destination: Path,
    input_signature: str,
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, Any] | None:
    best_collection = None
    best_count = 0
    profile_id = str(profile["profile_id"])

    for path in reusable_feature_paths(args, profile_id, destination):
        try:
            existing = read_gzip_json(path)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as error:
            print(f"[features] cannot reuse {path}: {error}")
            continue
        if existing.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            continue
        if existing.get("asm_normalization") != args.asm_normalization:
            continue
        if existing.get("palmtree_pooling", "mean") != args.palmtree_pooling:
            continue
        if not structural_profile_matches(profile, existing):
            continue

        normalized = normalize_existing_collection(
            existing, profile, input_signature, cases, archive_names
        )
        cached_count = count_cached_pairs(normalized, cases, archive_names)
        if cached_count > best_count:
            best_collection = normalized
            best_count = cached_count
            print(
                f"[features] partial reuse candidate {profile_id}: "
                f"{cached_count}/{len(cases) * len(archive_names)} pairs from {path}"
            )

    return best_collection


def missing_cases_by_archive(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    missing: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        binary_path = str(case["binary"])
        library_records = collection["cases"].setdefault(binary_path, {})
        for archive_name in archive_names:
            if archive_name not in library_records:
                missing.setdefault(archive_name, []).append(case)
    return missing


def merge_collection(
    destination: dict[str, Any],
    addition: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_name: str,
) -> None:
    if destination.get("asm_normalization", "legacy") != addition.get(
        "asm_normalization", "legacy"
    ):
        raise ValueError("Cannot merge different assembly normalization modes")
    if destination.get("palmtree_pooling", "mean") != addition.get(
        "palmtree_pooling", "mean"
    ):
        raise ValueError("Cannot merge different PalmTree pooling modes")
    for case in cases:
        binary_path = str(case["binary"])
        added_libraries = addition["cases"].get(binary_path, {})
        found, records = find_archive_records(added_libraries, archive_name)
        if not found:
            raise KeyError(
                f"Missing collected records for {archive_name} in {binary_path}"
            )
        destination["cases"].setdefault(binary_path, {})[archive_name] = records


def collect_missing_archive_features(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    collection: dict[str, Any],
    archive_name: str,
    missing_cases: list[dict[str, Any]],
    selected_archives: dict[str, dict[str, str]],
    input_signature: str,
) -> None:
    profile_id = str(profile["profile_id"])
    archive = selected_archives[archive_name]
    elf_dir, lib_dir = prepare_subset_inputs(
        args, profile_id, archive_name, missing_cases, archive
    )
    batch_dir = (
        args.output_dir
        / "collections_missing"
        / profile_id
        / safe_path_part(archive_name)
    )
    missing_signature = selection_signature(
        [
            {
                "input": input_signature,
                "profile": profile,
                "archive": archive,
                "cases": [case["variant"] for case in missing_cases],
            }
        ]
    )
    batch_signature = batch_dir / "input_signature.txt"
    resume_batch = (
        args.resume
        and batch_signature.is_file()
        and batch_signature.read_text(encoding="utf-8").strip()
        == missing_signature
    )
    if batch_dir.exists() and not resume_batch:
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_signature.write_text(missing_signature + "\n", encoding="utf-8")

    command = collection_command(
        args, profile, elf_dir, lib_dir, batch_dir, resume=resume_batch
    )
    print(
        f"[features] collect missing {profile_id} {archive_name}: "
        f"{len(missing_cases)} ELF(s)"
    )
    print("+", " ".join(command), flush=True)
    if args.dry_run:
        return

    result = subprocess.run(command, cwd=SCRIPT_DIR)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)

    addition = parse_feature_collection(
        profile,
        batch_dir / "reports/current",
        len(missing_cases),
        missing_signature,
        [archive_name],
    )
    merge_collection(collection, addition, missing_cases, archive_name)


def collect_features(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    elf_dir: Path,
    lib_dir: Path,
    input_signature: str,
    selected_archives: list[dict[str, str]],
    profiles: list[dict[str, float | int | str]],
) -> dict[str, dict[str, Any]]:
    collections = {}
    archive_names = [archive["name"] for archive in selected_archives]
    selected_archives_by_name = {archive["name"]: archive for archive in selected_archives}
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        profile_signature = selection_signature(
            [{"input": input_signature, **profile}]
        )
        destination = feature_path(args.output_dir, profile_id)
        if (args.resume or args.search_only) and destination.is_file():
            existing = read_gzip_json(destination)
            if (
                existing.get("input_signature") == profile_signature
                and existing.get("asm_normalization") == args.asm_normalization
                and existing.get("palmtree_pooling", "mean")
                == args.palmtree_pooling
            ):
                normalized = normalize_existing_collection(
                    existing, profile, profile_signature, cases, archive_names
                )
                expected_pairs = len(cases) * len(archive_names)
                cached_pairs = count_cached_pairs(
                    normalized, cases, archive_names
                )
                if cached_pairs == expected_pairs:
                    print(f"[features] reuse {profile_id}")
                    collections[profile_id] = normalized
                    continue
                if args.search_only:
                    raise ValueError(
                        f"Incomplete feature collection {destination}: "
                        f"{cached_pairs}/{expected_pairs} ELF/library pairs"
                    )
                print(
                    f"[features] discard incomplete {profile_id}: "
                    f"{cached_pairs}/{expected_pairs} ELF/library pairs"
                )
                destination.unlink()
            if args.search_only:
                partial = find_partial_reuse_collection(
                    args,
                    profile,
                    destination,
                    profile_signature,
                    cases,
                    archive_names,
                )
                if partial is None:
                    raise ValueError(
                        f"Feature selection mismatch in {destination}; recollect it"
                    )
                missing = missing_cases_by_archive(partial, cases, archive_names)
                if missing:
                    raise ValueError(
                        f"Feature selection mismatch in {destination}; "
                        f"missing {sorted(missing)}"
                    )
                collections[profile_id] = partial
                continue
            print(f"[features] selection changed, try partial reuse {profile_id}")
        if args.search_only:
            partial = find_partial_reuse_collection(
                args,
                profile,
                destination,
                profile_signature,
                cases,
                archive_names,
            )
            if partial is None:
                raise FileNotFoundError(
                    f"Missing compatible feature file for {profile_id}; "
                    "provide it through --reuse-feature-dir"
                )
            missing = missing_cases_by_archive(partial, cases, archive_names)
            if missing:
                raise ValueError(
                    f"Incomplete reused feature collection for {profile_id}; "
                    f"missing {sorted(missing)}"
                )
            print(f"[features] search-only reuse complete {profile_id}")
            collections[profile_id] = partial
            continue

        partial = find_partial_reuse_collection(
            args, profile, destination, profile_signature, cases, archive_names
        )
        if partial is not None:
            missing = missing_cases_by_archive(partial, cases, archive_names)
            if missing:
                print(
                    f"[features] partial reuse {profile_id}: "
                    f"{count_cached_pairs(partial, cases, archive_names)}/"
                    f"{len(cases) * len(archive_names)} pairs cached"
                )
                for archive_name, missing_cases in sorted(missing.items()):
                    collect_missing_archive_features(
                        args,
                        profile,
                        partial,
                        archive_name,
                        missing_cases,
                        selected_archives_by_name,
                        profile_signature,
                    )
            else:
                print(f"[features] partial reuse complete {profile_id}")
            if not args.dry_run:
                write_gzip_json(destination, partial)
                collections[profile_id] = partial
            continue

        batch_dir = (
            args.output_dir / "collections" / profile_id
        )
        batch_signature = batch_dir / "input_signature.txt"
        resume_batch = (
            args.resume
            and batch_signature.is_file()
            and batch_signature.read_text(encoding="utf-8").strip()
            == profile_signature
        )
        if resume_batch and not args.dry_run:
            try:
                recovered = parse_feature_collection(
                    profile,
                    batch_dir / "reports/current",
                    len(cases),
                    profile_signature,
                    archive_names,
                )
            except (OSError, ValueError, RuntimeError) as error:
                print(
                    f"[features] existing batch {profile_id} is not complete: "
                    f"{error}"
                )
            else:
                print(f"[features] recover complete batch {profile_id}")
                write_gzip_json(destination, recovered)
                collections[profile_id] = recovered
                continue
        if batch_dir.exists() and not resume_batch:
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_signature.write_text(profile_signature + "\n", encoding="utf-8")
        command = collection_command(
            args, profile, elf_dir, lib_dir, batch_dir, resume=resume_batch
        )
        print("+", " ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=SCRIPT_DIR)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

        collection = parse_feature_collection(
            profile,
            batch_dir / "reports/current",
            len(cases),
            profile_signature,
            archive_names,
        )
        write_gzip_json(destination, collection)
        collections[profile_id] = collection
    return collections


def feature_input_signature(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, Any]],
) -> str:
    return selection_signature(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "asm_normalization": args.asm_normalization,
            "palmtree_pooling": args.palmtree_pooling,
            "objective_order": args.objective_order,
            "elf_variants": [case["variant"] for case in cases],
            "archives": selected_archives,
        }
    )


def rodata_is_informative(
    record: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    if (
        not bool(params["rodata_filter_enabled"])
        or not record.get("rodata_has_rodata", False)
    ):
        return False

    min_bytes = int(params["rodata_min_bytes"])
    min_strings = int(params["rodata_min_strings"])
    min_ngrams = int(params["rodata_min_ngrams"])
    return (
        record["rodata_bytes"] >= min_bytes
        or record["rodata_strings"] >= min_strings
        or record["rodata_ngrams"] >= min_ngrams
    )


def rodata_is_penalty(record: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        rodata_is_informative(record, params)
        and record["rodata"] <= params["rodata_penalty_threshold"]
    )


def apply_replayed_rodata_bonus(
    block_score: float,
    record: dict[str, Any],
    params: dict[str, Any],
) -> float:
    """Replay the bounded production bonus for informative high .rodata."""
    if not rodata_is_informative(record, params):
        return block_score

    threshold = float(params.get("rodata_confirm_threshold", 0.70))
    bonus_weight = float(params.get("rodata_bonus_weight", 0.30))
    rodata_score = float(record["rodata"])
    if rodata_score < threshold or bonus_weight <= 0.0:
        return block_score

    confidence = (
        1.0
        if threshold >= 1.0
        else (rodata_score - threshold) / (1.0 - threshold)
    )
    confidence = min(1.0, max(0.0, confidence))
    adjusted = block_score + bonus_weight * confidence * (1.0 - block_score)
    return min(1.0, max(block_score, adjusted))


def block_window_passes(
    window: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    return (
        window["coverage_mean"] >= params["block_coverage_mean_threshold"]
        and window["coverage_ratio"] >= params["block_min_coverage_ratio"]
        and window["assignment_quality"]
        >= params["block_assignment_quality_threshold"]
        and window["assignment_ratio"] >= params["block_min_assignment_ratio"]
        and window["call_edge_ratio"] >= params["block_min_call_edge_ratio"]
        and window["function_concentration"]
        >= params["block_min_function_concentration"]
        and window["function_spread"] >= params["block_min_function_spread"]
    )


def record_passes(record: dict[str, Any], params: dict[str, Any]) -> bool:
    return record_match_score(record, params) is not None


def record_match_score(
    record: dict[str, Any],
    params: dict[str, Any],
) -> float | None:
    """Return the replayed CU score, or None when the CU is rejected."""
    candidate_windows = record.get("windows") or [record]
    passing_windows = [
        window
        for window in candidate_windows
        if block_window_passes(window, params)
    ]
    if not passing_windows or rodata_is_penalty(record, params):
        return None
    block_score = max(float(window["coverage_mean"]) for window in passing_windows)
    return apply_replayed_rodata_bonus(block_score, record, params)


def library_match_evidence(
    records: list[dict[str, Any]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """Return accepted CU records and their aggregated library score."""
    accepted: list[dict[str, Any]] = []
    scores: list[float] = []
    for record in records:
        candidate_windows = record.get("windows") or [record]
        structural_match = any(
            block_window_passes(window, params) for window in candidate_windows
        )
        score = record_match_score(record, params)
        if score is None:
            if (
                structural_match
                and str(params.get("library_score_aggregator", "mean"))
                == "top3_mean"
            ):
                # Preserve the structural top-3 denominator: a .rodata
                # rejection must not increase the library score.
                scores.append(0.0)
            continue
        accepted.append(record)
        scores.append(score)
    library_score = aggregate_library_score(
        scores,
        str(params.get("library_score_aggregator", "mean")),
        score_floor=float(params.get("block_coverage_mean_threshold", 0.0)),
    )
    return accepted, library_score


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    total = tp + tn + fp + fn

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    mcc_denominator = (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) ** 0.5
    return {
        "evaluated": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": ratio((tp * tn) - (fp * fn), mcc_denominator),
        "positive_prevalence": ratio(tp + fn, total),
    }


def calculate_cu_metrics(tp: int, fp: int, fn: int, evaluated_pairs: int) -> dict[str, float | int]:
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "cu_evaluated_pairs": evaluated_pairs,
        "cu_tp": tp,
        "cu_fp": fp,
        "cu_fn": fn,
        "cu_precision": ratio(tp, tp + fp),
        "cu_recall": ratio(tp, tp + fn),
        "cu_f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def cu_counts_for_prediction(
    archive_name: str,
    case: dict[str, Any],
    records: list[dict[str, Any]],
    successful: list[dict[str, Any]],
) -> tuple[dict[str, int | bool], bool]:
    """Score predicted CU identities; names are labels, never decision features."""
    family = archive_family(archive_name)
    expected = family in case["expected"]
    expected_cus_by_family = case.get("expected_cus", {})
    expected_cus = set(expected_cus_by_family.get(family, set()))

    has_record_names = all("name" in record for record in records)
    if not has_record_names:
        return {
            "cu_ground_truth": False,
            "expected_cu": 0,
            "cu_tp": 0,
            "cu_fp": 0,
            "cu_fn": 0,
        }, False

    predicted_cus = {
        normalize_cu_name(str(record["name"]))
        for record in successful
        if record.get("name")
    }

    if expected:
        if family not in expected_cus_by_family:
            return {
                "cu_ground_truth": False,
                "expected_cu": 0,
                "cu_tp": 0,
                "cu_fp": 0,
                "cu_fn": 0,
            }, False
        tp = len(predicted_cus & expected_cus)
        fp = len(predicted_cus - expected_cus)
        fn = len(expected_cus - predicted_cus)
    else:
        tp = 0
        fp = len(predicted_cus)
        fn = 0

    return {
        "cu_ground_truth": True,
        "expected_cu": len(expected_cus),
        "cu_tp": tp,
        "cu_fp": fp,
        "cu_fn": fn,
    }, True


def evaluate(
    params: dict[str, Any],
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    collected_cases = collection["cases"]
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    cu_counts = {"tp": 0, "fp": 0, "fn": 0}
    cu_evaluated_pairs = 0
    predictions = []

    for case in cases:
        binary_path = str(case["binary"])
        library_records = collected_cases.get(binary_path)
        if library_records is None:
            raise KeyError(f"No collected features for {binary_path}")

        for archive_name in archive_names:
            _, records = find_archive_records(library_records, archive_name)
            successful, library_score = library_match_evidence(records, params)
            predicted = bool(successful) and library_score >= float(
                params.get("library_min_score", 0.0)
            )
            expected = archive_family(archive_name) in case["expected"]
            cu_prediction, cu_evaluated = cu_counts_for_prediction(
                archive_name,
                case,
                records,
                successful,
            )
            if cu_evaluated:
                cu_evaluated_pairs += 1
                cu_counts["tp"] += int(cu_prediction["cu_tp"])
                cu_counts["fp"] += int(cu_prediction["cu_fp"])
                cu_counts["fn"] += int(cu_prediction["cu_fn"])
            classification = (
                "TP" if expected and predicted
                else "FN" if expected
                else "FP" if predicted
                else "TN"
            )
            counts[classification.lower()] += 1
            predictions.append(
                {
                    "variant": case["variant"],
                    "program": case["program"],
                    "compiler": case["compiler"],
                    "elf_optimization": case["elf_optimization"],
                    "library": archive_name,
                    "expected": expected,
                    "predicted": predicted,
                    "classification": classification,
                    "library_score": library_score,
                    "matched_cu": len(successful),
                    "total_cu": len(records),
                    **cu_prediction,
                }
            )

    library_metrics = calculate_metrics(**counts)
    metrics = {
        **library_metrics,
        "lib_accuracy": library_metrics["accuracy"],
        "lib_precision": library_metrics["precision"],
        "lib_recall": library_metrics["recall"],
        "lib_specificity": library_metrics["specificity"],
        "lib_f1": library_metrics["f1"],
        **calculate_cu_metrics(
            cu_counts["tp"],
            cu_counts["fp"],
            cu_counts["fn"],
            cu_evaluated_pairs,
        ),
    }
    return metrics, predictions


def objective(
    metrics: dict[str, float | int],
    objective_order: str,
) -> tuple[float, ...]:
    cu_metrics = (
        float(metrics["cu_f1"]),
        float(metrics["cu_precision"]),
        float(metrics["cu_recall"]),
    )
    lib_metrics = (
        float(metrics["lib_f1"]),
        float(metrics["lib_precision"]),
        float(metrics["lib_recall"]),
    )
    if objective_order == "lib_only":
        return lib_metrics
    if objective_order == "lib_first":
        return (*lib_metrics, *cu_metrics)
    return (*cu_metrics, *lib_metrics)


def decision_defaults(library_score_aggregator: str = "mean") -> dict[str, Any]:
    return {
        **DECISION_DEFAULTS,
        "library_score_aggregator": library_score_aggregator,
    }


def random_decision_params(
    rng: random.Random,
    library_score_aggregator: str = "mean",
) -> dict[str, Any]:
    params = decision_defaults(library_score_aggregator)
    for parameter, candidates in DECISION_GRIDS.items():
        params[parameter] = rng.choice(candidates)
    return params


def greedy_profile_search(
    args: argparse.Namespace,
    profile_id: str,
    collection: dict[str, Any],
    train_cases: list[dict[str, Any]],
    archive_names: list[str],
    rng: random.Random,
    split_label: str = "train",
    fold_id: str = "",
) -> tuple[dict[str, Any], dict[str, float | int], list[dict[str, Any]]]:
    best_params = decision_defaults(args.library_score_aggregator)
    best_metrics, _ = evaluate(
        best_params, collection, train_cases, archive_names
    )
    history = []

    for restart in range(args.restarts):
        params = (
            decision_defaults(args.library_score_aggregator)
            if restart == 0
            else random_decision_params(rng, args.library_score_aggregator)
        )
        current_metrics, _ = evaluate(
            params, collection, train_cases, archive_names
        )

        for round_number in range(1, args.max_rounds + 1):
            changed = False
            for parameter, candidates in DECISION_GRIDS.items():
                old_value = params[parameter]
                selected_value = old_value
                selected_metrics = current_metrics
                selected_key = objective(current_metrics, args.objective_order)

                for candidate in candidates:
                    trial = dict(params)
                    trial[parameter] = candidate
                    metrics, _ = evaluate(
                        trial, collection, train_cases, archive_names
                    )
                    key = objective(metrics, args.objective_order)
                    history.append(
                        {
                            "split": split_label,
                            "fold_id": fold_id,
                            "profile_id": profile_id,
                            "restart": restart,
                            "round": round_number,
                            "parameter": parameter,
                            "candidate": candidate,
                            **metrics,
                        }
                    )
                    if key > selected_key:
                        selected_value = candidate
                        selected_metrics = metrics
                        selected_key = key

                if selected_value != old_value:
                    changed = True
                    params[parameter] = selected_value
                    current_metrics = selected_metrics
            if not changed:
                break

        if objective(current_metrics, args.objective_order) > objective(
            best_metrics,
            args.objective_order,
        ):
            best_params = params
            best_metrics = current_metrics

    return best_params, best_metrics, history


def search_all_profiles(
    args: argparse.Namespace,
    collections: dict[str, dict[str, Any]],
    train_cases: list[dict[str, Any]],
    validation_cases: list[dict[str, Any]],
    archive_names: list[str],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, float | int],
    dict[str, float | int],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rng = random.Random(args.seed + 1)
    best_profile_id = ""
    best_params: dict[str, Any] = {}
    best_train_metrics: dict[str, float | int] = {}
    best_validation_metrics: dict[str, float | int] = {}
    all_history = []
    profile_rows = []

    for profile_id, collection in sorted(collections.items()):
        params, train_metrics, history = greedy_profile_search(
            args,
            profile_id,
            collection,
            train_cases,
            archive_names,
            rng,
        )
        validation_metrics, _ = evaluate(
            params,
            collection,
            validation_cases,
            archive_names,
        )
        all_history.extend(history)
        profile = collection["profile"]
        profile_rows.append(
            {
                "threshold_tuning_split": "train",
                "profile_selection_split": "validation",
                **profile,
                **{f"best_{key}": value for key, value in params.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        print(
            f"[{profile_id}] "
            f"train CU_F1={float(train_metrics['cu_f1']):.4f} "
            f"lib_F1={float(train_metrics['lib_f1']):.4f} | "
            f"validation CU_F1={float(validation_metrics['cu_f1']):.4f} "
            f"lib_F1={float(validation_metrics['lib_f1']):.4f}"
        )

        if (
            not best_validation_metrics
            or objective(validation_metrics, args.objective_order)
            > objective(best_validation_metrics, args.objective_order)
        ):
            best_profile_id = profile_id
            best_params = params
            best_train_metrics = train_metrics
            best_validation_metrics = validation_metrics

    return (
        best_profile_id,
        best_params,
        best_train_metrics,
        best_validation_metrics,
        all_history,
        profile_rows,
    )


def cross_validation_objective(
    fold_metrics: list[dict[str, float | int]],
    objective_order: str,
) -> tuple[float, ...]:
    """Prefer high mean fold quality, then low primary-F1 variability."""
    means = mean_fold_metrics(fold_metrics)
    base = objective(means, objective_order)
    primary_name = (
        "lib_f1"
        if objective_order in {"lib_first", "lib_only"}
        else "cu_f1"
    )
    primary_std = statistics.pstdev(
        float(metrics[primary_name]) for metrics in fold_metrics
    )
    return (base[0], -primary_std, *base[1:])


def search_all_profiles_grouped_cv(
    args: argparse.Namespace,
    collections: dict[str, dict[str, Any]],
    development_cases: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    archive_names: list[str],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, float | int],
    dict[str, float | int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Select a structural profile by grouped CV, then retune on development."""
    best_profile_id = ""
    best_fold_metrics: list[dict[str, float | int]] = []
    best_cv_key: tuple[float, ...] | None = None
    all_history: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for profile_id, collection in sorted(collections.items()):
        current_fold_metrics: list[dict[str, float | int]] = []
        for fold in folds:
            fold_id = str(fold["fold_id"])
            params, train_metrics, history = greedy_profile_search(
                args,
                profile_id,
                collection,
                fold["train_cases"],
                archive_names,
                random.Random(f"{args.seed}:{profile_id}:{fold_id}"),
                split_label="cv_train",
                fold_id=fold_id,
            )
            validation_metrics, _ = evaluate(
                params,
                collection,
                fold["validation_cases"],
                archive_names,
            )
            current_fold_metrics.append(validation_metrics)
            all_history.extend(history)
            fold_rows.append(
                {
                    "profile_id": profile_id,
                    "fold_id": fold_id,
                    "train_programs": ",".join(fold["train_programs"]),
                    "validation_programs": ",".join(fold["validation_programs"]),
                    **{f"best_{key}": value for key, value in params.items()},
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
            )

        pooled_metrics = aggregate_metrics(current_fold_metrics)
        mean_metrics = mean_fold_metrics(current_fold_metrics)
        lib_f1_std = statistics.pstdev(
            float(metrics["lib_f1"]) for metrics in current_fold_metrics
        )
        cu_f1_std = statistics.pstdev(
            float(metrics["cu_f1"]) for metrics in current_fold_metrics
        )
        profile = collection["profile"]
        profile_rows.append(
            {
                "threshold_tuning_split": "grouped_cv_train_folds",
                "profile_selection_split": "grouped_cv_validation_folds",
                **profile,
                **{f"cv_mean_{key}": value for key, value in mean_metrics.items()},
                "cv_std_lib_f1": lib_f1_std,
                "cv_std_cu_f1": cu_f1_std,
                **{f"cv_pooled_{key}": value for key, value in pooled_metrics.items()},
            }
        )
        print(
            f"[{profile_id}] grouped-CV "
            f"lib_F1={mean_metrics['lib_f1']:.4f}+/-{lib_f1_std:.4f} "
            f"CU_F1={mean_metrics['cu_f1']:.4f}+/-{cu_f1_std:.4f} | "
            f"pooled lib_F1={float(pooled_metrics['lib_f1']):.4f}"
        )

        key = cross_validation_objective(current_fold_metrics, args.objective_order)
        if best_cv_key is None or key > best_cv_key:
            best_profile_id = profile_id
            best_fold_metrics = current_fold_metrics
            best_cv_key = key

    if not best_profile_id:
        raise RuntimeError("Grouped cross-validation did not select a profile")

    best_collection = collections[best_profile_id]
    final_params, development_metrics, final_history = greedy_profile_search(
        args,
        best_profile_id,
        best_collection,
        development_cases,
        archive_names,
        random.Random(f"{args.seed}:{best_profile_id}:full-development"),
        split_label="full_development",
        fold_id="all",
    )
    all_history.extend(final_history)
    pooled_cv_metrics = aggregate_metrics(best_fold_metrics)
    mean_cv_metrics = mean_fold_metrics(best_fold_metrics)
    cv_summary = {
        "fold_count": len(folds),
        "mean_metrics": mean_cv_metrics,
        "pooled_metrics": pooled_cv_metrics,
        "lib_f1_std": statistics.pstdev(
            float(metrics["lib_f1"]) for metrics in best_fold_metrics
        ),
        "cu_f1_std": statistics.pstdev(
            float(metrics["cu_f1"]) for metrics in best_fold_metrics
        ),
    }
    return (
        best_profile_id,
        final_params,
        development_metrics,
        pooled_cv_metrics,
        all_history,
        profile_rows,
        fold_rows,
        cv_summary,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_metrics(
    params: dict[str, Any],
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    groups = [
        ("ALL", lambda case: True),
        ("gcc", lambda case: case["compiler"] == "gcc"),
        ("clang", lambda case: case["compiler"] == "clang"),
    ]
    groups.extend(
        (program, lambda case, selected=program: case["program"] == selected)
        for program in sorted({case["program"] for case in cases})
    )
    groups.extend(
        (
            f"optimization:{optimization}",
            lambda case, selected=optimization: case["elf_optimization"] == selected,
        )
        for optimization in sorted({case["elf_optimization"] for case in cases})
    )
    for group, predicate in groups:
        selected_cases = [case for case in cases if predicate(case)]
        metrics, _ = evaluate(params, collection, selected_cases, archive_names)
        rows.append({"group": group, **metrics})
    return rows


def metrics_by_library(
    params: dict[str, Any],
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> list[dict[str, Any]]:
    """Expose family-specific test failures hidden by aggregate class imbalance."""
    rows = []
    for archive_name in archive_names:
        metrics, _ = evaluate(params, collection, cases, [archive_name])
        rows.append(
            {
                "library": archive_name,
                "family": archive_family(archive_name),
                **metrics,
            }
        )
    return rows


def acquire_output_lock(output_dir: Path):
    """Prevent concurrent greedy runs from writing the same feature files."""
    lock_path = output_dir / ".greedy_threshold_search.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(
            f"Another greedy search is already using {output_dir} "
            f"(PID {owner}). Stop it or choose a different --output-dir."
        ) from error

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    args = parse_args()
    if args.search_only and args.collect_only:
        raise ValueError("--search-only and --collect-only are mutually exclusive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_lock = acquire_output_lock(args.output_dir)
    args.lib_roots = resolve_lib_roots(args)

    cases = load_cases(args)
    train_programs, validation_programs, test_programs = select_program_splits(
        cases,
        args.validation_program,
        args.test_program,
        args.seed,
        args.include_glibc,
        args.test_program_count,
    )
    train_pool = [case for case in cases if case["program"] in train_programs]
    validation_pool = [
        case for case in cases if case["program"] in validation_programs
    ]
    test_pool = [case for case in cases if case["program"] in test_programs]

    train_cases = select_split_cases(
        train_pool,
        None if args.elf_variants else args.train_elf_count,
        args.seed + 11,
        "train",
    )
    validation_cases = select_split_cases(
        validation_pool,
        args.validation_elf_count,
        args.seed + 12,
        "validation",
    )
    test_cases = select_split_cases(
        test_pool,
        args.test_elf_count,
        args.seed + 13,
        "test",
    )
    development_cases = sorted(
        [*train_cases, *validation_cases],
        key=lambda case: case["variant"],
    )
    search_cases = sorted(
        [*development_cases, *test_cases],
        key=lambda case: case["variant"],
    )
    rng = random.Random(args.seed)
    archive_selection_cases = (
        development_cases
        if args.validation_mode == "grouped_cv"
        else train_cases
    )
    positive_archives = select_positive_archives(
        args,
        archive_selection_cases,
        rng,
    )
    negative_archives = select_negative_archives(args, positive_archives, rng)
    selected_archives = sorted(
        positive_archives + negative_archives,
        key=lambda item: item["name"],
    )
    archive_names = [item["name"] for item in selected_archives]
    split_cases_by_name = {
        "train": train_cases,
        "validation": validation_cases,
        "development": development_cases,
        "test": test_cases,
    }
    split_label_counts = {
        name: split_library_label_counts(split_cases, archive_names)
        for name, split_cases in split_cases_by_name.items()
    }
    required_splits = (
        ("development", "test")
        if args.validation_mode == "grouped_cv"
        else ("train", "validation", "test")
    )
    unusable_splits = [
        name
        for name in required_splits
        if split_label_counts[name]["positive"] == 0
        or split_label_counts[name]["negative"] == 0
    ]
    if unusable_splits:
        raise ValueError(
            "Library panel does not provide both positive and negative labels in "
            + ", ".join(unusable_splits)
            + "; choose different --validation-program/--test-program values or "
            "expand the library panel"
        )
    if split_label_counts["test"]["positive"] < args.min_test_positive_labels:
        raise ValueError(
            "Final test contains only "
            f"{split_label_counts['test']['positive']} positive library label(s); "
            f"--min-test-positive-labels={args.min_test_positive_labels}. "
            "Add test programs/variants or lower the minimum only for a smoke test."
        )
    cv_folds = (
        build_grouped_cv_folds(
            development_cases,
            archive_names,
            args.cv_folds,
            args.seed,
        )
        if args.validation_mode == "grouped_cv"
        else []
    )
    structural_profile_selection = structural_profiles(
        args.structural_trials,
        args.seed,
    )
    input_signature = feature_input_signature(
        args,
        search_cases,
        selected_archives,
    )
    selection_report = {
        "seed": args.seed,
        "input_signature": input_signature,
        "positive_count": len(positive_archives),
        "negative_count": len(negative_archives),
        "archive_selection_split": (
            "development" if args.validation_mode == "grouped_cv" else "train"
        ),
        "validation_mode": args.validation_mode,
        "device": args.device,
        "asm_normalization": args.asm_normalization,
        "palmtree_pooling": args.palmtree_pooling,
        "library_score_aggregator": args.library_score_aggregator,
        "objective_order": args.objective_order,
        "max_positive_archive_bytes": args.max_positive_archive_bytes,
        "lib_roots": [str(root) for root in args.lib_roots],
        "elf_variants": [case["variant"] for case in search_cases],
        "splits": {
            "train": {
                "assigned_programs": sorted(train_programs),
                "programs": sorted({case["program"] for case in train_cases}),
                "elf_variants": [case["variant"] for case in train_cases],
                "library_labels": split_label_counts["train"],
            },
            "validation": {
                "assigned_programs": sorted(validation_programs),
                "programs": sorted({case["program"] for case in validation_cases}),
                "elf_variants": [case["variant"] for case in validation_cases],
                "library_labels": split_label_counts["validation"],
            },
            "development": {
                "assigned_programs": sorted(train_programs | validation_programs),
                "programs": sorted(
                    {case["program"] for case in development_cases}
                ),
                "elf_variants": [case["variant"] for case in development_cases],
                "library_labels": split_label_counts["development"],
            },
            "test": {
                "assigned_programs": sorted(test_programs),
                "programs": sorted({case["program"] for case in test_cases}),
                "elf_variants": [case["variant"] for case in test_cases],
                "library_labels": split_label_counts["test"],
            },
        },
        "cross_validation_folds": [
            {
                "fold_id": fold["fold_id"],
                "train_programs": fold["train_programs"],
                "validation_programs": fold["validation_programs"],
                "train_elf_variants": [
                    case["variant"] for case in fold["train_cases"]
                ],
                "validation_elf_variants": [
                    case["variant"] for case in fold["validation_cases"]
                ],
                "train_labels": fold["train_labels"],
                "validation_labels": fold["validation_labels"],
            }
            for fold in cv_folds
        ],
        "structural_profiles": structural_profile_selection,
        "archives": selected_archives,
    }
    (args.output_dir / "input_selection.json").write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_rows = [
        {
            "split": split_name,
            "variant": case["variant"],
            "program": case["program"],
            "compiler": case["compiler"],
            "elf_optimization": case["elf_optimization"],
            "binary": str(case["binary"]),
        }
        for split_name, split_cases in (
            ("train", train_cases),
            ("validation", validation_cases),
            ("test", test_cases),
        )
        for case in split_cases
    ]
    write_csv(args.output_dir / "split_cases.csv", split_rows)
    write_csv(
        args.output_dir / "confirmed_cu_ground_truth.csv",
        [
            {
                "split": (
                    "test"
                    if case["program"] in test_programs
                    else "development"
                ),
                "variant": case["variant"],
                "program": case["program"],
                "library_family": family,
                "compilation_unit": (
                    name if str(name).endswith(".o") else f"{name}.o"
                ),
                "ground_truth_method": "linker_map",
            }
            for case in search_cases
            for family, names in sorted(case.get("expected_cus", {}).items())
            for name in sorted(names)
        ],
    )
    if cv_folds:
        write_csv(
            args.output_dir / "cross_validation_cases.csv",
            [
                {
                    "fold_id": fold["fold_id"],
                    "role": role,
                    "variant": case["variant"],
                    "program": case["program"],
                    "compiler": case["compiler"],
                    "elf_optimization": case["elf_optimization"],
                    "binary": str(case["binary"]),
                }
                for fold in cv_folds
                for role, selected_cases in (
                    ("train", fold["train_cases"]),
                    ("validation", fold["validation_cases"]),
                )
                for case in selected_cases
            ],
        )

    elf_dir, lib_dir = prepare_inputs(args, search_cases, selected_archives)
    print(
        f"Selected {len(search_cases)} / {len(cases)} ELF files, "
        f"{len(positive_archives)} positive archives and "
        f"{len(negative_archives)} negative archives"
    )
    print(
        f"Train: {len(train_cases)} ELF files from "
        f"{', '.join(sorted({case['program'] for case in train_cases}))}"
    )
    print(
        f"Validation: {len(validation_cases)} ELF files from "
        f"{', '.join(sorted({case['program'] for case in validation_cases}))}"
    )
    print(
        f"Test: {len(test_cases)} ELF files from "
        f"{', '.join(sorted({case['program'] for case in test_cases}))}"
    )
    if cv_folds:
        print(
            f"Grouped CV: {len(cv_folds)} folds over "
            f"{len(development_cases)} development ELF files"
        )
        for fold in cv_folds:
            print(
                f"  {fold['fold_id']}: validation programs="
                f"{','.join(fold['validation_programs'])} "
                f"ELFs={len(fold['validation_cases'])} "
                f"positive labels={fold['validation_labels']['positive']}"
            )
    print(
        "Structural profiles: "
        + ", ".join(str(profile["profile_id"]) for profile in structural_profile_selection)
    )
    for archive in selected_archives:
        print(
            f"  [{archive['kind']}] {archive['name']} "
            f"({archive['source']} {archive['variant']}, "
            f"{int(archive['size'])} bytes)"
        )

    collections = collect_features(
        args,
        search_cases,
        elf_dir,
        lib_dir,
        input_signature,
        selected_archives,
        structural_profile_selection,
    )
    if args.dry_run or args.collect_only:
        return 0

    cv_result_rows: list[dict[str, Any]] = []
    cv_summary: dict[str, Any] | None = None
    if args.validation_mode == "grouped_cv":
        (
            best_profile_id,
            params,
            train_metrics,
            validation_metrics,
            history,
            profile_rows,
            cv_result_rows,
            cv_summary,
        ) = search_all_profiles_grouped_cv(
            args,
            collections,
            development_cases,
            cv_folds,
            archive_names,
        )
    else:
        (
            best_profile_id,
            params,
            train_metrics,
            validation_metrics,
            history,
            profile_rows,
        ) = search_all_profiles(
            args,
            collections,
            train_cases,
            validation_cases,
            archive_names,
        )
    best_collection = collections[best_profile_id]
    test_metrics, predictions = evaluate(
        params, best_collection, test_cases, archive_names
    )
    metadata_by_name = {
        archive["name"]: archive for archive in selected_archives
    }
    for prediction in predictions:
        prediction["split"] = "test"
        metadata = metadata_by_name[prediction["library"]]
        prediction["candidate_kind"] = metadata["kind"]
        prediction["candidate_source"] = metadata["source"]
        prediction["candidate_variant"] = metadata["variant"]

    test_group_rows = grouped_metrics(
        params,
        best_collection,
        test_cases,
        archive_names,
    )
    test_library_rows = metrics_by_library(
        params,
        best_collection,
        test_cases,
        archive_names,
    )
    program_group_names = {case["program"] for case in test_cases}
    program_metric_rows = [
        row for row in test_group_rows if row["group"] in program_group_names
    ]
    test_macro_program_metrics = {
        metric: statistics.fmean(float(row[metric]) for row in program_metric_rows)
        for metric in (
            "lib_accuracy",
            "lib_precision",
            "lib_recall",
            "lib_specificity",
            "lib_f1",
            "cu_f1",
        )
    }

    structural = {
        key: value
        for key, value in best_collection["profile"].items()
        if key != "profile_id"
    }
    tuned_parameters = {**structural, **params}
    report = {
        "best_profile_id": best_profile_id,
        "parameters": tuned_parameters,
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_macro_program_metrics": test_macro_program_metrics,
        "available_elf_count": len(cases),
        "searched_elf_count": len(search_cases),
        "train_elf_count": len(train_cases),
        "validation_elf_count": len(validation_cases),
        "test_elf_count": len(test_cases),
        "development_elf_count": len(development_cases),
        "splits": selection_report["splits"],
        "archive_selection_split": selection_report["archive_selection_split"],
        "validation_mode": args.validation_mode,
        "cross_validation": cv_summary,
        "threshold_tuning_split": (
            "grouped_cv_train_folds_then_full_development"
            if args.validation_mode == "grouped_cv"
            else "train"
        ),
        "profile_selection_split": (
            "grouped_cv_validation_folds"
            if args.validation_mode == "grouped_cv"
            else "validation"
        ),
        "final_evaluation_split": "test",
        "test_thresholds_frozen": True,
        "test_program_count": len({case["program"] for case in test_cases}),
        "test_library_labels": split_label_counts["test"],
        "device": args.device,
        "asm_normalization": args.asm_normalization,
        "palmtree_pooling": args.palmtree_pooling,
        "library_score_aggregator": args.library_score_aggregator,
        "seed": args.seed,
        "objective_order": args.objective_order,
        "input_signature": input_signature,
        "positive_archives": positive_archives,
        "negative_archives": negative_archives,
        "pipeline": "current block/.rodata matching only",
        "ground_truth_rule": (
            "archive has at least one included compilation unit according to "
            "the randomized linker-map ground truth"
        ),
        "not_tuned": {
            "rodata_confirm_threshold": "fixed by the offline .rodata study",
            "rodata_bonus_weight": "fixed by the offline .rodata study",
            "asm_model": "model choice, not a matching threshold",
            "asm_normalization": "feature preprocessing, not a matching threshold",
            "palmtree_pooling": "embedding adapter, not a matching threshold",
            "library_score_aggregator": (
                "model-selection choice fixed for this threshold search"
            ),
            "device": "execution device, not matching behavior",
            "timeout": "execution control, not matching behavior",
        },
    }
    (args.output_dir / "best_thresholds.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "greedy_history.csv", history)
    write_csv(args.output_dir / "structural_profiles.csv", profile_rows)
    write_csv(args.output_dir / "cross_validation_results.csv", cv_result_rows)
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_csv(
        args.output_dir / "metrics_by_group.csv",
        [{"split": "test", **row} for row in test_group_rows],
    )
    write_csv(
        args.output_dir / "test_metrics_by_library.csv",
        [{"split": "test", **row} for row in test_library_rows],
    )

    print(f"\nBest structural profile: {best_profile_id}")
    print("Best parameters:")
    for name, value in tuned_parameters.items():
        print(f"  {name}={value}")
    train_label = "Development" if cv_folds else "Train"
    validation_label = "Grouped-CV pooled" if cv_folds else "Validation"
    print(
        f"{train_label} CU_F1={float(train_metrics['cu_f1']):.4f} "
        f"CU_precision={float(train_metrics['cu_precision']):.4f} "
        f"lib_F1={float(train_metrics['lib_f1']):.4f}"
    )
    print(
        f"{validation_label} CU_F1={float(validation_metrics['cu_f1']):.4f} "
        f"CU_precision={float(validation_metrics['cu_precision']):.4f} "
        f"lib_F1={float(validation_metrics['lib_f1']):.4f}"
    )
    print(
        f"Test CU_F1={float(test_metrics['cu_f1']):.4f} "
        f"CU_precision={float(test_metrics['cu_precision']):.4f} "
        f"lib_F1={float(test_metrics['lib_f1']):.4f}"
    )
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
