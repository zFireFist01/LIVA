#!/usr/bin/env python3
"""Hybrid parameter search for the randomized static-ELF dataset.

Structural block-matching parameters are sampled through expensive pipeline
runs. Decision thresholds are then optimized offline over the emitted BLOCK_CU
records. ``--thesis-protocol`` runs the two-stage Section 5.3 Optuna procedure
on a separate development dataset and rejects incomplete panels. The default
legacy workflow keeps a test split and also supports greedy coordinate search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import csv
import fcntl
import gc
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import warnings

from analysis_cache import CachedCodeUnitLoader, packaged_common_identity
from asm import CodeUnit
from match import LIBRARY_SCORE_AGGREGATORS, aggregate_library_score
from run_libseeker_batch import select_libs as select_batch_libraries
from run_libseeker_batch import stable_library_labels as batch_library_labels


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR / "Test"
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "Dataset/ground_truth/unseen"
DEFAULT_LIB_BUILD_ROOT = REPO_ROOT / "Dataset/builds/libraries"
DEFAULT_OUTPUT = TEST_DIR / "greedy_threshold_results"
DEFAULT_ANALYSIS_CACHE = REPO_ROOT / "Dataset/cache"
OPTIMIZATIONS = ("O0", "O1", "O2", "O3", "Os")
DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES = 600_000
FEATURE_SCHEMA_VERSION = 10
DEFAULT_PROGRAMS = ("grep", "less", "sed", "gawk", "nano")

# The cheap structural screen needs at least one archive that is materially
# larger and structurally different from the ncurses pair.  Keep the panel
# size fixed by replacing the redundant tinfow representative with lzma;
# libssl/libcrypto remain reserved for the full-archive finalist stage.
STRUCTURAL_SCREEN_ARCHIVE_REPLACEMENTS = (
    ("libtinfow.a", "liblzma.a"),
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
    # Warm start based on the earlier grouped-CV Optuna search.  The new
    # exact-build search requires positive function coverage as well.
    "library_min_score": 0.975,
    "block_coverage_mean_threshold": 0.975,
    "block_assignment_quality_threshold": 0.875,
    "block_min_assignment_ratio": 0.40,
    "block_min_coverage_ratio": 0.65,
    "block_min_call_edge_ratio": 0.30,
    "block_min_function_concentration": 0.75,
    "block_min_function_spread": 0.30,
    "cu_min_function_coverage": 0.05,
    "rodata_filter_enabled": 1,
    "rodata_min_bytes": 512,
    "rodata_min_strings": 16,
    "rodata_min_ngrams": 32,
    "rodata_penalty_threshold": 0.50,
    "rodata_confirm_threshold": 0.90,
    "rodata_bonus_weight": 0.30,
    "rodata_string_weight": 0.70,
    "rodata_byte_only_weight": 0.50,
    "cross_cu_call_bonus_weight": 0.20,
    "cross_cu_call_penalty_weight": 0.0,
    "cross_cu_call_saturation_edges": 8,
}

DECISION_GRIDS = {
    "library_score_aggregator": LIBRARY_SCORE_AGGREGATORS,
    "library_min_score": tuple(sorted({
        *(round(index * 0.025, 3) for index in range(41)), 0.905,
    })),
    "block_coverage_mean_threshold": tuple(
        round(0.50 + (index * 0.025), 3) for index in range(20)
    ),
    "block_assignment_quality_threshold": tuple(sorted({
        *(round(0.50 + (index * 0.025), 3) for index in range(20)), 0.999,
    })),
    "block_min_assignment_ratio": tuple(
        round(index * 0.05, 3) for index in range(21)
    ),
    "block_min_coverage_ratio": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(16)
    ),
    "block_min_call_edge_ratio": tuple(
        round(index * 0.10, 3) for index in range(11)
    ),
    "block_min_function_concentration": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(16)
    ),
    "block_min_function_spread": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(16)
    ),
    "cu_min_function_coverage": tuple(
        round(index * 0.05, 3) for index in range(1, 21)
    ),
    "rodata_min_bytes": (0, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    "rodata_min_strings": (0, 1, 2, 3, 5, 8, 16),
    "rodata_min_ngrams": (0, 16, 32, 64, 128, 256, 512),
    "rodata_penalty_threshold": (
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.35,
        0.40,
        0.50,
    ),
    "rodata_confirm_threshold": (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    "rodata_bonus_weight": (0.10, 0.20, 0.30, 0.40, 0.50),
    "rodata_string_weight": tuple(round(index * 0.05, 2) for index in range(21)),
    "rodata_byte_only_weight": tuple(round(index * 0.05, 2) for index in range(21)),
    "cross_cu_call_bonus_weight": (0.025, 0.05, 0.10, 0.15, 0.20),
    "cross_cu_call_penalty_weight": (0.00, 0.01, 0.02, 0.05, 0.10, 0.15),
    "cross_cu_call_saturation_edges": (1, 2, 3, 5, 8),
}

# A deliberately small, replay-safe neighbourhood around the best exact-build
# solution found on the first cached shard panel.  The two coverage gates are
# fixed at the values used while collecting the batch logs: lowering either
# one offline would not restore windows removed by the online prefilter.
# Values equal to 1.0 are excluded from continuous [0, 1] thresholds so the
# selected configuration does not depend on a brittle, closed upper boundary.
LOCAL_REPLAY_SAFE_DECISION_GRIDS = {
    "library_score_aggregator": ("max", "top3_noisy_or"),
    "library_min_score": (0.95, 0.975),
    "block_coverage_mean_threshold": (0.975,),
    "block_assignment_quality_threshold": (0.50, 0.525, 0.55, 0.575, 0.60),
    "block_min_assignment_ratio": (0.85, 0.90, 0.95),
    "block_min_coverage_ratio": (0.65,),
    "block_min_call_edge_ratio": (0.50, 0.60, 0.70),
    "block_min_function_concentration": (0.85, 0.90, 0.95),
    "block_min_function_spread": (0.80, 0.85, 0.90, 0.95),
    "cu_min_function_coverage": (0.80, 0.85, 0.90, 0.95),
    "rodata_min_bytes": (64, 128, 256),
    "rodata_min_strings": (0, 1, 2, 3),
    "rodata_min_ngrams": (0, 16, 32, 64),
    "rodata_penalty_threshold": (0.00, 0.05, 0.10),
    "rodata_confirm_threshold": (0.85, 0.90, 0.95),
    "rodata_bonus_weight": (0.20, 0.30, 0.40),
    "rodata_string_weight": tuple(round(index * 0.10, 2) for index in range(11)),
    "rodata_byte_only_weight": tuple(round(index * 0.10, 2) for index in range(11)),
    "cross_cu_call_bonus_weight": (0.05, 0.10, 0.15),
    "cross_cu_call_penalty_weight": (0.05, 0.10, 0.15),
    "cross_cu_call_saturation_edges": (5, 8),
}

# Restrict offline replay to a broad high-confidence region when thousands of
# negative builds are present. The collection floor remains low enough to
# retain weaker true positives while avoiding exhaustive window assignment
# below any threshold that this study can select.
LARGE_UNIVERSE_DECISION_GRIDS = {
    **DECISION_GRIDS,
    "library_score_aggregator": ("max", "top3_mean", "top3_noisy_or"),
    "library_min_score": (0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995),
    "block_coverage_mean_threshold":
        (0.85, 0.875, 0.90, 0.925, 0.95, 0.975),
    "block_min_coverage_ratio":
        (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    "block_min_assignment_ratio":
        (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    "block_min_call_edge_ratio":
        (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    "block_min_function_concentration":
        (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    "block_min_function_spread":
        (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
}

REQUIRED_DECISION_COMPONENTS = {
    "rodata_filter_enabled": 1,
    "rodata_bonus_weight_min": min(DECISION_GRIDS["rodata_bonus_weight"]),
    "cross_cu_call_bonus_weight_min": min(
        DECISION_GRIDS["cross_cu_call_bonus_weight"]
    ),
    "cu_min_function_coverage_min": min(
        DECISION_GRIDS["cu_min_function_coverage"]
    ),
}

STRUCTURAL_DEFAULTS = {
    "block_threshold": 0.90,
    "block_locality_window_multiplier": 2.5,
    "block_locality_window_padding": 6,
    "block_min_instructions": 5,
}

STRUCTURAL_GRIDS = {
    "block_threshold": (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    "block_locality_window_multiplier": (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    "block_locality_window_padding": (0, 1, 2, 3, 4, 6),
    "block_min_instructions": (1, 2, 3, 4, 5, 8),
}


def activate_decision_search_space(name: str) -> None:
    """Select the process-wide grid before signatures or trials are built."""
    if name == "full":
        return
    if name not in ("local_replay_safe", "large_universe_fast"):
        raise ValueError(f"Unknown decision search space: {name}")
    DECISION_GRIDS.clear()
    DECISION_GRIDS.update(
        LOCAL_REPLAY_SAFE_DECISION_GRIDS
        if name == "local_replay_safe"
        else LARGE_UNIVERSE_DECISION_GRIDS
    )
    # The first Optuna trial is an enqueued warm-start built from
    # ``DECISION_DEFAULTS``.  Keep that trial valid after narrowing the
    # grid: some full-search defaults (e.g. 0.875) are intentionally not
    # present in the replay-safe local grid.
    for parameter, candidates in DECISION_GRIDS.items():
        current = DECISION_DEFAULTS.get(parameter)
        if current in candidates:
            continue
        numeric_candidates = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in candidates
        )
        if numeric_candidates and isinstance(current, (int, float)):
            DECISION_DEFAULTS[parameter] = min(
                candidates, key=lambda value: abs(float(value) - float(current))
            )
        else:
            DECISION_DEFAULTS[parameter] = candidates[0]
    REQUIRED_DECISION_COMPONENTS.update({
        "rodata_bonus_weight_min": min(DECISION_GRIDS["rodata_bonus_weight"]),
        "cross_cu_call_bonus_weight_min": min(
            DECISION_GRIDS["cross_cu_call_bonus_weight"]
        ),
        "cu_min_function_coverage_min": min(
            DECISION_GRIDS["cu_min_function_coverage"]
        ),
    })

SUMMARY_RE = re.compile(
    r"^(?:YES \[W\]|YES|NO)\s+\|\s+library=(?P<library>\S+)"
)
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect block-matching features from the randomized ELF builds, "
            "then tune structural and decision parameters with restarted greedy "
            "search or a persistent Optuna study."
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
            "all gcc-* and clang-* roots under Dataset/builds/libraries."
        ),
    )
    parser.add_argument(
        "--negative-libs-dir",
        type=Path,
        help=(
            "Optional legacy directory containing versioned negative archives. "
            "By default, negative candidates are resolved from the randomized "
            "ground-truth archive universe."
        ),
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
        help="Restrict the dataset to selected programs. Repeatable.",
    )
    parser.add_argument(
        "--all-programs",
        action="store_true",
        help=(
            "Use every program found under ground-truth-dir. Without this flag "
            "the historical five-program development panel remains the default."
        ),
    )
    parser.add_argument(
        "--full-dataset-search",
        action="store_true",
        help=(
            "Use every randomized ELF selected from ground-truth-dir for "
            "grouped-CV optimization, with no internal test holdout. This "
            "implies --all-programs; use a different dataset for final testing."
        ),
    )
    parser.add_argument(
        "--thesis-protocol",
        action="store_true",
        help=(
            "Use and validate the two-stage Section 5.3 protocol: 24 structural "
            "profiles, 150 screening ELF, 720 screening builds, all 1200 "
            "development ELF, five program groups, and 3000 final Optuna "
            "trials. Requires the development ground truth and library builds."
        ),
    )
    parser.add_argument(
        "--validation-program",
        action="append",
        help=(
            "Assign this whole program to validation. Repeatable. When omitted, "
            "one program is selected deterministically from --seed."
        ),
    )
    parser.add_argument(
        "--test-program",
        action="append",
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
        default=None,
        help=(
            "Number of program-grouped development folds. Defaults to 5 for "
            "--full-dataset-search and 3 otherwise."
        ),
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
    parser.add_argument(
        "--analysis-cache-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_CACHE,
        help="Persistent radare2 + PalmTree cache used during feature collection.",
    )
    parser.add_argument(
        "--no-analysis-cache",
        action="store_true",
        help="Disable the persistent analysis/embedding cache.",
    )
    parser.add_argument(
        "--analysis-cache-only",
        action="store_true",
        help=(
            "Require a complete validated packaged cache and forbid radare2, "
            "PalmTree, and cache writes. Every selected ELF and archive is "
            "verified before feature matching starts."
        ),
    )
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--structural-trials",
        type=int,
        default=24,
        help=(
            "Number of expensive structural matching profiles to evaluate "
            "(default: 24). Each profile requires its own matching collection."
        ),
    )
    parser.add_argument(
        "--extra-high-structural-profiles",
        action="store_true",
        help=(
            "Append six high-threshold interaction profiles over 0.875-0.99 "
            "without changing existing profile IDs."
        ),
    )
    parser.add_argument(
        "--structural-profile-id",
        action="append",
        default=[],
        help=(
            "Restrict collection/search to the named generated structural "
            "profile. Repeatable; intended for distributed collect-only workers."
        ),
    )
    parser.add_argument(
        "--staged-structural-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Screen structural profiles on a positive-aware, bounded program "
            "panel, then collect the top finalists on a larger balanced "
            "positive-aware panel. "
            "Enabled by default for full-dataset Optuna searches."
        ),
    )
    parser.add_argument(
        "--structural-screen-variants-per-program",
        type=int,
        default=1,
        help=(
            "Variants retained per program during staged structural screening "
            "(default: 1). The program limit is applied afterward."
        ),
    )
    parser.add_argument(
        "--structural-screen-program-count",
        type=int,
        default=30,
        help=(
            "Maximum programs in the expensive structural screen (default: "
            "30). Programs with exact positive labels are selected first; "
            "the final tuning panel still covers all programs. 0 uses all."
        ),
    )
    parser.add_argument(
        "--structural-finalists",
        type=int,
        default=3,
        help=(
            "Number of top screened structural profiles to collect on the "
            "final tuning panel and optimize jointly (default: 3)."
        ),
    )
    parser.add_argument(
        "--structural-screen-max-archive-bytes",
        type=int,
        default=600_000,
        help=(
            "Maximum archive size used only during structural screening. "
            "The representative liblzma.a replaces libtinfow.a even when "
            "liblzma.a exceeds this limit. "
            "The finalist profiles are still collected against the complete "
            "selected archive panel. 0 disables the screen limit."
        ),
    )
    parser.add_argument(
        "--structural-screen-archive-count",
        type=int,
        default=0,
        help=(
            "Maximum representative archives in the structural screen. "
            "Retains positive and negative family coverage before extra "
            "builds; 0 keeps every archive under the size limit."
        ),
    )
    parser.add_argument(
        "--structural-screen-optuna-trials",
        type=int,
        default=None,
        help=(
            "Approximate total Optuna budget divided equally across structural "
            "profiles during screening. Defaults to at least 120 trials per "
            "profile, or 1200 total when that is larger."
        ),
    )
    parser.add_argument(
        "--staged-final-variants-per-program",
        type=int,
        default=2,
        help=(
            "Balanced variants retained per program for final threshold tuning "
            "after structural screening (default: 2)."
        ),
    )
    parser.add_argument(
        "--staged-final-program-count",
        type=int,
        default=75,
        help=(
            "Maximum positive-aware programs retained in the final staged "
            "tuning panel (default: 75, therefore at most 150 ELF with the "
            "default two variants per program). 0 uses every program."
        ),
    )
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=4,
        help=(
            "Concurrent cached ELF matching processes during feature "
            "collection (default: 4)."
        ),
    )
    parser.add_argument(
        "--fast-negative-bound",
        action="store_true",
        help=(
            "Use the provably safe global-block upper bound to skip costly "
            "window diagnostics for CUs that cannot pass the collection "
            "coverage floor."
        ),
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Greedy coordinate-search restarts for each structural profile.",
    )
    parser.add_argument(
        "--search-strategy",
        choices=("greedy", "optuna"),
        default="greedy",
        help=(
            "Offline decision search. optuna jointly selects the structural "
            "profile and every replayable decision parameter."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=1000,
        help=(
            "Total persistent Optuna trial budget, including pruned and resumed "
            "trials (default: 1000)."
        ),
    )
    parser.add_argument(
        "--optuna-workers",
        type=int,
        default=1,
        help=(
            "Parallel replay evaluators for Optuna trials. Sampling and SQLite "
            "writes stay in the parent process (default: 1)."
        ),
    )
    parser.add_argument(
        "--decision-search-space",
        choices=("full", "local_replay_safe", "large_universe_fast"),
        default="full",
        help=(
            "Decision grid to optimize. local_replay_safe fixes the two "
            "coverage prefilters at the batch-collection floors, excludes "
            "1.0 from normalized thresholds, and searches a narrow "
            "neighbourhood around the first exact-build optimum. "
            "large_universe_fast keeps high-confidence thresholds broad "
            "enough for large negative panels while raising the online "
            "window prefilter floor."
        ),
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=0,
        help="Optional Optuna search timeout in seconds; 0 disables it.",
    )
    parser.add_argument(
        "--optuna-study-name",
        help=(
            "Persistent Optuna study name. By default it is derived from the "
            "feature input signature."
        ),
    )
    parser.add_argument(
        "--optuna-storage",
        type=Path,
        help=(
            "SQLite study file. Defaults to output-dir/optuna_study.sqlite3."
        ),
    )
    parser.add_argument(
        "--train-elf-count",
        type=int,
        default=None,
        help=(
            "Maximum number of training ELF files used to tune thresholds. "
            "Defaults to 6 for the historical panel and 18 when more than five "
            "programs are selected."
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
            "Number of present archives used by the search. The same number "
            "of absent exact builds is drawn preferentially from local "
            "alternate source versions of those archives, with a ground-truth "
            "candidate fallback."
        ),
    )
    parser.add_argument(
        "--negative-lib-count",
        type=int,
        default=None,
        help=(
            "Number of absent exact-build candidates. Defaults to the positive "
            "count. Larger values first retain one near-version hard negative "
            "per positive family, then maximize distinct absent families, "
            "compilers, optimizations, and versions."
        ),
    )
    parser.add_argument(
        "--matrix-negative-builds",
        action="store_true",
        help=(
            "Draw absent exact-build candidates from the installed library "
            "matrix even when collecting new structural features."
        ),
    )
    parser.add_argument(
        "--max-matrix-negative-archive-bytes",
        type=int,
        default=0,
        help=(
            "Size cap for negative builds drawn from the installed matrix; "
            "0 disables it. Linked positives and local version hard negatives "
            "are unaffected."
        ),
    )
    parser.add_argument(
        "--include-all-tuning-positive-builds",
        action="store_true",
        help=(
            "After selecting the bounded tuning ELF panel, add every linked "
            "exact build except libc.a as a candidate. This makes a missed "
            "linked build count as FN in the tuning F1."
        ),
    )
    parser.add_argument(
        "--ground-truth-unit",
        choices=("exact_build", "library_family"),
        default="exact_build",
        help=(
            "Classification identity used for TP/TN/FP/FN. exact_build "
            "requires the archive basename, source/version, full compiler "
            "and library optimization to match (default). library_family "
            "preserves the legacy, less strict evaluation."
        ),
    )
    parser.add_argument(
        "--max-positive-archive-bytes",
        type=int,
        default=DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES,
        help=(
            "Prefer positive archives no larger than this many bytes, using "
            "the smallest available compiler/optimization build. The same "
            "limit applies to local versioned hard negatives. 0 disables "
            "the size filter. Defaults to 600000 to avoid very slow libraries "
            "such as libiconv.a during threshold tuning."
        ),
    )
    parser.add_argument(
        "--library-score-aggregator",
        choices=LIBRARY_SCORE_AGGREGATORS,
        default="top3_noisy_or",
        help=(
            "Initial CU-to-library score aggregator. All supported aggregators "
            "are then compared offline; feature collections are "
            "aggregator-independent."
        ),
    )
    parser.add_argument(
        "--objective-order",
        choices=("cu_first", "lib_first", "lib_only"),
        default="lib_first",
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
            "optuna_threshold_search output directory. Repeatable. This lets "
            "a new library selection reuse already collected profile/ELF/"
            "library records and compute only missing pairs."
        ),
    )
    parser.add_argument(
        "--import-batch-reports-dir",
        type=Path,
        help=(
            "Decision-only mode: import replay-complete .report.txt and "
            ".features.jsonl.gz files produced by run_libseeker_batch instead "
            "of running matching. Requires --search-only and a single fixed "
            "structural profile."
        ),
    )
    parser.add_argument(
        "--batch-library-matrix",
        type=Path,
        default=REPO_ROOT / "Dataset/manifests/library_matrix.tsv",
        help="Library matrix used to recover exact-build aliases in batch logs.",
    )
    parser.add_argument(
        "--batch-library-root",
        type=Path,
        default=DEFAULT_LIB_BUILD_ROOT,
        help="Build root corresponding to --batch-library-matrix.",
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
    parser.add_argument(
        "--keep-collection-artifacts",
        action="store_true",
        help=(
            "Keep the uncompressed per-profile reports, logs, and JSONL files. "
            "By default they are removed only after the compact feature archive "
            "has been written and its gzip integrity has been verified."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def archive_family(name: str) -> str:
    lowered = Path(name).name.lower()
    # Exact-build candidates are exposed to main.py as unique aliases such as
    # ``libz.a.0123abcd...``.  Recover the real archive basename before
    # deriving its family; the suffix is identity metadata, not a filename.
    archive_suffix = lowered.find(".a.")
    if archive_suffix >= 0:
        lowered = lowered[: archive_suffix + 2]
    # An archive family is its logical archive name, not the source package.
    # For example, libc and libm must remain distinct even though both are
    # built by glibc; likewise libpcre2-8 and libpcre2-posix are distinct.
    return lowered.removesuffix(".a")


BuildIdentity = tuple[str, str, str, str]


def archive_build_identity(archive: dict[str, Any]) -> BuildIdentity:
    """Return the exact build identity requested by the evaluation protocol."""
    return (
        str(archive.get("name", "")),
        str(archive.get("source", "")),
        str(archive.get("compiler", archive.get("compiler_root", ""))),
        str(archive.get("optimization", "")),
    )


def exact_build_label(identity: BuildIdentity) -> str:
    """Create a stable main.py-safe alias for one exact archive build."""
    archive_name = identity[0]
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{archive_name}.{digest}"


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


def confirmed_cus_for_archive(archive: dict[str, Any]) -> set[str] | None:
    """Return exact linker-map CU labels for one archive, when available."""
    confirmed_members = archive.get("confirmed_compilation_units")
    if confirmed_members is not None:
        if archive.get("confirmed_compilation_units_method") != "linker_map":
            return None
        return {
            normalize_cu_name(str(member))
            for member in confirmed_members
        }

    compilation_units = archive.get("compilation_units")
    if compilation_units is None:
        return None
    return {
        normalize_cu_name(str(unit["compilation_unit"]))
        for unit in compilation_units
        if unit.get("included")
        and unit.get("ground_truth_method") == "linker_map"
    }


def archive_is_present_in_ground_truth(archive: dict[str, Any]) -> bool:
    if "included_compilation_units" not in archive:
        return True
    return int(archive.get("included_compilation_units", 0) or 0) > 0


def resolve_ground_truth_archive_path(
    value: str | Path,
    artifact_root: Path,
) -> Path | None:
    """Resolve portable and container-recorded archive paths locally."""
    declared = Path(str(value)).expanduser()
    if declared.is_file():
        return declared.resolve()
    if not declared.is_absolute():
        for root in (artifact_root, REPO_ROOT / "Dataset"):
            candidate = root / declared
            if candidate.is_file():
                return candidate.resolve()
        return None

    # Reproduction metadata records /workspace/.../Dataset paths.  Preserve
    # the suffix below Dataset so the same ground truth works in any checkout.
    parts = declared.parts
    try:
        dataset_index = parts.index("Dataset")
    except ValueError:
        return None
    suffix = parts[dataset_index + 1 :]
    for root in (artifact_root, REPO_ROOT / "Dataset"):
        candidate = root.joinpath(*suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_lib_roots(args: argparse.Namespace) -> list[Path]:
    if args.lib_root:
        roots = [path.resolve() for path in args.lib_root]
    else:
        build_root = (
            args.batch_library_root
            if getattr(args, "thesis_protocol", False)
            else DEFAULT_LIB_BUILD_ROOT
        )
        roots = (
            [
                path.resolve()
                for path in sorted(build_root.iterdir())
                if path.is_dir()
                and (
                    path.name.startswith("gcc-")
                    or path.name.startswith("clang-")
                )
            ]
            if build_root.is_dir()
            else []
        )

    if not roots:
        raise FileNotFoundError("No compiler-specific library roots found")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Library root not found: {root}")
    return roots


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_programs and args.program:
        raise ValueError("--all-programs cannot be combined with --program")

    ground_truth_records = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.ground_truth_dir.rglob("ground_truth.json"))
    ]
    available_programs = {
        str(data.get("program", ""))
        for _, data in ground_truth_records
        if data.get("program")
    }
    requested_variants = set(args.elf_variants)
    if args.all_programs or (requested_variants and not args.program):
        programs = set(available_programs)
    elif args.program:
        programs = set(args.program)
    else:
        programs = set(DEFAULT_PROGRAMS)
    programs.update(args.validation_program or [])
    programs.update(args.test_program or [])
    unknown_programs = programs - available_programs
    if unknown_programs and (
        args.program or args.validation_program or args.test_program
    ):
        raise ValueError(
            "Requested programs not present in ground truth: "
            + ", ".join(sorted(unknown_programs))
        )
    compilers = set(args.compiler or ("gcc", "clang"))
    cases = []

    artifact_root = args.ground_truth_dir.parent.parent
    for _path, data in ground_truth_records:
        program = str(data.get("program", ""))
        if program not in programs:
            continue
        if not data.get("elf_optimization"):
            continue
        if requested_variants and data.get("variant") not in requested_variants:
            continue
        compiler = str(data.get("compiler", "")).split("-", maxsplit=1)[0]
        if compiler not in compilers:
            continue
        declared_binary = Path(str(data["binary"]))
        binary = (
            declared_binary
            if declared_binary.is_absolute()
            else artifact_root / declared_binary
        ).resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"Ground-truth ELF not found: {binary}")
        randomized_libraries = {
            str(library)
            for library in data.get("libs", [])
        } or {
            str(library)
            for library in data.get("seeded_library_selection", {})
        }
        candidate_archives = []
        for archive in data.get("archives", []):
            library = str(archive.get("library", ""))
            if not library or library not in randomized_libraries:
                continue
            resolved_archive = resolve_ground_truth_archive_path(
                str(archive.get("archive", "")), artifact_root
            )
            candidate_archives.append(
                {
                    "name": Path(str(archive.get("archive", ""))).name,
                    "path": str(resolved_archive) if resolved_archive else "",
                    "present": archive_is_present_in_ground_truth(archive),
                    "library": library,
                    "source": str(archive.get("source") or library),
                    "optimization": str(archive.get("optimization", "")),
                    "compiler": str(archive.get("compiler", "")),
                    "size": (
                        resolved_archive.stat().st_size
                        if resolved_archive is not None
                        else int(archive.get("archive_size", 0) or 0)
                    ),
                    "sha256": str(archive.get("archive_sha256", "")),
                    "confirmed_cus": confirmed_cus_for_archive(archive),
                }
            )
        present_candidate_archives = [
            archive
            for archive in candidate_archives
            if archive["present"]
        ]
        expected = {
            archive_family(str(archive["name"]))
            for archive in present_candidate_archives
        }
        archive_sources: dict[str, set[str]] = {}
        for archive in present_candidate_archives:
            archive_sources.setdefault(str(archive["name"]), set()).add(
                str(archive["source"])
            )
        expected_cus = included_cus_from_ground_truth(data)
        expected_cus_by_build = {
            archive_build_identity(archive): set(archive["confirmed_cus"])
            for archive in present_candidate_archives
            if archive.get("confirmed_cus") is not None
        }
        cases.append(
            {
                "variant": str(data["variant"]),
                "program": program,
                "compiler": compiler,
                "compiler_config": str(data["compiler"]),
                "elf_optimization": str(data["elf_optimization"]),
                "binary": binary,
                "binary_sha256": str(data.get("binary_sha256", "")),
                "binary_size": int(data.get("binary_size", 0) or 0),
                "expected": expected,
                "expected_builds": {
                    archive_build_identity(archive)
                    for archive in present_candidate_archives
                },
                "expected_cus": expected_cus,
                "expected_cus_by_build": expected_cus_by_build,
                "archive_names": {
                    str(archive["name"])
                    for archive in present_candidate_archives
                },
                "archive_sources": archive_sources,
                "candidate_archives": candidate_archives,
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


def complete_batch_feature_cases(
    reports_dir: Path,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only cases whose replay batch report and gzip feature are final."""
    selected = []
    for case in cases:
        reports = sorted(reports_dir.glob(f"{case['variant']}_*.report.txt"))
        if len(reports) != 1:
            continue
        report = reports[0]
        feature = report.with_name(
            report.name.replace(".report.txt", ".features.jsonl.gz")
        )
        if not feature.is_file():
            continue
        try:
            with report.open("rb") as handle:
                handle.seek(max(0, report.stat().st_size - 16_384))
                tail = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        if "Done processing in " not in tail:
            continue
        selected.append(case)
    if not selected:
        raise ValueError(
            f"No complete replay batch features found in {reports_dir}"
        )
    return selected


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


def ground_truth_archive_pool(
    cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return locally resolvable randomized candidate archives by basename."""
    pool: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        for record in case.get("candidate_archives", []):
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                continue
            identity = (
                str(record.get("name", "")),
                str(path.resolve()),
                str(record.get("sha256", "")),
            )
            if not identity[0] or identity in seen:
                continue
            seen.add(identity)
            normalized = dict(record)
            normalized["path"] = str(path.resolve())
            normalized["size"] = path.stat().st_size
            pool.setdefault(identity[0], []).append(normalized)
    return pool


def exact_ground_truth_archive_pool(
    cases: list[dict[str, Any]],
) -> dict[BuildIdentity, list[dict[str, Any]]]:
    """Return locally resolvable archives grouped by exact build identity."""
    pool: dict[BuildIdentity, list[dict[str, Any]]] = {}
    seen: set[tuple[BuildIdentity, str, str]] = set()
    for case in cases:
        for record in case.get("candidate_archives", []):
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                continue
            identity = archive_build_identity(record)
            if not all(identity):
                continue
            occurrence = (
                identity,
                str(path.resolve()),
                str(record.get("sha256", "")),
            )
            if occurrence in seen:
                continue
            seen.add(occurrence)
            normalized = dict(record)
            normalized["path"] = str(path.resolve())
            normalized["size"] = path.stat().st_size
            pool.setdefault(identity, []).append(normalized)

    return pool


def unambiguous_exact_archive_pool(
    pool: dict[BuildIdentity, list[dict[str, Any]]],
) -> dict[BuildIdentity, list[dict[str, Any]]]:
    """Keep only identities with one pinned SHA-256 across the dataset."""
    return {
        identity: records
        for identity, records in pool.items()
        if len({str(record.get("sha256", "")) for record in records}) == 1
        and all(record.get("sha256") for record in records)
    }


def ambiguous_exact_builds(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe tuple labels that map to more than one archive content hash."""
    pool = exact_ground_truth_archive_pool(cases)
    return [
        {
            "build_identity": list(identity),
            "sha256": sorted(
                {str(record.get("sha256", "")) for record in records}
            ),
        }
        for identity, records in sorted(pool.items())
        if len({str(record.get("sha256", "")) for record in records}) > 1
    ]


def selected_archive_record(
    name: str,
    records: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    """Choose the smallest reproducible build for one archive basename."""
    record = min(
        records,
        key=lambda item: (
            int(item.get("size", 0)),
            str(item.get("compiler", "")),
            str(item.get("optimization", "")),
            str(item.get("path", "")),
        ),
    )
    compiler = str(record.get("compiler", ""))
    optimization = str(record.get("optimization", ""))
    return {
        "name": name,
        "path": str(record["path"]),
        "kind": kind,
        "source": str(record.get("source", record.get("library", "ground_truth"))),
        "library": str(record.get("library", "")),
        "variant": "/".join(value for value in (compiler, optimization) if value),
        "compiler_root": compiler,
        "optimization": optimization,
        "size": int(record["size"]),
        "sha256": str(record.get("sha256", "")),
    }


def selected_exact_archive_record(
    identity: BuildIdentity,
    records: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    """Select a reproducible file and expose it under an exact-build alias."""
    record = min(
        records,
        key=lambda item: (
            int(item.get("size", 0)),
            str(item.get("path", "")),
        ),
    )
    archive_name, source, compiler, optimization = identity
    return {
        "name": exact_build_label(identity),
        "archive_name": archive_name,
        "path": str(record["path"]),
        "kind": kind,
        "source": source,
        "library": str(record.get("library", "")),
        "variant": f"{source}/{compiler}/{optimization}",
        "compiler_root": compiler,
        "compiler": compiler,
        "optimization": optimization,
        "build_identity": list(identity),
        "provenance": "linker_map_ground_truth",
        "size": int(record["size"]),
        "sha256": str(record.get("sha256", "")),
    }


def verify_selected_archive_digests(
    selected_archives: list[dict[str, Any]],
) -> None:
    """Reject archives that do not match their pinned selected-file digest."""
    for archive in selected_archives:
        expected_digest = str(archive.get("sha256", ""))
        if not expected_digest:
            raise ValueError(
                f"Exact build {archive['name']} has no selected-file SHA-256"
            )
        path = Path(str(archive["path"]))
        with path.open("rb") as handle:
            actual_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"Exact build archive differs from its pinned content: {path} "
                f"({actual_digest} != {expected_digest})"
            )


def validated_cache_inventory(cache_dir: Path) -> dict[str, Any]:
    """Require the completeness contract emitted by a packaged shard."""
    inventory_path = cache_dir / "shard_inventory.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "--analysis-cache-only requires cache/shard_inventory.json"
        ) from error
    complete = (
        inventory.get("valid") is True
        and inventory.get("errors") == []
        and inventory.get("full_library_catalog") is True
        and inventory.get("experiment_cache_records")
        == inventory.get("elf_records")
        and inventory.get("library_archives_cached")
        == inventory.get("selected_library_archives_total")
    )
    if not complete:
        raise ValueError(
            "--analysis-cache-only requires a complete, validated shard "
            f"inventory: {inventory_path}"
        )
    return inventory


def validate_cache_only_inputs(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prove every matching input is cached without invoking analysis."""
    if not args.analysis_cache_only:
        return None

    inventory = validated_cache_inventory(args.analysis_cache_dir)
    loader = CachedCodeUnitLoader(
        SCRIPT_DIR / "palmtree/model/transformer.ep19",
        device=args.device,
        pooling=args.palmtree_pooling,
        asm_normalization=args.asm_normalization,
        cache_dir=args.analysis_cache_dir,
        write_cache=False,
        cache_only=True,
    )
    if loader.cache is None:  # pragma: no cover - constructor guarantees it
        raise RuntimeError("Cache-only loader was created without a cache")

    unique_cases = {
        str(Path(case["binary"]).resolve()): case for case in cases
    }
    failures: list[str] = []
    for path_text, case in sorted(unique_cases.items()):
        path = Path(path_text)
        identity = loader.identity(path, CodeUnit.TYPE_ELF)
        declared_size = int(case.get("binary_size", 0) or 0)
        declared_digest = str(case.get("binary_sha256", ""))
        if not declared_size or not declared_digest:
            failures.append(f"ELF provenance incomplete: {case['variant']}")
            continue
        if identity["binary_size"] != declared_size:
            failures.append(f"ELF size mismatch: {case['variant']}")
            continue
        if identity["binary_sha256"] != declared_digest:
            failures.append(f"ELF SHA-256 mismatch: {case['variant']}")
            continue
        if not loader.cache.contains(identity):
            failures.append(f"ELF cache miss: {case['variant']}")

    for archive in selected_archives:
        path = Path(str(archive["path"]))
        if loader.inspect_archive_index(path) is None:
            failures.append(f"archive/member cache miss: {archive['name']}")

    if failures:
        details = "; ".join(failures[:12])
        if len(failures) > 12:
            details += f"; ... and {len(failures) - 12} more"
        raise ValueError(
            "Cache-only preflight rejected the selected inputs; no matching "
            f"was started: {details}"
        )

    common_identity = packaged_common_identity(args.analysis_cache_dir)
    return {
        "enabled": True,
        "cache_dir": str(args.analysis_cache_dir),
        "inventory": inventory,
        "common_identity_sha256": selection_signature(common_identity),
        "verified_elf_count": len(unique_cases),
        "verified_archive_count": len(selected_archives),
    }


def library_build_diversity_report(
    archives: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose version, compiler and optimization coverage per real archive."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for archive in archives:
        grouped.setdefault(str(archive["archive_name"]), []).append(archive)
    return {
        name: {
            "positive_source_versions": sorted(
                {str(item["source"]) for item in items if item["kind"] == "positive"}
            ),
            "negative_source_versions": sorted(
                {str(item["source"]) for item in items if item["kind"] == "negative"}
            ),
            "all_source_versions": sorted(
                {str(item["source"]) for item in items}
            ),
            "compilers": sorted({str(item["compiler"]) for item in items}),
            "optimizations": sorted(
                {str(item["optimization"]) for item in items}
            ),
        }
        for name, items in sorted(grouped.items())
    }


def require_library_build_diversity(
    diversity: dict[str, dict[str, Any]],
) -> None:
    """Fail an exact-build Optuna panel that lacks component diversity."""
    for archive_name, values in diversity.items():
        # Negative-only families intentionally contribute breadth: requiring
        # two candidates for every one would halve the number of unrelated
        # libraries represented by a fixed negative budget. Positive families
        # remain strictly diverse in every exact-build component.
        if not values["positive_source_versions"]:
            continue
        for component, label in (
            ("all_source_versions", "source versions"),
            ("compilers", "compiler versions"),
            ("optimizations", "optimization levels"),
        ):
            if len(values[component]) < 2:
                raise ValueError(
                    f"Exact-build Optuna panel has fewer than two {label} "
                    f"for {archive_name}. Provide alternate local builds or "
                    "select another archive panel."
                )


def select_exact_ground_truth_positive_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = unambiguous_exact_archive_pool(
        exact_ground_truth_archive_pool(cases)
    )
    present = {
        identity
        for case in cases
        for identity in case.get("expected_builds", set())
        if identity in pool and (args.include_glibc or identity[0] != "libc.a")
    }
    eligible = [
        identity
        for identity in present
        if args.max_positive_archive_bytes <= 0
        or min(int(record["size"]) for record in pool[identity])
        <= args.max_positive_archive_bytes
    ]
    eligible.sort(
        key=lambda identity: (
            min(int(record["size"]) for record in pool[identity]),
            identity,
        )
    )

    prevalence = {
        identity: sum(
            identity in case.get("expected_builds", set()) for case in cases
        )
        for identity in eligible
    }

    # Exact-build Optuna must learn that a close version is still a different
    # class.  Use two genuinely linked source versions of each selected archive
    # before adding more families.  Besides producing better hard positives,
    # this keeps the expensive panel compact: six positives normally become
    # three archive families x two versions rather than six unrelated archives.
    if (
        getattr(args, "search_strategy", "") == "optuna"
        and args.positive_lib_count >= 2
    ):
        identities_by_archive: dict[str, list[BuildIdentity]] = {}
        for identity in eligible:
            identities_by_archive.setdefault(identity[0], []).append(identity)
        # Each chosen family must admit two positive builds that differ in all
        # three exact-build components.  Merely having two source versions is
        # insufficient for larger panels (for example, it could leave the
        # compiler or optimization constant and fail the diversity contract).
        diverse_pairs_by_archive: dict[
            str, list[tuple[BuildIdentity, BuildIdentity]]
        ] = {}
        for archive_name, identities in identities_by_archive.items():
            pairs = [
                (left, right)
                for index, left in enumerate(identities)
                for right in identities[index + 1:]
                if left[1] != right[1]
                and left[2] != right[2]
                and left[3] != right[3]
            ]
            if pairs:
                diverse_pairs_by_archive[archive_name] = pairs
        archive_count = min(
            len(diverse_pairs_by_archive),
            max(1, args.positive_lib_count // 2),
        )
        ranked_archives = sorted(
            diverse_pairs_by_archive,
            key=lambda archive_name: (
                -max(
                    prevalence[left] + prevalence[right]
                    for left, right in diverse_pairs_by_archive[archive_name]
                ),
                min(
                    int(record["size"])
                    for identity in identities_by_archive[archive_name]
                    for record in pool[identity]
                ),
                archive_name,
            ),
        )[:archive_count]

        selected: list[BuildIdentity] = []
        covered_compilers: set[str] = set()
        covered_optimizations: set[str] = set()
        for archive_name in ranked_archives:
            pair = max(
                diverse_pairs_by_archive[archive_name],
                key=lambda values: (
                    len({values[0][2], values[1][2]} - covered_compilers),
                    len({values[0][3], values[1][3]} - covered_optimizations),
                    prevalence[values[0]] + prevalence[values[1]],
                    -sum(
                        min(int(record["size"]) for record in pool[identity])
                        for identity in values
                    ),
                    values,
                ),
            )
            selected.extend(pair)
            covered_compilers.update(identity[2] for identity in pair)
            covered_optimizations.update(identity[3] for identity in pair)

        # An odd requested count gets one additional build after every family
        # has received its component-diverse pair.
        if len(selected) < args.positive_lib_count:
            remaining = [
                identity
                for archive_name in ranked_archives
                for identity in identities_by_archive[archive_name]
                if identity not in selected
            ]
            if remaining:
                selected.append(max(remaining, key=lambda item: prevalence[item]))

        if len(selected) == args.positive_lib_count:
            return sorted(
                (
                    selected_exact_archive_record(
                        identity, pool[identity], "positive"
                    )
                    for identity in selected
                ),
                key=lambda item: str(item["name"]),
            )

    # Cover source/version and archive names first.  Once every archive is
    # represented, use the remaining slots to cover compiler and optimization
    # values before prevalence, rather than repeatedly selecting near-identical
    # positive builds just because they occur in a few more ELFs.
    selected: list[BuildIdentity] = []
    covered_sources: set[str] = set()
    covered_archives: set[str] = set()
    covered_compilers: set[str] = set()
    covered_optimizations: set[str] = set()
    eligible_archives = {identity[0] for identity in eligible}
    while len(selected) < args.positive_lib_count:
        remaining = [identity for identity in eligible if identity not in selected]
        if not remaining:
            break
        archive_coverage_complete = covered_archives == eligible_archives
        chosen = max(
            remaining,
            key=lambda identity: (
                identity[1] not in covered_sources,
                identity[0] not in covered_archives,
                archive_coverage_complete
                and identity[3] not in covered_optimizations,
                archive_coverage_complete
                and identity[2] not in covered_compilers,
                prevalence[identity],
                -min(int(record["size"]) for record in pool[identity]),
                identity,
            ),
        )
        selected.append(chosen)
        covered_sources.add(chosen[1])
        covered_archives.add(chosen[0])
        covered_compilers.add(chosen[2])
        covered_optimizations.add(chosen[3])
    if len(selected) < args.positive_lib_count:
        print(
            f"[WARN] Requested {args.positive_lib_count} positive exact builds, "
            f"but only {len(selected)} resolvable builds satisfy the size limit"
        )
    return sorted(
        (
            selected_exact_archive_record(identity, pool[identity], "positive")
            for identity in selected
        ),
        key=lambda item: str(item["name"]),
    )


def select_ground_truth_positive_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.ground_truth_unit == "exact_build":
        return select_exact_ground_truth_positive_archives(args, cases)
    pool = ground_truth_archive_pool(cases)
    positive_names = {
        str(record["name"])
        for case in cases
        for record in case.get("candidate_archives", [])
        if record.get("present") and record.get("name") in pool
    }
    if not args.include_glibc:
        positive_names.discard("libc.a")

    eligible = []
    skipped = []
    for name in positive_names:
        minimum_size = min(int(record["size"]) for record in pool[name])
        item = (minimum_size, name)
        if (
            args.max_positive_archive_bytes > 0
            and minimum_size > args.max_positive_archive_bytes
        ):
            skipped.append(item)
        else:
            eligible.append(item)
    eligible.sort()
    skipped.sort()
    for size, name in skipped:
        print(
            f"[positive-skip] {name} size={size} exceeds "
            f"--max-positive-archive-bytes={args.max_positive_archive_bytes}"
        )

    # First cover distinct randomized library families, then fill by size.
    selected_names: list[str] = []
    selected_libraries: set[str] = set()
    for _, name in eligible:
        libraries = {str(record.get("library", "")) for record in pool[name]}
        if libraries - selected_libraries:
            selected_names.append(name)
            selected_libraries.update(libraries)
            if len(selected_names) == args.positive_lib_count:
                break
    if len(selected_names) < args.positive_lib_count:
        selected_names.extend(
            name
            for _, name in eligible
            if name not in selected_names
        )
        selected_names = selected_names[: args.positive_lib_count]

    if len(selected_names) < args.positive_lib_count:
        print(
            f"[WARN] Requested {args.positive_lib_count} positive archives, "
            f"but only {len(selected_names)} resolvable archives satisfy the size limit"
        )
    return sorted(
        (
            selected_archive_record(name, pool[name], "positive")
            for name in selected_names
        ),
        key=lambda item: str(item["name"]),
    )


def select_positive_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, str]]:
    if args.positive_lib_count < 1:
        raise ValueError("--positive-lib-count must be at least 1")

    if ground_truth_archive_pool(cases):
        return select_ground_truth_positive_archives(args, cases)

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


def select_legacy_negative_archives(
    args: argparse.Namespace,
    positive_archives: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    if args.negative_libs_dir is None:
        return []
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


def local_versioned_negative_archive_pool(
    args: argparse.Namespace,
    positive_archives: list[dict[str, Any]],
    present_identities: set[BuildIdentity],
) -> dict[BuildIdentity, list[dict[str, Any]]]:
    """Find alternate source versions of the same archive in local builds."""
    positive_sources_by_archive: dict[str, set[str]] = {}
    for archive in positive_archives:
        positive_sources_by_archive.setdefault(
            str(archive["archive_name"]), set()
        ).add(str(archive["source"]))

    pool: dict[BuildIdentity, list[dict[str, Any]]] = {}
    for root in args.lib_roots:
        for source_dir in sorted(root.iterdir()):
            if not source_dir.is_dir():
                continue
            for archive_name, linked_sources in positive_sources_by_archive.items():
                if source_dir.name in linked_sources:
                    continue
                for optimization in OPTIMIZATIONS:
                    path = source_dir / optimization / "install/lib" / archive_name
                    if not path.is_file():
                        continue
                    size = path.stat().st_size
                    if size < 4096 or (
                        args.max_positive_archive_bytes > 0
                        and size > args.max_positive_archive_bytes
                    ):
                        continue
                    identity = (
                        archive_name,
                        source_dir.name,
                        root.name,
                        optimization,
                    )
                    if identity in present_identities:
                        continue
                    pool.setdefault(identity, []).append(
                        {
                            "name": archive_name,
                            "path": str(path.resolve()),
                            "source": source_dir.name,
                            "compiler": root.name,
                            "optimization": optimization,
                            "size": size,
                        }
                    )
    return pool


def source_version_order(source: str) -> tuple[int, ...]:
    """Prefer nearby, newer historical versions for hard negatives."""
    match = re.search(r"-(\d[0-9.]*)", source)
    return (
        tuple(int(part) for part in re.findall(r"\d+", match.group(1)))
        if match else ()
    )


def batch_matrix_absent_archive_pool(
    args: argparse.Namespace,
    present_identities: set[BuildIdentity],
) -> dict[BuildIdentity, list[dict[str, Any]]]:
    """Return cached matrix builds that are never linked in the source pool."""
    if (
        getattr(args, "import_batch_reports_dir", None) is None
        and not getattr(args, "matrix_negative_builds", False)
    ):
        return {}
    batch_args = SimpleNamespace(
        library_matrix=args.batch_library_matrix,
        library_root=args.batch_library_root,
        library_metadata={},
    )
    root = Path(args.batch_library_root).resolve()
    pool: dict[BuildIdentity, list[dict[str, Any]]] = {}
    for path in select_batch_libraries(batch_args):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) < 6:
            continue
        compiler, source, optimization = parts[0], parts[1], parts[2]
        archive_name = resolved.name
        identity = (archive_name, source, compiler, optimization)
        if (
            identity in present_identities
            or archive_name == "libc.a"
            or resolved.stat().st_size < 4096
            or (
                getattr(args, "max_matrix_negative_archive_bytes", 0) > 0
                and resolved.stat().st_size
                > args.max_matrix_negative_archive_bytes
            )
            or any(
                token in archive_name
                for token in ("_pic.a", "_g.a", "nonshared", "BrokenLocale")
            )
        ):
            continue
        pool.setdefault(identity, []).append({
            "name": archive_name,
            "path": str(resolved),
            "source": source,
            "compiler": compiler,
            "optimization": optimization,
            "size": resolved.stat().st_size,
        })
    return pool


def select_negative_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    positive_archives: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select absent candidates from ground truth, with a legacy fallback."""
    requested_count = (
        args.negative_lib_count
        if getattr(args, "negative_lib_count", None) is not None
        else len(positive_archives)
    )
    if requested_count < 1:
        raise ValueError("--negative-lib-count must be at least 1")
    if args.ground_truth_unit == "exact_build":
        pool = unambiguous_exact_archive_pool(
            exact_ground_truth_archive_pool(cases)
        )
        positive_identities = {
            tuple(item["build_identity"])
            for item in positive_archives
        }
        present_identities = {
            identity
            for case in cases
            for identity in case.get("expected_builds", set())
        }
        preferred_order = {
            name: index for index, name in enumerate(PREFERRED_NEGATIVE_ARCHIVES)
        }
        positive_archives_names = {identity[0] for identity in positive_identities}
        positive_sources = {identity[1] for identity in positive_identities}
        local_pool = local_versioned_negative_archive_pool(
            args, positive_archives, present_identities
        )
        archive_sources: dict[str, set[str]] = {}
        archive_compilers: dict[str, set[str]] = {}
        archive_optimizations: dict[str, set[str]] = {}
        for identity in positive_identities:
            archive_sources.setdefault(identity[0], set()).add(identity[1])
            archive_compilers.setdefault(identity[0], set()).add(identity[2])
            archive_optimizations.setdefault(identity[0], set()).add(identity[3])

        local_selected: list[BuildIdentity] = []
        local_source_coverage: set[str] = set()
        # Give every selected archive its closest available alternate source
        # version first; a different compiler/optimization of the linked
        # version would not test version-sensitive false positives.
        for archive_name in sorted(positive_archives_names):
            candidates = [
                identity for identity in local_pool
                if identity[0] == archive_name
            ]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda identity: (
                    source_version_order(identity[1]),
                    identity[3] not in archive_optimizations.get(
                        archive_name, set()
                    ),
                    identity[2] not in archive_compilers.get(
                        archive_name, set()
                    ),
                    -min(int(record["size"]) for record in local_pool[identity]),
                    identity,
                ),
            )
            local_selected.append(chosen)
            local_source_coverage.add(chosen[1])
            archive_sources.setdefault(chosen[0], set()).add(chosen[1])
            archive_compilers.setdefault(chosen[0], set()).add(chosen[2])
            archive_optimizations.setdefault(chosen[0], set()).add(chosen[3])

        local_selected = local_selected[:requested_count]

        selected = []
        for identity in local_selected:
            record = min(
                local_pool[identity],
                key=lambda item: (int(item["size"]), str(item["path"])),
            )
            with Path(str(record["path"])).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            selected.append(
                {
                    **selected_exact_archive_record(
                        identity, [{**record, "sha256": digest}], "negative"
                    ),
                    "provenance": "local_versioned_build_absent_from_linker_map",
                }
            )

        # Broaden the negative universe with cached libraries that are not
        # linked by any ELF in the source pool. Select distinct archive names
        # first, while rotating compiler, optimization, and source versions.
        matrix_pool = batch_matrix_absent_archive_pool(args, present_identities)
        matrix_identities = [
            identity
            for identity in matrix_pool
            if identity not in local_selected
            and identity not in positive_identities
        ]
        covered_negative_names = {item[0] for item in local_selected}
        covered_negative_sources = {item[1] for item in local_selected}
        covered_negative_compilers = {item[2] for item in local_selected}
        covered_negative_optimizations = {item[3] for item in local_selected}
        while len(selected) < requested_count and matrix_identities:
            identity = max(
                matrix_identities,
                key=lambda item: (
                    item[0] not in covered_negative_names,
                    item[2] not in covered_negative_compilers,
                    item[3] not in covered_negative_optimizations,
                    item[1] not in covered_negative_sources,
                    -min(int(record["size"]) for record in matrix_pool[item]),
                    item,
                ),
            )
            matrix_identities.remove(identity)
            record = min(
                matrix_pool[identity],
                key=lambda item: (int(item["size"]), str(item["path"])),
            )
            with Path(str(record["path"])).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            selected.append({
                **selected_exact_archive_record(
                    identity, [{**record, "sha256": digest}], "negative"
                ),
                "provenance": "cached_build_absent_from_source_pool",
            })
            covered_negative_names.add(identity[0])
            covered_negative_sources.add(identity[1])
            covered_negative_compilers.add(identity[2])
            covered_negative_optimizations.add(identity[3])

        absent_identities = [
            identity
            for identity in pool
            if identity not in present_identities
            and identity not in positive_identities
            and identity[0] != "libc.a"
            and identity not in local_selected
        ]
        while len(selected) < requested_count and absent_identities:
            identity = max(
                absent_identities,
                key=lambda item: (
                    item[0] not in covered_negative_names,
                    item[2] not in covered_negative_compilers,
                    item[3] not in covered_negative_optimizations,
                    item[1] not in covered_negative_sources,
                    item[0] not in positive_archives_names,
                    -preferred_order.get(item[0], len(preferred_order)),
                    -min(int(record["size"]) for record in pool[item]),
                    item,
                ),
            )
            absent_identities.remove(identity)
            selected.append(
                selected_exact_archive_record(identity, pool[identity], "negative")
            )
            covered_negative_names.add(identity[0])
            covered_negative_sources.add(identity[1])
            covered_negative_compilers.add(identity[2])
            covered_negative_optimizations.add(identity[3])
        if len(selected) < requested_count:
            raise ValueError(
                f"Need {requested_count} absent exact builds, found "
                f"{len(selected)} in the ground-truth candidate universe"
            )
        return sorted(selected, key=lambda item: str(item["name"]))

    count = len(positive_archives)
    positive_names = {str(item["name"]) for item in positive_archives}
    pool = ground_truth_archive_pool(cases)
    present_names = {
        str(record["name"])
        for case in cases
        for record in case.get("candidate_archives", [])
        if record.get("present")
    }
    absent_names = [
        name
        for name in pool
        if name not in present_names
        and name not in positive_names
        and name != "libc.a"
    ]
    preferred_order = {
        name: index for index, name in enumerate(PREFERRED_NEGATIVE_ARCHIVES)
    }
    absent_names.sort(
        key=lambda name: (
            preferred_order.get(name, len(preferred_order)),
            min(int(record["size"]) for record in pool[name]),
            name,
        )
    )
    selected = [
        selected_archive_record(name, pool[name], "negative")
        for name in absent_names[:count]
    ]

    if len(selected) < count and args.negative_libs_dir is not None:
        legacy = select_legacy_negative_archives(args, positive_archives, rng)
        already_selected = positive_names | {str(item["name"]) for item in selected}
        selected.extend(
            item
            for item in legacy
            if str(item["name"]) not in already_selected
        )
        selected = selected[:count]

    if len(selected) < count:
        raise ValueError(
            f"Need {count} negative archives, found {len(selected)} in the "
            "ground-truth candidate universe. Select more programs or provide "
            "--negative-libs-dir."
        )
    return sorted(selected, key=lambda item: str(item["name"]))


def selection_signature(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def matching_code_signature() -> dict[str, str]:
    """Pin feature reuse to the exact online matching implementation."""
    return {
        name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
        for name in ("main.py", "match.py")
    }


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

    ignored_families = set() if include_glibc else {"libc"}
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
    # The thesis classifies an ELF/archive-family pair.  Counting every build
    # separately would multiply both positives and negatives by the number of
    # variants of that archive and distort the grouped-fold balance.
    labels = (
        archive_names
        if cases and "expected_candidates" in cases[0]
        else sorted({archive_family(name) for name in archive_names})
    )
    positive = sum(
        candidate_is_expected(case, label)
        for case in cases
        for label in labels
    )
    total = len(cases) * len(labels)
    return {"positive": positive, "negative": total - positive, "total": total}


def configure_candidate_ground_truth(
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, Any]],
    ground_truth_unit: str,
) -> None:
    """Bind selected aliases to either exact-build or legacy family labels."""
    if ground_truth_unit != "exact_build":
        for case in cases:
            case.pop("expected_candidates", None)
            case.pop("expected_cus_by_candidate", None)
        return

    identities_by_label = {
        str(archive["name"]): tuple(archive["build_identity"])
        for archive in selected_archives
    }
    for case in cases:
        expected_builds = set(case.get("expected_builds", set()))
        expected_cus_by_build = case.get("expected_cus_by_build", {})
        case["expected_candidates"] = {
            label
            for label, identity in identities_by_label.items()
            if identity in expected_builds
        }
        case["expected_cus_by_candidate"] = {
            label: set(expected_cus_by_build[identity])
            for label, identity in identities_by_label.items()
            if identity in expected_cus_by_build
        }


def candidate_is_expected(case: dict[str, Any], candidate_name: str) -> bool:
    """Classify one candidate without relaxing exact-build identities."""
    if "expected_candidates" in case:
        return candidate_name in case["expected_candidates"]
    return archive_family(candidate_name) in case["expected"]


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

    # Prefer distinct programs carrying previously unseen positive families.
    # Pure random balancing frequently selected six unrelated/negative unseen
    # variants, leaving grouped CV with positives in only one program.
    family_frequency: dict[str, int] = {}
    for case in cases:
        for family in set(case.get("expected", set())):
            family_frequency[family] = family_frequency.get(family, 0) + 1

    remaining = list(cases)
    random.Random(seed).shuffle(remaining)
    selected: list[dict[str, Any]] = []
    covered_families: set[str] = set()
    selected_programs: set[str] = set()
    compiler_counts: dict[str, int] = {}
    while remaining and len(selected) < count:
        def candidate_key(case: dict[str, Any]) -> tuple[Any, ...]:
            families = set(case.get("expected", set()))
            new_families = families - covered_families
            coverage_weight = sum(
                1.0 / family_frequency[family]
                for family in new_families
                if family_frequency.get(family)
            )
            compiler = str(case["compiler"])
            is_new_program = str(case["program"]) not in selected_programs
            return (
                bool(families) and is_new_program,
                coverage_weight,
                len(new_families),
                is_new_program,
                -compiler_counts.get(compiler, 0),
            )

        best_index = max(
            range(len(remaining)),
            key=lambda index: candidate_key(remaining[index]),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        covered_families.update(chosen.get("expected", set()))
        selected_programs.add(str(chosen["program"]))
        compiler = str(chosen["compiler"])
        compiler_counts[compiler] = compiler_counts.get(compiler, 0) + 1

    return sorted(selected, key=lambda case: case["variant"])


def select_structural_screen_cases(
    cases: list[dict[str, Any]],
    variants_per_program: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep every program and preserve positive labels before balancing builds."""
    if variants_per_program < 1:
        raise ValueError(
            "--structural-screen-variants-per-program must be at least 1"
        )
    cases_by_program: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_program.setdefault(str(case["program"]), []).append(case)

    selected: list[dict[str, Any]] = []
    combination_counts: dict[tuple[str, str], int] = {}
    compiler_counts: dict[str, int] = {}
    optimization_counts: dict[str, int] = {}
    label_frequency: dict[str, int] = {}
    for case in cases:
        for label in case.get("expected_candidates", case.get("expected", set())):
            label_frequency[str(label)] = label_frequency.get(str(label), 0) + 1
    covered_labels: set[str] = set()
    rng = random.Random(f"{seed}:structural-screen")
    program_order = sorted(cases_by_program)
    rng.shuffle(program_order)

    for round_index in range(variants_per_program):
        for program in program_order:
            already_selected = {
                str(case["variant"])
                for case in selected
                if str(case["program"]) == program
            }
            candidates = [
                case
                for case in cases_by_program[program]
                if str(case["variant"]) not in already_selected
            ]
            if not candidates:
                continue
            rng.shuffle(candidates)

            def balance_key(case: dict[str, Any]) -> tuple[float | int, ...]:
                compiler = str(case["compiler"])
                optimization = str(case["elf_optimization"])
                combination = (compiler, optimization)
                labels = {
                    str(label)
                    for label in case.get(
                        "expected_candidates", case.get("expected", set())
                    )
                }
                unseen_labels = labels - covered_labels
                return (
                    bool(labels),
                    sum(1.0 / label_frequency[label] for label in unseen_labels),
                    len(unseen_labels),
                    len(labels),
                    -combination_counts.get(combination, 0),
                    -compiler_counts.get(compiler, 0),
                    -optimization_counts.get(optimization, 0),
                    int(round_index == 0),
                )

            chosen = max(candidates, key=balance_key)
            selected.append(chosen)
            covered_labels.update(
                str(label)
                for label in chosen.get(
                    "expected_candidates", chosen.get("expected", set())
                )
            )
            compiler = str(chosen["compiler"])
            optimization = str(chosen["elf_optimization"])
            combination = (compiler, optimization)
            combination_counts[combination] = (
                combination_counts.get(combination, 0) + 1
            )
            compiler_counts[compiler] = compiler_counts.get(compiler, 0) + 1
            optimization_counts[optimization] = (
                optimization_counts.get(optimization, 0) + 1
            )

    missing_programs = set(cases_by_program) - {
        str(case["program"]) for case in selected
    }
    if missing_programs:
        raise RuntimeError(
            "Structural screen lost programs: " + ", ".join(sorted(missing_programs))
        )
    return sorted(selected, key=lambda case: case["variant"])


def select_thesis_screen_cases(
    cases: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Choose two ELF per development program with 9–10 of every build type."""
    combinations = [
        (compiler, optimization)
        for compiler in THESIS_COMPILERS
        for optimization in THESIS_OPTIMIZATIONS
    ]
    by_program: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for case in cases:
        program = str(case["program"])
        combination = (
            str(case.get("compiler_config", case["compiler"])),
            str(case["elf_optimization"]),
        )
        program_cases = by_program.setdefault(program, {})
        if combination in program_cases:
            raise ValueError(f"Duplicate development build: {program} {combination}")
        program_cases[combination] = case
    if len(by_program) != 75 or any(
        set(program_cases) != set(combinations)
        for program_cases in by_program.values()
    ):
        raise ValueError(
            "Section 5.3 requires 75 development programs with all 16 "
            "compiler/optimization variants each (1200 ELF)."
        )
    rng = random.Random(f"{seed}:thesis-screen")
    programs = sorted(by_program)
    rng.shuffle(programs)
    counts = {combination: 0 for combination in combinations}
    selected = []
    for program in programs:
        order = list(combinations)
        rng.shuffle(order)
        choices = sorted(order, key=lambda combination: counts[combination])[:2]
        for combination in choices:
            selected.append(by_program[program][combination])
            counts[combination] += 1
    if len(selected) != 150 or set(counts.values()) != {9, 10}:
        raise RuntimeError(
            "Could not balance the 150 screening ELF at 9–10 per build type: "
            f"{counts}"
        )
    return sorted(selected, key=lambda case: str(case["variant"]))


def select_structural_screen_programs(
    cases: list[dict[str, Any]],
    program_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bound screen cost while retaining programs with candidate positives."""
    if program_count < 0:
        raise ValueError("--structural-screen-program-count cannot be negative")
    cases_by_program: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_program.setdefault(str(case["program"]), []).append(case)
    if program_count == 0 or program_count >= len(cases_by_program):
        return sorted(cases, key=lambda case: case["variant"])

    labels_by_program = {
        program: set().union(
            *(
                set(case.get("expected_candidates", case.get("expected", set())))
                for case in program_cases
            )
        )
        for program, program_cases in cases_by_program.items()
    }
    label_frequency: dict[str, int] = {}
    for labels in labels_by_program.values():
        for label in labels:
            label_frequency[str(label)] = label_frequency.get(str(label), 0) + 1

    rng = random.Random(f"{seed}:structural-screen-programs")
    remaining = sorted(cases_by_program)
    rng.shuffle(remaining)
    chosen: list[str] = []
    covered_labels: set[str] = set()
    compiler_counts: dict[str, int] = {}
    optimization_counts: dict[str, int] = {}
    while remaining and len(chosen) < program_count:
        def program_key(program: str) -> tuple[Any, ...]:
            labels = {str(label) for label in labels_by_program[program]}
            new_labels = labels - covered_labels
            program_cases = cases_by_program[program]
            return (
                bool(labels),
                sum(1.0 / label_frequency[label] for label in new_labels),
                len(new_labels),
                len(labels),
                -sum(
                    compiler_counts.get(str(case["compiler"]), 0)
                    for case in program_cases
                ),
                -sum(
                    optimization_counts.get(str(case["elf_optimization"]), 0)
                    for case in program_cases
                ),
            )

        selected = max(remaining, key=program_key)
        remaining.remove(selected)
        chosen.append(selected)
        covered_labels.update(
            str(label) for label in labels_by_program[selected]
        )
        for case in cases_by_program[selected]:
            compiler = str(case["compiler"])
            optimization = str(case["elf_optimization"])
            compiler_counts[compiler] = compiler_counts.get(compiler, 0) + 1
            optimization_counts[optimization] = (
                optimization_counts.get(optimization, 0) + 1
            )

    chosen_set = set(chosen)
    return sorted(
        [case for case in cases if str(case["program"]) in chosen_set],
        key=lambda case: case["variant"],
    )


def select_structural_screen_archives(
    selected_archives: list[dict[str, Any]],
    max_bytes: int,
    max_count: int = 0,
    screen_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a bounded but structurally diverse screening archive panel."""
    if max_bytes < 0:
        raise ValueError(
            "--structural-screen-max-archive-bytes cannot be negative"
        )
    if max_count < 0:
        raise ValueError("--structural-screen-archive-count cannot be negative")
    screen_archives = [
        archive
        for archive in selected_archives
        if max_bytes == 0 or int(archive["size"]) <= max_bytes
    ]
    def basename(archive: dict[str, Any]) -> str:
        return str(archive.get("archive_name", archive["name"]))

    for removed_name, added_name in STRUCTURAL_SCREEN_ARCHIVE_REPLACEMENTS:
        removed = next(
            (archive for archive in screen_archives if basename(archive) == removed_name),
            None,
        )
        added = next(
            (
                archive
                for archive in selected_archives
                if basename(archive) == added_name
                and archive not in screen_archives
            ),
            None,
        )
        if removed is None or added is None:
            continue
        screen_archives.remove(removed)
        screen_archives.append(added)

    if max_count and len(screen_archives) > max_count:
        prevalence = {
            str(archive["name"]): sum(
                candidate_is_expected(case, str(archive["name"]))
                for case in (screen_cases or [])
            )
            for archive in screen_archives
        }
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for archive in screen_archives:
            groups.setdefault(
                (str(archive["kind"]), basename(archive)), []
            ).append(archive)
        representatives = [
            max(
                archives,
                key=lambda archive: (
                    prevalence[str(archive["name"])],
                    -int(archive["size"]),
                    str(archive["name"]),
                ),
            )
            for archives in groups.values()
        ]
        linked_positives = sorted(
            (
                archive for archive in screen_archives
                if str(archive["kind"]) == "positive"
                and prevalence[str(archive["name"])] > 0
            ),
            key=lambda archive: (
                -prevalence[str(archive["name"])],
                int(archive["size"]),
                str(archive["name"]),
            ),
        )
        selected = linked_positives[:max_count - 1]
        retained = {str(archive["name"]) for archive in selected}
        positive_basenames = {
            basename(item) for item in screen_archives
            if str(item["kind"]) == "positive"
        }
        negative_representatives = sorted(
            (
                archive for archive in representatives
                if str(archive["kind"]) == "negative"
            ),
            key=lambda archive: (
                str(archive.get("provenance", ""))
                != "local_versioned_build_absent_from_linker_map",
                basename(archive) not in positive_basenames,
                int(archive["size"]), str(archive["name"])
            ),
        )
        for archive in negative_representatives:
            if len(selected) >= max_count:
                break
            selected.append(archive)
            retained.add(str(archive["name"]))
        for archive in representatives:
            if len(selected) >= max_count:
                break
            if str(archive["name"]) not in retained:
                selected.append(archive)
                retained.add(str(archive["name"]))
        extras = sorted(
            (archive for archive in screen_archives
             if str(archive["name"]) not in retained),
            key=lambda archive: (
                -prevalence[str(archive["name"])],
                str(archive["kind"]) != "negative",
                int(archive["size"]),
                str(archive["name"]),
            ),
        )
        screen_archives = selected + extras[:max_count - len(selected)]

    screen_archives.sort(key=lambda archive: str(archive["name"]))
    kinds = {str(archive["kind"]) for archive in screen_archives}
    if not {"positive", "negative"}.issubset(kinds):
        raise ValueError(
            "Structural screen archive limit must retain at least one positive "
            "and one negative archive"
        )
    return screen_archives


THESIS_VERSION_ROLES = ("current", "minor-alternative", "major-alternative")
THESIS_COMPILERS = (
    "gcc-11-11.5.0", "gcc-13-13.3.0",
    "clang-14-14.0.6", "clang-18-18.1.3",
)
THESIS_OPTIMIZATIONS = ("O0", "O2", "O3", "Os")


def load_thesis_library_corpus(
    args: argparse.Namespace, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Load the entire versioned matrix, preserving each build as a candidate."""
    with args.batch_library_matrix.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if row.get("status") == "selected" and row.get("path")
        ]
    families = {archive_family(str(row["archive"])) for row in rows}
    if len(rows) != 4440 or len(families) != 164:
        raise ValueError(
            "The Section 5.3 corpus requires 4440 selected builds from 164 "
            f"archive families; {args.batch_library_matrix} contains "
            f"{len(rows)} builds from {len(families)} families. Supply the "
            "development matrix used by the thesis before tuning."
        )
    batch_args = SimpleNamespace(
        library_matrix=args.batch_library_matrix,
        library_root=args.batch_library_root,
        library_metadata={},
    )
    paths = select_batch_libraries(batch_args)
    if len(paths) != len(rows):
        raise ValueError(
            "The development library matrix contains duplicate selected paths: "
            f"{len(rows)} rows resolve to {len(paths)} distinct builds."
        )
    labels = batch_library_labels(paths, batch_args.library_metadata)
    rows_by_path = {
        (args.batch_library_root / str(row["path"])).resolve().as_posix(): row
        for row in rows
    }
    expected_families = set().union(*(case["expected"] for case in cases))
    archives = []
    for path in paths:
        row = rows_by_path[path.as_posix()]
        label = labels[path.as_posix()]
        archives.append({
            "name": label,
            "archive_name": row["archive"],
            "path": path.as_posix(),
            "kind": (
                "positive" if archive_family(label) in expected_families
                else "negative"
            ),
            "source": row["project_candidates"],
            "role": row["role"],
            "compiler": row["toolchain"],
            "compiler_root": row["toolchain"],
            "optimization": row["optimization"],
            "variant": "/".join((
                row["role"], row["toolchain"], row["optimization"]
            )),
            "size": path.stat().st_size,
        })
    return sorted(archives, key=lambda archive: str(archive["name"]))


def select_thesis_screen_archives(
    archives: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select 60 families with one build per version/compiler pair.

    Every selected family must have three builds at each optimization level.
    An incomplete source matrix raises an error instead of silently changing
    the screening corpus described in Section 5.3.
    """
    by_family: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for archive in archives:
        family = archive_family(str(archive["name"]))
        pair = (str(archive["role"]), str(archive["compiler"]))
        by_family.setdefault(family, {}).setdefault(pair, []).append(archive)
    pairs = [
        (role, compiler)
        for role in THESIS_VERSION_ROLES
        for compiler in THESIS_COMPILERS
    ]
    prevalence = {
        family: sum(family in case["expected"] for case in cases)
        for family in by_family
    }

    def balanced_builds(
        family_pairs: dict[tuple[str, str], list[dict[str, Any]]], rotation: int
    ) -> list[dict[str, Any]] | None:
        if any(not family_pairs.get(pair) for pair in pairs):
            return None
        chosen: list[dict[str, Any]] = []
        counts = {optimization: 0 for optimization in THESIS_OPTIMIZATIONS}

        def assign(index: int) -> bool:
            if index == len(pairs):
                return all(count == 3 for count in counts.values())
            options = sorted(
                family_pairs[pairs[index]],
                key=lambda archive: (
                    (
                        THESIS_OPTIMIZATIONS.index(str(archive["optimization"]))
                        - index - rotation
                    ) % len(THESIS_OPTIMIZATIONS),
                    str(archive["name"]),
                ),
            )
            for archive in options:
                optimization = str(archive["optimization"])
                if optimization not in counts or counts[optimization] >= 3:
                    continue
                counts[optimization] += 1
                chosen.append(archive)
                if assign(index + 1):
                    return True
                chosen.pop()
                counts[optimization] -= 1
            return False

        return chosen if assign(0) else None

    eligible: list[tuple[str, list[dict[str, Any]]]] = []
    for index, family in enumerate(sorted(by_family)):
        chosen = balanced_builds(by_family[family], index)
        if chosen is not None:
            eligible.append((family, chosen))
    positives = sorted(
        (item for item in eligible if prevalence[item[0]]),
        key=lambda item: (-prevalence[item[0]], item[0]),
    )
    negatives = sorted(
        (item for item in eligible if not prevalence[item[0]]),
        key=lambda item: item[0],
    )
    if not positives or not negatives or len(eligible) < 60:
        raise ValueError(
            "The Section 5.3 screening panel needs 60 families with all "
            "12 version/compiler pairs and a balanced optimization assignment, "
            "including positive and negative families; the matrix provides "
            f"only {len(eligible)} eligible families "
            f"({len(positives)} positive, {len(negatives)} negative)."
        )
    selected = [*positives[:59], *negatives]
    if len(selected) < 60:
        selected = [*positives, *negatives]
    result = [archive for _, group in selected[:60] for archive in group]
    if len(result) != 720:
        raise RuntimeError(f"Expected 720 screening builds, found {len(result)}")
    return sorted(result, key=lambda archive: str(archive["name"]))


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
    candidate_names = set(archive_names)
    candidate_families = {archive_family(name) for name in archive_names}

    def positive_families(selected: list[dict[str, Any]]) -> set[str]:
        if selected and "expected_candidates" in selected[0]:
            return set().union(
                *(
                    set(case.get("expected_candidates", set())) & candidate_names
                    for case in selected
                )
            )
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
    parameter_names = tuple(STRUCTURAL_DEFAULTS)
    default_tuple = tuple(STRUCTURAL_DEFAULTS.values())
    minimum_count = max(len(STRUCTURAL_GRIDS[name]) for name in parameter_names)
    if count < minimum_count:
        raise ValueError(
            f"--structural-trials must be at least {minimum_count} "
            "to cover every structural parameter value"
        )

    # Balanced interaction design: profile_000 is the production baseline;
    # every other row combines values from all structural dimensions.  Each
    # grid value is guaranteed to occur before random balanced repetitions are
    # added, avoiding the old one-factor-at-a-time design with no interactions.
    slots = count - 1
    columns: dict[str, list[float | int]] = {}
    for parameter in parameter_names:
        default = STRUCTURAL_DEFAULTS[parameter]
        values: list[float | int] = []
        cycle = 0
        while len(values) < slots:
            cycle_values = (
                [
                    value
                    for value in STRUCTURAL_GRIDS[parameter]
                    if value != default
                ]
                if cycle == 0
                else list(STRUCTURAL_GRIDS[parameter])
            )
            random.Random(f"{seed}:{parameter}:{cycle}").shuffle(cycle_values)
            values.extend(cycle_values)
            cycle += 1
        columns[parameter] = values[:slots]

    chosen = [default_tuple]
    for index in range(slots):
        combination = tuple(columns[name][index] for name in parameter_names)
        if combination in chosen:
            alternatives = list(
                itertools.product(
                    *(STRUCTURAL_GRIDS[name] for name in parameter_names)
                )
            )
            alternatives = [value for value in alternatives if value not in chosen]
            if not alternatives:
                raise ValueError("Requested more unique structural profiles than exist")
            combination = random.Random(
                f"{seed}:unique-profile:{index}"
            ).choice(alternatives)
        chosen.append(combination)

    profiles = []
    for index, values in enumerate(chosen):
        profile = dict(zip(parameter_names, values))
        profile["profile_id"] = f"profile_{index:03d}"
        profiles.append(profile)
    return profiles


def append_extra_high_structural_profiles(
    profiles: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Densify the high block-threshold range, preserving cached profile IDs."""
    expanded = list(profiles)
    for threshold, multiplier in (
        (0.875, 1.5),
        (0.925, 2.5),
        (0.975, 2.5),
        (0.975, 5.0),
        (0.99, 2.5),
        (0.99, 4.0),
    ):
        expanded.append({
            **STRUCTURAL_DEFAULTS,
            "block_threshold": threshold,
            "block_locality_window_multiplier": multiplier,
            "profile_id": f"profile_{len(expanded):03d}",
        })
    return expanded


def screen_optuna_trial_budget(
    requested: int | None,
    profile_count: int,
) -> int:
    """Scale cheap offline screening trials with costly profile collections."""
    if requested is not None:
        if requested < 1:
            raise ValueError("--structural-screen-optuna-trials must be at least 1")
        return requested
    return max(1200, profile_count * 120)


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
        "--offline-ablation-features",
        "--jobs",
        str(args.collection_workers),
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
        "--block-min-instructions",
        str(profile["block_min_instructions"]),
        "--block-coverage-mean-threshold",
        str(min(DECISION_GRIDS["block_coverage_mean_threshold"])),
        "--block-assignment-quality-threshold",
        "0",
        "--block-min-assignment-ratio",
        "0",
        "--block-min-coverage-ratio",
        str(min(DECISION_GRIDS["block_min_coverage_ratio"])),
        "--block-min-call-edge-ratio",
        "0",
        "--block-min-function-concentration",
        "0",
        "--block-min-function-spread",
        "0",
        "--cu-min-function-coverage",
        "0",
    ]
    if args.no_analysis_cache:
        command.append("--no-analysis-cache")
    else:
        command.extend(
            ["--analysis-cache-dir", str(args.analysis_cache_dir)]
        )
        if args.analysis_cache_only:
            command.append("--analysis-cache-only")
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])
    if getattr(args, "fast_negative_bound", False):
        command.append("--fast-negative-bound")
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
    included_libraries: set[str] | None = None,
) -> tuple[
    str,
    dict[str, list[dict[str, Any]]],
    set[str],
    dict[int, list[int]],
]:
    """Stream one feature file and retain only fields used by replay."""
    binary_path = ""
    libraries: dict[str, list[dict[str, Any]]] = {}
    pooling_modes: set[str] = set()
    seen_records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    source_call_targets: dict[int, list[int]] = {}
    duplicate_count = 0
    decoder = json.JSONDecoder()

    block_fields = (
        "coverage_mean",
        "coverage_min",
        "coverage_ratio",
        "assignment_quality",
        "assignment_ratio",
        "call_edge_ratio",
        "function_concentration",
        "function_spread",
        "function_coverage",
    )
    block_count_fields = (
        "call_edges_evaluated",
        "call_edges_total",
    )

    def parse_function_matches(
        raw_matches: list[Any],
    ) -> list[int]:
        if not raw_matches:
            return []
        if not isinstance(raw_matches[0], dict):
            flattened = [int(value) for value in raw_matches]
            if len(flattened) % 2:
                raise ValueError(f"Odd flattened function mapping in {path}")
            return flattened

        flattened = []
        for match in raw_matches:
            target_index = int(match["target_function_index"])
            source_index = int(match["source_function_index"])
            flattened.extend((target_index, source_index))
            # Schema-3 replay logs keep the ELF call graph once in the header.
            # Older autonomous records repeat it inside every mapping.
            if "source_call_targets" in match:
                targets = [
                    int(target)
                    for target in match.get("source_call_targets", [])
                ]
                previous = source_call_targets.get(source_index)
                if previous is not None and previous != targets:
                    raise ValueError(
                        f"Conflicting source calls for function {source_index} in {path}"
                    )
                source_call_targets[source_index] = targets
        return flattened

    def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        target_function_count = int(
            payload.get("target_function_count")
            or len(payload.get("target_functions", []))
        )

        def compact_window(window: dict[str, Any]) -> dict[str, Any]:
            raw_mapping = window.get("function_matches", [])
            mapping = parse_function_matches(raw_mapping)
            mapped_targets = {
                int(match["target_function_index"])
                for match in raw_mapping
                if isinstance(match, dict)
                and float(match.get("coverage_ratio", 0.0)) > 0.0
            } if raw_mapping and isinstance(raw_mapping[0], dict) else {
                mapping[index] for index in range(0, len(mapping), 2)
            }
            return {
                **{field: float(window.get(field, 0.0)) for field in block_fields},
                **{
                    field: int(window.get(field, 0))
                    for field in block_count_fields
                },
                "function_coverage": float(
                    window.get(
                        "function_coverage",
                        (
                            len(mapped_targets) / target_function_count
                            if target_function_count
                            else 0.0
                        ),
                    )
                ),
                "function_matches": mapping,
            }

        windows = [compact_window(window) for window in payload.get("windows", [])]
        record = {
            "name": str(payload.get("name", "")),
            "target_cu_index": int(payload["target_cu_index"]),
            "windows": windows,
            "inter_cu_calls": [
                {
                    "caller_function_index": int(edge["caller_function_index"]),
                    "callee_cu_index": int(edge["callee_cu_index"]),
                    "callee_function_index": int(edge["callee_function_index"]),
                }
                for edge in payload.get("inter_cu_calls", [])
            ],
            "target_function_count": target_function_count,
            "rodata": float(payload.get("rodata", 0.0)),
            "rodata_has_rodata": bool(payload.get("rodata_has_rodata", False)),
            "rodata_strings": int(payload.get("rodata_strings", 0)),
            "rodata_ngrams": int(payload.get("rodata_ngrams", 0)),
            "rodata_bytes": int(payload.get("rodata_bytes", 0)),
        }
        if "rodata_string_informative" in payload:
            record.update({
                "rodata_string_score": float(payload["rodata_string_score"]),
                "rodata_byte_score": float(payload["rodata_byte_score"]),
                "rodata_string_informative": bool(payload["rodata_string_informative"]),
                "rodata_byte_informative": bool(payload["rodata_byte_informative"]),
            })
        if not windows:
            record.update(
                {field: float(payload.get(field, 0.0)) for field in block_fields}
            )
            record.update(
                {
                    field: int(payload.get(field, 0))
                    for field in block_count_fields
                }
            )
            raw_mapping = payload.get("function_matches", [])
            mapping = parse_function_matches(raw_mapping)
            mapped_targets = {
                int(match["target_function_index"])
                for match in raw_mapping
                if isinstance(match, dict)
                and float(match.get("coverage_ratio", 0.0)) > 0.0
            } if raw_mapping and isinstance(raw_mapping[0], dict) else {
                mapping[index] for index in range(0, len(mapping), 2)
            }
            record["function_matches"] = mapping
            record["function_coverage"] = float(
                payload.get(
                    "function_coverage",
                    (
                        len(mapped_targets) / target_function_count
                        if target_function_count
                        else 0.0
                    ),
                )
            )
        return record

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
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
                if payload.get("type") == "source_call_targets":
                    record_binary_path = str(
                        Path(str(payload["binary_path"])).resolve()
                    )
                    if binary_path and record_binary_path != binary_path:
                        raise ValueError(
                            f"Mixed binary paths in feature file {path}"
                        )
                    binary_path = record_binary_path
                    for source_index, targets in payload.get("calls", []):
                        source_call_targets[int(source_index)] = [
                            int(target) for target in targets
                        ]
                    continue
                if payload.get("type") != "block_cu":
                    continue
                pooling_modes.add(
                    str(payload.get("palmtree_pooling", "mean"))
                )
                record_binary_path = str(
                    Path(str(payload["binary_path"])).resolve()
                )
                library = str(payload["library"])
                if (
                    included_libraries is not None
                    and library not in included_libraries
                ):
                    continue
                record_name = str(payload.get("name", ""))
                record = compact_payload(payload)
                identity = (
                    record_binary_path,
                    library,
                    record_name,
                    int(record["target_cu_index"]),
                )
                previous = seen_records.get(identity)
                if previous is not None:
                    if previous != record:
                        raise ValueError(
                            "Conflicting duplicate CU feature for "
                            f"binary={record_binary_path}, library={library}, "
                            f"name={record_name} in {path}"
                        )
                    duplicate_count += 1
                    continue
                seen_records[identity] = record

                if binary_path and record_binary_path != binary_path:
                    raise ValueError(f"Mixed binary paths in feature file {path}")
                binary_path = record_binary_path
                libraries.setdefault(library, []).append(record)

    if duplicate_count:
        print(
            f"[features] recovered {path.name}: ignored "
            f"{duplicate_count} exact duplicate CU record(s)"
        )

    return binary_path, libraries, pooling_modes, source_call_targets


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
    source_call_targets_by_case: dict[str, dict[int, list[int]]] = {}
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
            report.name.replace(".report.txt", ".features.jsonl.gz")
        )
        if not feature_file.is_file():
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
            feature_source_call_targets,
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
        source_call_targets_by_case[binary_path] = feature_source_call_targets
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
        "source_call_targets": source_call_targets_by_case,
    }


def selected_batch_aliases(
    args: argparse.Namespace,
    selected_archives: list[dict[str, Any]],
) -> dict[str, str]:
    """Map batch-run aliases to the exact-build labels used by this search."""
    batch_args = SimpleNamespace(
        library_matrix=args.batch_library_matrix,
        library_root=args.batch_library_root,
        library_metadata={},
    )
    libraries = select_batch_libraries(batch_args)
    aliases_by_path = batch_library_labels(
        libraries, batch_args.library_metadata
    )
    selected: dict[str, str] = {}
    for archive in selected_archives:
        path = Path(str(archive["path"])).resolve().as_posix()
        alias = aliases_by_path.get(path)
        if alias is None:
            raise ValueError(
                "Selected exact build is absent from the batch library matrix: "
                f"{path}"
            )
        if alias in selected:
            raise ValueError(f"Duplicate batch library alias: {alias}")
        selected[alias] = str(archive["name"])
    return selected


def import_batch_feature_collection(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    profile_signature: str,
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stream only selected builds from complete replay-ready batch logs."""
    if any(
        profile.get(name) != value
        for name, value in STRUCTURAL_DEFAULTS.items()
    ):
        raise ValueError(
            "Batch feature import supports only the structural profile used "
            f"by the current run: {STRUCTURAL_DEFAULTS}"
        )
    reports_dir = args.import_batch_reports_dir
    if reports_dir is None or not reports_dir.is_dir():
        raise FileNotFoundError(f"Batch reports directory not found: {reports_dir}")

    alias_to_candidate = selected_batch_aliases(args, selected_archives)
    included_aliases = set(alias_to_candidate)
    imported_cases: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_maps: dict[str, dict[int, list[int]]] = {}
    normalization_modes: set[str] = set()
    pooling_modes: set[str] = set()

    for case in cases:
        matches = sorted(
            reports_dir.glob(f"{case['variant']}_*.report.txt")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one complete batch report for {case['variant']}, "
                f"found {len(matches)}"
            )
        report = matches[0]
        report_text = report.read_text(encoding="utf-8", errors="replace")
        if "Feature collection mode: replay_complete" not in report_text:
            raise ValueError(f"Batch report is not replay-complete: {report}")
        normalization = report_asm_normalization(report)
        pooling = report_palmtree_pooling(report)
        normalization_modes.add(normalization)
        pooling_modes.add(pooling)
        report_binary, report_libraries = parse_report(report)
        missing_summaries = included_aliases - set(report_libraries)
        if missing_summaries:
            raise ValueError(
                f"Incomplete batch report {report}: missing "
                + ", ".join(sorted(missing_summaries))
            )

        feature_file = report.with_name(
            report.name.replace(".report.txt", ".features.jsonl.gz")
        )
        if not feature_file.is_file():
            raise FileNotFoundError(f"Batch feature file not found: {feature_file}")
        (
            feature_binary,
            libraries,
            feature_pooling,
            source_calls,
        ) = parse_feature_jsonl(feature_file, included_libraries=included_aliases)
        expected_binary = str(Path(case["binary"]).resolve())
        actual_binary = feature_binary or report_binary
        if actual_binary != expected_binary:
            raise ValueError(
                f"Batch feature binary mismatch for {case['variant']}: "
                f"{actual_binary} != {expected_binary}"
            )
        if feature_pooling and feature_pooling != {pooling}:
            raise ValueError(
                f"Batch feature/report pooling mismatch: {feature_file}"
            )
        imported_cases[expected_binary] = {
            candidate: list(libraries.get(alias, []))
            for alias, candidate in alias_to_candidate.items()
        }
        source_maps[expected_binary] = source_calls

    if normalization_modes != {args.asm_normalization}:
        raise ValueError(
            "Imported batch normalization mismatch: "
            f"{sorted(normalization_modes)}"
        )
    if pooling_modes != {args.palmtree_pooling}:
        raise ValueError(
            f"Imported batch pooling mismatch: {sorted(pooling_modes)}"
        )
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "asm_normalization": args.asm_normalization,
        "palmtree_pooling": args.palmtree_pooling,
        "profile": profile,
        "input_signature": profile_signature,
        "cases": imported_cases,
        "source_call_targets": source_maps,
    }


def compact_feature_collection(collection: dict[str, Any]) -> dict[str, Any]:
    """Discard feature fields that cannot affect any replayed decision.

    Raw JSONL remains deliberately verbose for diagnostics. Persistent search
    collections only need threshold metrics, CU identities, .rodata counts, and
    the minimal function/call mapping used by cross-CU replay.
    """
    block_fields = (
        "coverage_mean",
        "coverage_min",
        "coverage_ratio",
        "assignment_quality",
        "assignment_ratio",
        "call_edge_ratio",
        "function_concentration",
        "function_spread",
        "function_coverage",
        "call_edges_evaluated",
        "call_edges_total",
    )

    cases = collection.get("cases", {})
    source_maps = collection.setdefault("source_call_targets", {})
    for binary_path, libraries in cases.items():
        existing_source_map = source_maps.get(binary_path, {})
        source_map = {
            int(source_index): [int(target) for target in targets]
            for source_index, targets in existing_source_map.items()
        }

        def compact_matches(matches: list[Any]) -> list[int]:
            if not matches:
                return []
            if not isinstance(matches[0], dict):
                flattened = [int(value) for value in matches]
                if len(flattened) % 2:
                    raise ValueError("Odd flattened function mapping")
                return flattened
            flattened = []
            for match in matches:
                target_index = int(match["target_function_index"])
                source_index = int(match["source_function_index"])
                flattened.extend((target_index, source_index))
                if "source_call_targets" in match:
                    targets = [
                        int(target)
                        for target in match.get("source_call_targets", [])
                    ]
                    previous = source_map.get(source_index)
                    if previous is not None and previous != targets:
                        raise ValueError(
                            f"Conflicting calls for source function {source_index}"
                        )
                    source_map[source_index] = targets
            return flattened

        def compact_window(window: dict[str, Any]) -> dict[str, Any]:
            return {
                **{field: window[field] for field in block_fields},
                "function_matches": compact_matches(
                    window.get("function_matches", [])
                ),
            }

        for library, records in list(libraries.items()):
            compact_records = []
            for record in records:
                windows = [
                    compact_window(window) for window in record.get("windows", [])
                ]
                compact_record = {
                    "name": str(record.get("name", "")),
                    "windows": windows,
                    "inter_cu_calls": [
                        {
                            "caller_function_index": int(
                                edge["caller_function_index"]
                            ),
                            "callee_cu_index": int(edge["callee_cu_index"]),
                            "callee_function_index": int(
                                edge["callee_function_index"]
                            ),
                        }
                        for edge in record.get("inter_cu_calls", [])
                    ],
                    "rodata": float(record.get("rodata", 0.0)),
                    "rodata_has_rodata": bool(
                        record.get("rodata_has_rodata", False)
                    ),
                    "rodata_strings": int(record.get("rodata_strings", 0)),
                    "rodata_ngrams": int(record.get("rodata_ngrams", 0)),
                    "rodata_bytes": int(record.get("rodata_bytes", 0)),
                }
                if "rodata_string_informative" in record:
                    compact_record.update({
                        "rodata_string_score": float(record["rodata_string_score"]),
                        "rodata_byte_score": float(record["rodata_byte_score"]),
                        "rodata_string_informative": bool(record["rodata_string_informative"]),
                        "rodata_byte_informative": bool(record["rodata_byte_informative"]),
                    })
                if "target_cu_index" in record:
                    compact_record["target_cu_index"] = int(
                        record["target_cu_index"]
                    )
                # Schema-8 records normally contain windows. Top-level block
                # evidence is only needed for legacy/fallback records.
                if not windows:
                    compact_record.update(
                        {field: record[field] for field in block_fields}
                    )
                    compact_record["function_matches"] = compact_matches(
                        record.get("function_matches", [])
                    )
                compact_records.append(compact_record)
            libraries[library] = compact_records
        source_maps[binary_path] = source_map
    return collection


def write_gzip_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON and verify the gzip stream before committing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))
        with gzip.open(temporary, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def persist_feature_collection(
    path: Path,
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> None:
    expected_pairs = len(cases) * len(archive_names)
    actual_pairs = count_cached_pairs(collection, cases, archive_names)
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"Refusing to persist incomplete feature collection {path}: "
            f"{actual_pairs}/{expected_pairs} ELF/library pairs"
        )
    compact_feature_collection(collection)
    write_gzip_json(path, collection)


def cleanup_profile_artifacts(args: argparse.Namespace, profile_id: str) -> None:
    """Remove recomputable intermediates after a verified archive is durable."""
    if args.keep_collection_artifacts or args.dry_run:
        return
    removed = []
    for path in (
        args.output_dir / "collections" / profile_id,
        args.output_dir / "collections_missing" / profile_id,
        args.output_dir / "inputs" / "missing" / profile_id,
    ):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(args.output_dir)))
    if removed:
        print(f"[features] cleaned {profile_id}: " + ", ".join(removed))


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
    existing_source_maps = collection.get("source_call_targets", {})
    normalized_source_maps = {}
    for case in cases:
        binary_path = str(case["binary"])
        existing_libraries = existing_cases.get(binary_path, {})
        normalized_libraries = {}
        for archive_name in archive_names:
            found, records = find_archive_records(existing_libraries, archive_name)
            if found:
                normalized_libraries[archive_name] = records
        normalized_cases[binary_path] = normalized_libraries
        normalized_source_maps[binary_path] = existing_source_maps.get(
            binary_path, {}
        )
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "asm_normalization": collection.get("asm_normalization", "legacy"),
        "palmtree_pooling": collection.get("palmtree_pooling", "mean"),
        "profile": profile,
        "input_signature": input_signature,
        "cases": normalized_cases,
        "source_call_targets": normalized_source_maps,
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
        added_source_map = addition.get("source_call_targets", {}).get(
            binary_path, {}
        )
        destination.setdefault("source_call_targets", {})[binary_path] = (
            added_source_map
        )


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
        if (
            args.import_batch_reports_dir is not None
            and args.dry_run
            and not destination.is_file()
        ):
            # Validate the exact-build-to-batch aliases during preflight too.
            # Otherwise a wrong matrix can pass --dry-run and fail only when
            # the real feature import begins.
            selected_batch_aliases(args, selected_archives)
            print(
                f"[dry-run] would import replay-complete batch logs for {profile_id}"
            )
            collections[profile_id] = {}
            continue
        if args.import_batch_reports_dir is not None and not destination.is_file():
            print(
                f"[features] importing replay-complete batch logs for {profile_id}",
                flush=True,
            )
            imported = import_batch_feature_collection(
                args,
                profile,
                profile_signature,
                cases,
                selected_archives,
            )
            persist_feature_collection(
                destination, imported, cases, archive_names
            )
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
                    compact_feature_collection(normalized)
                    if not args.search_only:
                        persist_feature_collection(
                            destination, normalized, cases, archive_names
                        )
                        cleanup_profile_artifacts(args, profile_id)
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
                compact_feature_collection(partial)
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
            compact_feature_collection(partial)
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
                persist_feature_collection(
                    destination, partial, cases, archive_names
                )
                cleanup_profile_artifacts(args, profile_id)
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
                persist_feature_collection(
                    destination, recovered, cases, archive_names
                )
                cleanup_profile_artifacts(args, profile_id)
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
        persist_feature_collection(destination, collection, cases, archive_names)
        cleanup_profile_artifacts(args, profile_id)
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
            "ground_truth_unit": args.ground_truth_unit,
            "matching_code_signature": matching_code_signature(),
            "fast_negative_bound": args.fast_negative_bound,
            "analysis_cache_only": args.analysis_cache_only,
            "analysis_cache_identity": (
                packaged_common_identity(args.analysis_cache_dir)
                if args.analysis_cache_only
                else None
            ),
            "imported_batch_features": (
                {
                    "reports_dir": str(args.import_batch_reports_dir),
                    "library_matrix_sha256": hashlib.sha256(
                        args.batch_library_matrix.read_bytes()
                    ).hexdigest(),
                }
                if args.import_batch_reports_dir is not None
                else None
            ),
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
        and replay_rodata_score(record, params) <= params["rodata_penalty_threshold"]
    )


def replay_rodata_score(record: dict[str, Any], params: dict[str, Any]) -> float:
    """Recombine raw evidence with the trial weights; retain legacy fallback."""
    if "rodata_string_informative" not in record:
        return float(record["rodata"])
    string_informative = bool(record["rodata_string_informative"])
    byte_informative = bool(record["rodata_byte_informative"])
    if string_informative and byte_informative:
        string_weight = float(params.get("rodata_string_weight", 0.70))
        return (
            string_weight * float(record["rodata_string_score"])
            + (1.0 - string_weight) * float(record["rodata_byte_score"])
        )
    if string_informative:
        return float(record["rodata_string_score"])
    if byte_informative:
        return float(params.get("rodata_byte_only_weight", 0.50)) * float(record["rodata_byte_score"])
    return 0.0


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
    rodata_score = replay_rodata_score(record, params)
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
        and window.get("function_coverage", 0.0)
        >= params.get("cu_min_function_coverage", 0.0)
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


def selected_record_window(
    record: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the highest-scoring passing window used for relation replay."""
    passing = [
        window
        for window in (record.get("windows") or [record])
        if block_window_passes(window, params)
    ]
    if not passing or rodata_is_penalty(record, params):
        return None
    return max(passing, key=lambda window: float(window["coverage_mean"]))


def replay_cross_cu_call_adjustment(
    score: float,
    accepted_records: list[dict[str, Any]],
    params: dict[str, Any],
    source_call_targets: dict[int | str, list[int]],
    selected_by_cu: dict[int, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> float:
    """Replay cross-CU call evidence entirely from collected features."""
    if 0.0 <= float(score) <= 1.0 and not (
        float(params.get("cross_cu_call_bonus_weight", 0.0))
        or float(params.get("cross_cu_call_penalty_weight", 0.0))
    ):
        return score
    if selected_by_cu is None:
        selected_by_cu = {}
        for record in accepted_records:
            if "target_cu_index" not in record:
                continue
            selected = selected_record_window(record, params)
            if selected is not None:
                selected_by_cu[int(record["target_cu_index"])] = (
                    record, selected
                )

    mapping_by_cu = {}
    for cu_index, (_record, window) in selected_by_cu.items():
        flattened = window.get("function_matches", [])
        mapping_by_cu[cu_index] = {
            int(flattened[index]): int(flattened[index + 1])
            for index in range(0, len(flattened), 2)
        }
    matched_edges = 0
    evaluable_edges = 0
    expected_edges = 0
    for caller_cu_index, (caller_record, caller_window) in selected_by_cu.items():
        caller_mapping = mapping_by_cu[caller_cu_index]
        for edge in caller_record.get("inter_cu_calls", []):
            callee_cu_index = int(edge["callee_cu_index"])
            callee_entry = selected_by_cu.get(callee_cu_index)
            if callee_entry is None:
                continue
            expected_edges += 1
            caller_match = caller_mapping.get(
                int(edge["caller_function_index"])
            )
            callee_mapping = mapping_by_cu[callee_cu_index]
            callee_match = callee_mapping.get(
                int(edge["callee_function_index"])
            )
            if caller_match is None or callee_match is None:
                continue
            evaluable_edges += 1
            caller_targets = source_call_targets.get(
                caller_match,
                source_call_targets.get(str(caller_match), []),
            )
            if int(callee_match) in {int(target) for target in caller_targets}:
                matched_edges += 1

    if not evaluable_edges or not expected_edges:
        return score
    ratio = matched_edges / evaluable_edges
    coverage = evaluable_edges / expected_edges
    saturation = max(
        1,
        int(params.get("cross_cu_call_saturation_edges", 3)),
    )
    reliability = min(1.0, evaluable_edges / saturation) * coverage
    bounded_score = min(1.0, max(0.0, float(score)))
    positive = (
        float(params.get("cross_cu_call_bonus_weight", 0.0))
        * reliability
        * ratio
        * (1.0 - bounded_score)
    )
    negative = (
        float(params.get("cross_cu_call_penalty_weight", 0.0))
        * reliability
        * (1.0 - ratio)
        * bounded_score
    )
    return min(1.0, max(0.0, bounded_score + positive - negative))


def library_match_evidence(
    records: list[dict[str, Any]],
    params: dict[str, Any],
    source_call_targets: dict[int | str, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Return accepted CU records and their aggregated library score."""
    accepted: list[dict[str, Any]] = []
    scores: list[float] = []
    selected_by_cu: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    aggregator = str(params.get("library_score_aggregator", "mean"))
    for record in records:
        selected_window = None
        for window in record.get("windows") or [record]:
            if block_window_passes(window, params) and (
                selected_window is None
                or float(window["coverage_mean"])
                > float(selected_window["coverage_mean"])
            ):
                selected_window = window
        if selected_window is None:
            continue
        if rodata_is_penalty(record, params):
            if aggregator == "top3_mean":
                # Preserve the structural top-3 denominator: a .rodata
                # rejection must not increase the library score.
                scores.append(0.0)
            continue
        score = apply_replayed_rodata_bonus(
            float(selected_window["coverage_mean"]), record, params
        )
        accepted.append(record)
        scores.append(score)
        if "target_cu_index" in record:
            selected_by_cu[int(record["target_cu_index"])] = (
                record, selected_window
            )
    library_score = aggregate_library_score(
        scores,
        aggregator,
        score_floor=float(params.get("block_coverage_mean_threshold", 0.0)),
    )
    library_score = replay_cross_cu_call_adjustment(
        library_score,
        accepted,
        params,
        source_call_targets or {},
        selected_by_cu,
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
    expected = candidate_is_expected(case, archive_name)
    if "expected_candidates" in case:
        expected_cus_by_identity = case.get("expected_cus_by_candidate", {})
        expected_cus = set(expected_cus_by_identity.get(archive_name, set()))
        has_expected_cu_ground_truth = archive_name in expected_cus_by_identity
    else:
        expected_cus_by_identity = case.get("expected_cus", {})
        expected_cus = set(expected_cus_by_identity.get(family, set()))
        has_expected_cu_ground_truth = family in expected_cus_by_identity

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
        if not has_expected_cu_ground_truth:
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
    *,
    include_predictions: bool = True,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    collected_cases = collection["cases"]
    collected_source_calls = collection.get("source_call_targets", {})
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    cu_counts = {"tp": 0, "fp": 0, "fn": 0}
    cu_evaluated_pairs = 0
    predictions = []

    for case in cases:
        binary_path = str(case["binary"])
        library_records = collected_cases.get(binary_path)
        if library_records is None:
            raise KeyError(f"No collected features for {binary_path}")
        source_call_targets = collected_source_calls.get(binary_path, {})
        family_decisions: dict[str, tuple[bool, bool]] = {}
        exact_build = "expected_candidates" in case
        first_prediction = len(predictions)

        for archive_name in archive_names:
            _, records = find_archive_records(library_records, archive_name)
            successful, library_score = library_match_evidence(
                records,
                params,
                source_call_targets,
            )
            predicted = bool(successful) and library_score >= float(
                params.get("library_min_score", 0.0)
            )
            expected = candidate_is_expected(case, archive_name)
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
            if exact_build:
                counts[classification.lower()] += 1
            else:
                family = archive_family(archive_name)
                prior_expected, prior_predicted = family_decisions.get(
                    family, (False, False)
                )
                family_decisions[family] = (
                    prior_expected or expected,
                    prior_predicted or predicted,
                )
            if include_predictions:
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

        if not exact_build:
            family_classifications = {}
            for expected, predicted in family_decisions.values():
                classification = (
                    "tp" if expected and predicted
                    else "fn" if expected
                    else "fp" if predicted
                    else "tn"
                )
                counts[classification] += 1
            if include_predictions:
                for family, (expected, predicted) in family_decisions.items():
                    family_classifications[family] = (
                        "TP" if expected and predicted
                        else "FN" if expected
                        else "FP" if predicted
                        else "TN"
                    )
                for prediction in predictions[first_prediction:]:
                    prediction["candidate_classification"] = prediction[
                        "classification"
                    ]
                    prediction["classification"] = family_classifications[
                        archive_family(str(prediction["library"]))
                    ]

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


def decision_defaults(
    library_score_aggregator: str = "top3_noisy_or",
) -> dict[str, Any]:
    return {
        **DECISION_DEFAULTS,
        "library_score_aggregator": library_score_aggregator,
    }


def random_decision_params(
    rng: random.Random,
    library_score_aggregator: str = "top3_noisy_or",
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


def optuna_decision_params(trial: Any) -> dict[str, Any]:
    """Suggest replayable decisions while keeping required evidence active."""
    params = decision_defaults()
    params.update(
        {
            parameter: trial.suggest_categorical(parameter, list(candidates))
            for parameter, candidates in DECISION_GRIDS.items()
        }
    )
    return params


# Forked replay workers inherit the already collected feature data without
# serializing it for every trial. Only the parent samples trials and touches
# Optuna storage.
_OPTUNA_REPLAY_CONTEXT: dict[str, list[list[tuple[list[tuple[Any, Any]], bool]]]] | None = None


def prepare_optuna_replay_pairs(
    collections: dict[str, dict[str, Any]],
    folds: list[dict[str, Any]],
    archive_names: list[str],
    ground_truth_unit: str = "exact_build",
) -> dict[str, list[list[tuple[list[tuple[Any, Any]], bool]]]]:
    """Resolve one ELF/family decision per fold for offline Optuna replay."""
    validation_cases = [
        case for fold in folds for case in fold["validation_cases"]
    ]
    case_paths = [str(case["binary"]) for case in validation_cases]
    if len(case_paths) != len(set(case_paths)):
        raise ValueError("Grouped CV validation folds contain duplicate ELFs")
    pairs = {}
    for profile_id, collection in collections.items():
        profile_folds = []
        for fold in folds:
            fold_pairs = []
            for case in fold["validation_cases"]:
                binary_path = str(case["binary"])
                library_records = collection["cases"].get(binary_path)
                if library_records is None:
                    raise KeyError(f"No collected features for {binary_path}")
                source_map = collection.get("source_call_targets", {}).get(
                    binary_path, {}
                )
                decisions: dict[str, tuple[list[tuple[Any, Any]], bool]] = {}
                for archive_name in archive_names:
                    _, records = find_archive_records(
                        library_records, archive_name
                    )
                    expected = candidate_is_expected(case, archive_name)
                    if not records and not expected:
                        # This candidate cannot change F1; another build of
                        # the same family may still supply positive evidence.
                        continue
                    label = (
                        archive_name if ground_truth_unit == "exact_build"
                        else archive_family(archive_name)
                    )
                    alternatives, was_expected = decisions.setdefault(
                        label, ([], False)
                    )
                    if records:
                        alternatives.append((records, source_map))
                    decisions[label] = (alternatives, was_expected or expected)
                fold_pairs.extend(decisions.values())
            profile_folds.append(fold_pairs)
        pairs[profile_id] = profile_folds
    return pairs


def evaluate_optuna_trial(
    profile_id: str,
    params: dict[str, Any],
    replay_pairs: dict[str, list[list[tuple[list[tuple[Any, Any]], bool]]]],
) -> float:
    minimum_score = float(params.get("library_min_score", 0.0))
    fold_f1 = []
    for fold_pairs in replay_pairs[profile_id]:
        tp = fp = fn = 0
        for alternatives, expected in fold_pairs:
            predicted = False
            for records, source_map in alternatives:
                successful, score = library_match_evidence(
                    records, params, source_map
                )
                if successful and score >= minimum_score:
                    predicted = True
                    break
            tp += int(predicted and expected)
            fp += int(predicted and not expected)
            fn += int(expected and not predicted)
        denominator = 2 * tp + fp + fn
        fold_f1.append(2 * tp / denominator if denominator else 0.0)
    return statistics.fmean(fold_f1)


def evaluate_optuna_trial_worker(
    profile_id: str, params: dict[str, Any]
) -> float:
    if _OPTUNA_REPLAY_CONTEXT is None:
        raise RuntimeError("Optuna replay worker was not initialized")
    return evaluate_optuna_trial(profile_id, params, _OPTUNA_REPLAY_CONTEXT)


def search_all_profiles_optuna(
    args: argparse.Namespace,
    collections: dict[str, dict[str, Any]],
    development_cases: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    archive_names: list[str],
    input_signature: str,
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
    """Jointly optimize structural profile and decisions for library F1."""
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as error:
        raise RuntimeError(
            "Optuna is required for --search-strategy optuna. Install the "
            "updated thesis_code/environment.yml or run `pip install optuna`."
        ) from error

    profile_ids = sorted(collections)
    if not profile_ids:
        raise ValueError("Optuna search requires at least one feature collection")
    if not folds:
        raise ValueError("Optuna search requires grouped cross-validation folds")

    profile_signature = selection_signature(
        [collections[profile_id]["profile"] for profile_id in profile_ids]
    )
    study_name = args.optuna_study_name or (
        f"library_f1_{input_signature[:12]}_{profile_signature[:8]}"
    )
    args.optuna_storage.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.optuna_storage}"
    try:
        previous_study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
    except KeyError:
        previous_trial_count = 0
    else:
        previous_trial_count = len(previous_study.trials)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=args.seed + (previous_trial_count * 1_000_003),
            n_startup_trials=max(50, len(profile_ids) * 2),
            multivariate=True,
        )
    # Five grouped folds are cheap to replay once matching features exist, and
    # their positive-label prevalence is necessarily uneven.  Finish every
    # trial so a weak early fold cannot discard a configuration that performs
    # well on the remaining program groups.
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    expected_attrs = {
        "input_signature": input_signature,
        "profile_signature": profile_signature,
        "decision_grid_signature": selection_signature(
            {
                "grids": DECISION_GRIDS,
                "required_components": REQUIRED_DECISION_COMPONENTS,
            }
        ),
        "objective": "mean_grouped_cv_library_f1",
        "ground_truth_unit": args.ground_truth_unit,
        "fold_count": len(folds),
        "pruner": "none",
    }
    for name, expected in expected_attrs.items():
        existing = study.user_attrs.get(name)
        if (
            name == "decision_grid_signature"
            and existing is None
            and study.trials
        ):
            raise ValueError(
                f"Optuna study {study_name!r} predates the mandatory "
                ".rodata/cross-CU/function-coverage search domain. "
                "Choose another study/storage to avoid reusing unconstrained trials."
            )
        if existing is not None and existing != expected:
            raise ValueError(
                f"Optuna study {study_name!r} has incompatible {name}: "
                f"{existing!r} != {expected!r}. Choose another study/storage."
            )
        study.set_user_attr(name, expected)

    if not study.trials:
        default_params = decision_defaults(args.library_score_aggregator)
        baseline = {
            parameter: default_params[parameter]
            for parameter in DECISION_GRIDS
        }
        for profile_id in profile_ids:
            study.enqueue_trial({"profile_id": profile_id, **baseline})

    replay_pairs = prepare_optuna_replay_pairs(
        collections, folds, archive_names, args.ground_truth_unit
    )

    def suggest_trial(trial: Any) -> tuple[str, dict[str, Any]]:
        profile_id = trial.suggest_categorical("profile_id", profile_ids)
        return profile_id, optuna_decision_params(trial)

    def objective_function(trial: Any) -> float:
        profile_id, params = suggest_trial(trial)
        return evaluate_optuna_trial(profile_id, params, replay_pairs)

    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    finished_count = sum(
        trial.state in finished_states for trial in study.get_trials(deepcopy=False)
    )
    remaining_trials = max(0, args.optuna_trials - finished_count)
    if remaining_trials:
        print(
            f"Optuna study {study_name}: {finished_count} finished, "
            f"running up to {remaining_trials} additional trials"
        )
        if args.optuna_workers == 1:
            study.optimize(
                objective_function,
                n_trials=remaining_trials,
                timeout=args.optuna_timeout or None,
                gc_after_trial=False,
                show_progress_bar=sys.stderr.isatty(),
            )
        else:
            global _OPTUNA_REPLAY_CONTEXT
            _OPTUNA_REPLAY_CONTEXT = replay_pairs
            started = time.monotonic()
            submitted = 0
            pending: dict[Any, Any] = {}
            try:
                with ProcessPoolExecutor(
                    max_workers=args.optuna_workers,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    while pending or submitted < remaining_trials:
                        while (
                            len(pending) < args.optuna_workers
                            and submitted < remaining_trials
                            and (
                                not args.optuna_timeout
                                or time.monotonic() - started < args.optuna_timeout
                            )
                        ):
                            trial = study.ask()
                            profile_id, params = suggest_trial(trial)
                            pending[
                                executor.submit(
                                    evaluate_optuna_trial_worker,
                                    profile_id,
                                    params,
                                )
                            ] = trial
                            submitted += 1
                        if not pending:
                            break
                        completed, _ = wait(
                            pending, return_when=FIRST_COMPLETED
                        )
                        for future in completed:
                            trial = pending.pop(future)
                            try:
                                value = future.result()
                            except BaseException:
                                study.tell(trial, state=TrialState.FAIL)
                                raise
                            study.tell(trial, value)
                    print(
                        f"Parallel Optuna replay: {submitted} trials with "
                        f"{args.optuna_workers} workers"
                    )
            finally:
                _OPTUNA_REPLAY_CONTEXT = None
    else:
        print(
            f"Optuna study {study_name}: persistent budget of "
            f"{args.optuna_trials} trials already reached"
        )

    best_trial = study.best_trial
    best_profile_id = str(best_trial.params["profile_id"])
    best_params = decision_defaults(args.library_score_aggregator)
    best_params.update(
        {
            parameter: best_trial.params[parameter]
            for parameter in DECISION_GRIDS
        }
    )
    best_collection = collections[best_profile_id]
    development_metrics, _ = evaluate(
        best_params,
        best_collection,
        development_cases,
        archive_names,
    )
    best_fold_metrics = [
        evaluate(
            best_params,
            best_collection,
            fold["validation_cases"],
            archive_names,
        )[0]
        for fold in folds
    ]
    pooled_cv_metrics = aggregate_metrics(best_fold_metrics)
    mean_cv_metrics = mean_fold_metrics(best_fold_metrics)
    cv_summary = {
        "fold_count": len(folds),
        "objective": "mean_grouped_cv_library_f1",
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "study_name": study.study_name,
        "storage": str(args.optuna_storage),
        "finished_trial_count": sum(
            trial.state in finished_states
            for trial in study.get_trials(deepcopy=False)
        ),
        "mean_metrics": mean_cv_metrics,
        "pooled_metrics": pooled_cv_metrics,
        "lib_f1_std": statistics.pstdev(
            float(metrics["lib_f1"]) for metrics in best_fold_metrics
        ),
        "cu_f1_std": statistics.pstdev(
            float(metrics["cu_f1"]) for metrics in best_fold_metrics
        ),
    }

    history = []
    for trial in study.get_trials(deepcopy=False):
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value if trial.value is not None else "",
            "datetime_start": (
                trial.datetime_start.isoformat() if trial.datetime_start else ""
            ),
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration else ""
            ),
        }
        row.update(
            {
                f"parameter_{name}": trial.params.get(name, "")
                for name in ("profile_id", *DECISION_GRIDS)
            }
        )
        history.append(row)

    complete_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == TrialState.COMPLETE
    ]
    profile_rows = []
    for profile_id in profile_ids:
        candidates = [
            trial
            for trial in complete_trials
            if trial.params.get("profile_id") == profile_id
        ]
        best_for_profile = max(
            candidates,
            key=lambda trial: float(trial.value),
            default=None,
        )
        profile_rows.append(
            {
                **collections[profile_id]["profile"],
                "optuna_complete_trials": len(candidates),
                "cv_mean_lib_f1": (
                    best_for_profile.value if best_for_profile is not None else ""
                ),
                "best_trial": (
                    best_for_profile.number if best_for_profile is not None else ""
                ),
            }
        )

    fold_rows = []
    for fold, metrics in zip(folds, best_fold_metrics):
        fold_rows.append(
            {
                "profile_id": best_profile_id,
                "fold_id": fold["fold_id"],
                "validation_programs": ",".join(fold["validation_programs"]),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    return (
        best_profile_id,
        best_params,
        development_metrics,
        pooled_cv_metrics,
        history,
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
    groups: dict[str, list[str]] = {}
    for archive_name in archive_names:
        family = archive_family(archive_name)
        key = archive_name if cases and "expected_candidates" in cases[0] else family
        groups.setdefault(key, []).append(archive_name)
    for label, candidates in sorted(groups.items()):
        metrics, _ = evaluate(params, collection, cases, candidates)
        rows.append(
            {
                "library": label,
                "family": archive_family(label),
                **metrics,
            }
        )
    return rows


def acquire_output_lock(output_dir: Path):
    """Prevent concurrent threshold searches from writing the same feature files."""
    lock_path = output_dir / ".optuna_threshold_search.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(
            f"Another threshold search is already using {output_dir} "
            f"(PID {owner}). Stop it or choose a different --output-dir."
        ) from error

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    args = parse_args()
    if args.thesis_protocol:
        if args.import_batch_reports_dir is not None or args.extra_high_structural_profiles:
            raise ValueError(
                "--thesis-protocol requires collecting its 24 structural "
                "profiles; batch import and extra profiles are incompatible"
            )
        if args.structural_profile_id:
            raise ValueError("--thesis-protocol requires all 24 structural profiles")
        if args.ground_truth_dir == DEFAULT_GROUND_TRUTH:
            raise ValueError(
                "--thesis-protocol requires --ground-truth-dir pointing to the "
                "separate 1200-ELF development corpus; the default is held-out data"
            )
        args.full_dataset_search = True
        args.search_strategy = "optuna"
        args.validation_mode = "grouped_cv"
        args.ground_truth_unit = "library_family"
        args.decision_search_space = "full"
        args.include_glibc = True
        args.structural_trials = 24
        args.structural_screen_variants_per_program = 2
        args.structural_screen_program_count = 75
        args.structural_screen_archive_count = 720
        args.structural_screen_max_archive_bytes = 0
        args.structural_finalists = 1
        args.staged_final_variants_per_program = 16
        args.staged_final_program_count = 75
        args.staged_structural_search = True
        args.cv_folds = 5
        args.optuna_trials = 3000
    activate_decision_search_space(args.decision_search_space)
    if args.search_only and args.collect_only:
        raise ValueError("--search-only and --collect-only are mutually exclusive")
    if args.no_analysis_cache and args.analysis_cache_only:
        raise ValueError(
            "--analysis-cache-only cannot be combined with --no-analysis-cache"
        )
    if args.import_batch_reports_dir is not None:
        if not args.search_only:
            raise ValueError(
                "--import-batch-reports-dir requires --search-only"
            )
        if args.structural_trials != 1:
            raise ValueError(
                "Decision-only batch import requires --structural-trials 1"
            )
    if args.full_dataset_search:
        conflicting = [
            option
            for option, value in (
                ("--program", args.program),
                ("--validation-program", args.validation_program),
                ("--test-program", args.test_program),
                ("--elf", args.elf_variants),
                ("--compiler", args.compiler),
            )
            if value
        ]
        conflicting.extend(
            option
            for option, value in (
                ("--train-elf-count", args.train_elf_count),
                ("--validation-elf-count", args.validation_elf_count),
                ("--test-elf-count", args.test_elf_count),
            )
            if value is not None
        )
        if conflicting:
            raise ValueError(
                "--full-dataset-search cannot be combined with dataset subset "
                "options: " + ", ".join(conflicting)
            )
        if args.validation_mode != "grouped_cv":
            raise ValueError(
                "--full-dataset-search requires --validation-mode grouped_cv"
            )
        args.all_programs = True
    if args.search_strategy == "optuna" and args.validation_mode != "grouped_cv":
        raise ValueError("--search-strategy optuna requires grouped_cv")
    if args.search_strategy == "optuna":
        args.objective_order = "lib_only"
    if args.staged_structural_search is None:
        args.staged_structural_search = bool(
            args.full_dataset_search and args.search_strategy == "optuna"
        )
    if args.import_batch_reports_dir is not None:
        # A batch import contains one already-computed structural profile.
        # Build the bounded final panel directly and skip structural screening.
        args.staged_structural_search = False
    if args.staged_structural_search and not (
        args.full_dataset_search and args.search_strategy == "optuna"
    ):
        raise ValueError(
            "--staged-structural-search requires --full-dataset-search and "
            "--search-strategy optuna"
        )
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be at least 1")
    if args.optuna_workers < 1:
        raise ValueError("--optuna-workers must be at least 1")
    if args.max_matrix_negative_archive_bytes < 0:
        raise ValueError(
            "--max-matrix-negative-archive-bytes cannot be negative"
        )
    if args.optuna_timeout < 0:
        raise ValueError("--optuna-timeout cannot be negative")
    args.structural_screen_optuna_trials = screen_optuna_trial_budget(
        args.structural_screen_optuna_trials,
        args.structural_trials,
    )
    if args.structural_screen_program_count < 0:
        raise ValueError("--structural-screen-program-count cannot be negative")
    if args.structural_finalists < 1:
        raise ValueError("--structural-finalists must be at least 1")
    if args.staged_final_variants_per_program < 1:
        raise ValueError("--staged-final-variants-per-program must be at least 1")
    if args.staged_final_program_count < 0:
        raise ValueError("--staged-final-program-count cannot be negative")
    if args.collection_workers < 1:
        raise ValueError("--collection-workers must be at least 1")
    args.ground_truth_dir = args.ground_truth_dir.expanduser().resolve()
    if not args.ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {args.ground_truth_dir}"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.analysis_cache_dir = args.analysis_cache_dir.expanduser().resolve()
    args.import_batch_reports_dir = (
        args.import_batch_reports_dir.expanduser().resolve()
        if args.import_batch_reports_dir is not None
        else None
    )
    args.batch_library_matrix = args.batch_library_matrix.expanduser().resolve()
    args.batch_library_root = args.batch_library_root.expanduser().resolve()
    if args.thesis_protocol:
        if not args.batch_library_matrix.is_file():
            raise FileNotFoundError(
                f"Thesis library matrix not found: {args.batch_library_matrix}"
            )
        if not args.batch_library_root.is_dir():
            raise FileNotFoundError(
                f"Thesis library build root not found: {args.batch_library_root}"
            )
    if args.import_batch_reports_dir is not None:
        if not args.import_batch_reports_dir.is_dir():
            raise FileNotFoundError(
                f"Batch reports directory not found: {args.import_batch_reports_dir}"
            )
        if not args.batch_library_matrix.is_file():
            raise FileNotFoundError(
                f"Batch library matrix not found: {args.batch_library_matrix}"
            )
        if not args.batch_library_root.is_dir():
            raise FileNotFoundError(
                f"Batch library root not found: {args.batch_library_root}"
            )
    args.optuna_storage = (
        args.optuna_storage.expanduser().resolve()
        if args.optuna_storage is not None
        else args.output_dir / "optuna_study.sqlite3"
    )
    if args.negative_libs_dir is not None:
        args.negative_libs_dir = args.negative_libs_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_lock = acquire_output_lock(args.output_dir)
    args.lib_roots = resolve_lib_roots(args)

    cases = load_cases(args)
    if args.import_batch_reports_dir is not None:
        before = len(cases)
        cases = complete_batch_feature_cases(
            args.import_batch_reports_dir, cases
        )
        print(
            f"Decision-only source: {len(cases)}/{before} ELF have complete "
            "replay batch features"
        )
    args.cv_folds = args.cv_folds or (5 if args.full_dataset_search else 3)
    if args.full_dataset_search:
        search_cases = sorted(cases, key=lambda case: case["variant"])
        development_cases = search_cases
        train_cases = search_cases
        validation_cases: list[dict[str, Any]] = []
        test_cases: list[dict[str, Any]] = []
        train_programs = {str(case["program"]) for case in search_cases}
        validation_programs: set[str] = set()
        test_programs: set[str] = set()
    else:
        if args.train_elf_count is None:
            selected_program_count = len({case["program"] for case in cases})
            args.train_elf_count = 18 if selected_program_count > 5 else 6
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
    if args.thesis_protocol:
        selected_archives = load_thesis_library_corpus(
            args, archive_selection_cases
        )
        positive_archives = [
            archive for archive in selected_archives
            if archive["kind"] == "positive"
        ]
        negative_archives = [
            archive for archive in selected_archives
            if archive["kind"] == "negative"
        ]
    else:
        positive_archives = select_positive_archives(
            args,
            archive_selection_cases,
            rng,
        )
        negative_archives = select_negative_archives(
            args,
            archive_selection_cases,
            positive_archives,
            rng,
        )
        selected_archives = sorted(
            positive_archives + negative_archives,
            key=lambda item: item["name"],
        )
    if args.ground_truth_unit == "exact_build":
        verify_selected_archive_digests(selected_archives)
    build_diversity = (
        library_build_diversity_report(selected_archives)
        if args.ground_truth_unit == "exact_build"
        else {}
    )
    excluded_ambiguous_builds = (
        ambiguous_exact_builds(archive_selection_cases)
        if args.ground_truth_unit == "exact_build"
        else []
    )
    if args.ground_truth_unit == "exact_build" and args.search_strategy == "optuna":
        require_library_build_diversity(build_diversity)
    configure_candidate_ground_truth(
        cases,
        selected_archives,
        args.ground_truth_unit,
    )
    if args.thesis_protocol:
        optimization_cases = development_cases
    elif args.staged_structural_search or args.import_batch_reports_dir is not None:
        optimization_cases = select_structural_screen_programs(
            select_structural_screen_cases(
                search_cases,
                args.staged_final_variants_per_program,
                args.seed + 202,
            ),
            args.staged_final_program_count,
            args.seed + 303,
        )
    else:
        optimization_cases = development_cases
    if args.include_all_tuning_positive_builds:
        if args.ground_truth_unit != "exact_build":
            raise ValueError(
                "--include-all-tuning-positive-builds requires exact_build"
            )
        pool = unambiguous_exact_archive_pool(
            exact_ground_truth_archive_pool(archive_selection_cases)
        )
        selected_identities = {
            tuple(item["build_identity"]) for item in selected_archives
        }
        tuning_identities = {
            identity
            for case in optimization_cases
            for identity in case.get("expected_builds", set())
            if identity[0] != "libc.a"
        }
        missing_from_pool = tuning_identities - set(pool)
        if missing_from_pool:
            raise ValueError(
                "Linked exact builds are not uniquely pinned in the archive "
                f"catalog: {sorted(missing_from_pool)}"
            )
        additions = [
            selected_exact_archive_record(identity, pool[identity], "positive")
            for identity in sorted(tuning_identities - selected_identities)
        ]
        if additions:
            positive_archives.extend(additions)
            selected_archives = sorted(
                positive_archives + negative_archives,
                key=lambda item: item["name"],
            )
            verify_selected_archive_digests(selected_archives)
            build_diversity = library_build_diversity_report(selected_archives)
            configure_candidate_ground_truth(
                cases, selected_archives, args.ground_truth_unit
            )
            print(
                f"Added {len(additions)} linked exact builds from the "
                "75-ELF tuning panel so omissions count as FN"
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
        ("development",)
        if args.full_dataset_search
        else (
            ("development", "test")
            if args.validation_mode == "grouped_cv"
            else ("train", "validation", "test")
        )
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
    if (
        not args.full_dataset_search
        and split_label_counts["test"]["positive"]
        < args.min_test_positive_labels
    ):
        raise ValueError(
            "Final test contains only "
            f"{split_label_counts['test']['positive']} positive library label(s); "
            f"--min-test-positive-labels={args.min_test_positive_labels}. "
            "Add test programs/variants or lower the minimum only for a smoke test."
        )
    structural_profile_selection = (
        [{"profile_id": "profile_000", **STRUCTURAL_DEFAULTS}]
        if args.import_batch_reports_dir is not None
        else structural_profiles(args.structural_trials, args.seed)
    )
    if args.extra_high_structural_profiles:
        if args.import_batch_reports_dir is not None:
            raise ValueError(
                "--extra-high-structural-profiles cannot be used with batch import"
            )
        structural_profile_selection = append_extra_high_structural_profiles(
            structural_profile_selection
        )
    if args.structural_profile_id:
        requested_profile_ids = set(args.structural_profile_id)
        known_profile_ids = {
            str(profile["profile_id"])
            for profile in structural_profile_selection
        }
        unknown_profile_ids = requested_profile_ids - known_profile_ids
        if unknown_profile_ids:
            raise ValueError(
                "Unknown --structural-profile-id value(s): "
                + ", ".join(sorted(unknown_profile_ids))
            )
        structural_profile_selection = [
            profile
            for profile in structural_profile_selection
            if str(profile["profile_id"]) in requested_profile_ids
        ]
    if args.thesis_protocol:
        structural_screen_cases = select_thesis_screen_cases(
            development_cases, args.seed
        )
    elif args.staged_structural_search:
        structural_screen_cases = select_structural_screen_programs(
            select_structural_screen_cases(
                search_cases,
                args.structural_screen_variants_per_program,
                args.seed,
            ),
            args.structural_screen_program_count,
            args.seed,
        )
    else:
        structural_screen_cases = []
    cv_folds = (
        build_grouped_cv_folds(
            optimization_cases,
            archive_names,
            args.cv_folds,
            args.seed,
        )
        if args.validation_mode == "grouped_cv"
        else []
    )
    if args.thesis_protocol:
        structural_screen_archives = select_thesis_screen_archives(
            selected_archives, structural_screen_cases
        )
    elif args.staged_structural_search:
        structural_screen_archives = select_structural_screen_archives(
            selected_archives,
            args.structural_screen_max_archive_bytes,
            args.structural_screen_archive_count,
            structural_screen_cases,
        )
    else:
        structural_screen_archives = []
    structural_screen_archive_names = [
        archive["name"] for archive in structural_screen_archives
    ]
    cache_only_report = validate_cache_only_inputs(
        args,
        [*structural_screen_cases, *optimization_cases],
        selected_archives,
    )
    structural_screen_folds = (
        build_grouped_cv_folds(
            structural_screen_cases,
            structural_screen_archive_names,
            args.cv_folds,
            args.seed + 101,
        )
        if args.staged_structural_search
        else []
    )
    if args.thesis_protocol:
        if len(structural_profile_selection) != 24:
            raise ValueError("Section 5.3 requires exactly 24 structural profiles")
        if len(optimization_cases) != 1200 or len(structural_screen_cases) != 150:
            raise ValueError(
                "Section 5.3 requires 1200 development ELF and 150 "
                "screening ELF"
            )
        if len(structural_screen_archives) != 720:
            raise ValueError("Section 5.3 requires 720 screening builds")
        for label, current_folds, elf_count in (
            ("full", cv_folds, 240),
            ("screen", structural_screen_folds, 30),
        ):
            if len(current_folds) != 5 or any(
                len(fold["validation_programs"]) != 15
                or len(fold["validation_cases"]) != elf_count
                for fold in current_folds
            ):
                raise ValueError(
                    f"Section 5.3 requires five {label} folds of 15 "
                    f"programs and {elf_count} ELF each"
                )
    input_signature = feature_input_signature(
        args,
        optimization_cases,
        selected_archives,
    )
    selection_report = {
        "thesis_protocol": args.thesis_protocol,
        "ground_truth_unit": args.ground_truth_unit,
        "matching_code_signature": matching_code_signature(),
        "seed": args.seed,
        "input_signature": input_signature,
        "required_decision_components": REQUIRED_DECISION_COMPONENTS,
        "analysis_cache_only": cache_only_report,
        "library_build_diversity": build_diversity,
        "excluded_ambiguous_exact_builds": excluded_ambiguous_builds,
        "full_dataset_search": args.full_dataset_search,
        "staged_structural_search": (
            {
                "enabled": True,
                "variants_per_program": (
                    args.structural_screen_variants_per_program
                ),
                "program_limit": args.structural_screen_program_count,
                "finalist_count": args.structural_finalists,
                "elf_count": len(structural_screen_cases),
                "program_count": len(
                    {case["program"] for case in structural_screen_cases}
                ),
                "max_archive_bytes": (
                    args.structural_screen_max_archive_bytes
                ),
                "archive_names": structural_screen_archive_names,
                "optuna_trials": args.structural_screen_optuna_trials,
                "elf_variants": [
                    case["variant"] for case in structural_screen_cases
                ],
                "final_tuning_variants_per_program": (
                    args.staged_final_variants_per_program
                ),
                "final_tuning_program_limit": (
                    args.staged_final_program_count
                ),
                "final_tuning_elf_count": len(optimization_cases),
                "final_tuning_elf_variants": [
                    case["variant"] for case in optimization_cases
                ],
            }
            if args.staged_structural_search
            else {"enabled": False}
        ),
        "collection_workers": args.collection_workers,
        "search_strategy": args.search_strategy,
        "optuna": (
            {
                "trial_budget": args.optuna_trials,
                "timeout_seconds": args.optuna_timeout,
                "study_name": args.optuna_study_name,
                "storage": str(args.optuna_storage),
                "objective": "mean_grouped_cv_library_f1",
            }
            if args.search_strategy == "optuna"
            else None
        ),
        "positive_count": len(positive_archives),
        "negative_count": len(negative_archives),
        "archive_selection_split": (
            "full_dataset"
            if args.full_dataset_search
            else (
                "development" if args.validation_mode == "grouped_cv" else "train"
            )
        ),
        "validation_mode": args.validation_mode,
        "device": args.device,
        "asm_normalization": args.asm_normalization,
        "palmtree_pooling": args.palmtree_pooling,
        "initial_library_score_aggregator": args.library_score_aggregator,
        "objective_order": args.objective_order,
        "max_positive_archive_bytes": args.max_positive_archive_bytes,
        "lib_roots": [str(root) for root in args.lib_roots],
        "elf_variants": [case["variant"] for case in search_cases],
        "splits": {
            "full_dataset": {
                "assigned_programs": sorted(train_programs),
                "programs": sorted({case["program"] for case in search_cases}),
                "elf_variants": [case["variant"] for case in search_cases],
                "library_labels": split_library_label_counts(
                    search_cases, archive_names
                ),
                "used_for_optimization": (
                    args.full_dataset_search
                    and not args.staged_structural_search
                    and args.import_batch_reports_dir is None
                ),
            },
            "staged_tuning_panel": {
                "programs": sorted(
                    {case["program"] for case in optimization_cases}
                ),
                "elf_variants": [
                    case["variant"] for case in optimization_cases
                ],
                "library_labels": split_library_label_counts(
                    optimization_cases, archive_names
                ),
                "used_for_optimization": (
                    args.staged_structural_search
                    or args.import_batch_reports_dir is not None
                ),
            },
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
    write_csv(
        args.output_dir / "structural_profile_candidates.csv",
        structural_profile_selection,
    )
    exported_splits = (
        (("full_dataset", search_cases),)
        if args.full_dataset_search
        else (
            ("train", train_cases),
            ("validation", validation_cases),
            ("test", test_cases),
        )
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
        for split_name, split_cases in exported_splits
        for case in split_cases
    ]
    write_csv(args.output_dir / "split_cases.csv", split_rows)
    if args.staged_structural_search:
        for filename, panel_cases in (
            ("structural_screen_cases.csv", structural_screen_cases),
            ("tuning_cases.csv", optimization_cases),
        ):
            write_csv(
                args.output_dir / filename,
                [
                    {
                        "variant": case["variant"],
                        "program": case["program"],
                        "compiler": case["compiler"],
                        "elf_compiler_version": case["binary"].parent.parent.name,
                        "elf_optimization": case["elf_optimization"],
                        "binary": str(case["binary"]),
                        "positive_candidate_builds": ",".join(
                            sorted(case.get("expected_candidates", set()))
                        ),
                    }
                    for case in panel_cases
                ],
            )
    if args.ground_truth_unit == "exact_build":
        write_csv(
            args.output_dir / "candidate_builds.csv",
            [
                {
                    "kind": archive["kind"],
                    "candidate": archive["name"],
                    "archive": archive["archive_name"],
                    "source_version": archive["source"],
                    "compiler": archive["compiler"],
                    "optimization": archive["optimization"],
                    "size_bytes": archive["size"],
                    "sha256": archive["sha256"],
                    "provenance": archive["provenance"],
                    "path": archive["path"],
                }
                for archive in selected_archives
            ],
        )
    confirmed_cu_rows = []
    candidate_metadata = {
        str(archive["name"]): archive for archive in selected_archives
    }
    for case in optimization_cases:
        split = (
            "staged_tuning_panel"
            if args.staged_structural_search
            else "full_dataset"
            if args.full_dataset_search
            else "test" if case["program"] in test_programs else "development"
        )
        if args.ground_truth_unit == "exact_build":
            expected_cus = case.get("expected_cus_by_candidate", {})
        else:
            expected_cus = case.get("expected_cus", {})
        for candidate, names in sorted(expected_cus.items()):
            metadata = candidate_metadata.get(candidate, {})
            for name in sorted(names):
                confirmed_cu_rows.append(
                    {
                        "split": split,
                        "variant": case["variant"],
                        "program": case["program"],
                        "candidate": candidate,
                        "library_family": archive_family(candidate),
                        "build_identity": "/".join(
                            metadata.get("build_identity", [])
                        ),
                        "compilation_unit": (
                            name if str(name).endswith(".o") else f"{name}.o"
                        ),
                        "ground_truth_method": "linker_map",
                    }
                )
    write_csv(
        args.output_dir / "confirmed_cu_ground_truth.csv", confirmed_cu_rows
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

    elf_dir, lib_dir = prepare_inputs(args, optimization_cases, selected_archives)
    print(
        f"Selected {len(search_cases)} / {len(cases)} ELF files, "
        f"{len(positive_archives)} positive archives and "
        f"{len(negative_archives)} negative archives"
    )
    if args.full_dataset_search:
        print(
            f"Full search source: {len(search_cases)} ELF files from "
            f"{len(train_programs)} programs; no internal test holdout"
        )
        if args.staged_structural_search:
            print(
                f"Balanced final tuning panel: {len(optimization_cases)} ELF "
                f"files covering "
                f"{len({case['program'] for case in optimization_cases})} "
                "positive-aware programs"
            )
        elif args.import_batch_reports_dir is not None:
            print(
                f"Decision-only tuning panel: {len(optimization_cases)} ELF "
                f"files covering "
                f"{len({case['program'] for case in optimization_cases})} "
                "positive-aware programs"
            )
    else:
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
            f"{len(optimization_cases)} tuning ELF files"
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
    if len(selected_archives) <= 100:
        for archive in selected_archives:
            print(
                f"  [{archive['kind']}] {archive['name']} "
                f"({archive['source']} {archive['variant']}, "
                f"{int(archive['size'])} bytes)"
            )
    else:
        print(
            f"Archive details: {args.output_dir / 'candidate_builds.csv'}"
        )

    final_structural_profiles = structural_profile_selection
    if args.staged_structural_search:
        screen_args = copy.copy(args)
        screen_args.output_dir = args.output_dir / "structural_screen"
        screen_args.reuse_feature_dir = list(args.reuse_feature_dir)
        screen_args.output_dir.mkdir(parents=True, exist_ok=True)
        screen_input_signature = feature_input_signature(
            screen_args,
            structural_screen_cases,
            structural_screen_archives,
        )
        screen_elf_dir, screen_lib_dir = prepare_inputs(
            screen_args,
            structural_screen_cases,
            structural_screen_archives,
        )
        print(
            "Staged structural screen: "
            f"{len(structural_screen_cases)} ELF files from "
            f"{len({case['program'] for case in structural_screen_cases})} "
            f"programs, {len(structural_screen_archives)} bounded archives, "
            f"{len(structural_profile_selection)} profiles, "
            f"{args.collection_workers} workers"
        )
        trials_per_screen_profile = max(
            60,
            (
                args.structural_screen_optuna_trials
                + len(structural_profile_selection)
                - 1
            )
            // len(structural_profile_selection),
        )
        screen_candidates = []
        screen_history = []
        screen_profile_rows = []
        screen_cv_rows = []
        for profile in structural_profile_selection:
            profile_id = str(profile["profile_id"])
            profile_args = copy.copy(screen_args)
            profile_args.optuna_storage = (
                screen_args.output_dir
                / "profile_studies"
                / f"{profile_id}.sqlite3"
            )
            profile_args.optuna_study_name = (
                f"{args.optuna_study_name}_screen_{profile_id}"
                if args.optuna_study_name
                else None
            )
            profile_args.optuna_trials = trials_per_screen_profile
            profile_collections = collect_features(
                profile_args,
                structural_screen_cases,
                screen_elf_dir,
                screen_lib_dir,
                screen_input_signature,
                structural_screen_archives,
                [profile],
            )
            if args.dry_run or args.collect_only:
                del profile_collections
                gc.collect()
                continue
            (
                _profile_best_id,
                _profile_params,
                _profile_train_metrics,
                _profile_validation_metrics,
                profile_history,
                profile_rows,
                profile_cv_rows,
                profile_cv_summary,
            ) = search_all_profiles_optuna(
                profile_args,
                profile_collections,
                structural_screen_cases,
                structural_screen_folds,
                structural_screen_archive_names,
                screen_input_signature,
            )
            candidate = {
                **profile,
                "best_value": float(profile_cv_summary["best_value"]),
                "lib_f1_std": float(profile_cv_summary["lib_f1_std"]),
                "finished_trial_count": int(
                    profile_cv_summary["finished_trial_count"]
                ),
                "study": profile_cv_summary,
            }
            screen_candidates.append(candidate)
            screen_history.extend(
                {"screen_profile_id": profile_id, **row}
                for row in profile_history
            )
            screen_profile_rows.extend(profile_rows)
            screen_cv_rows.extend(profile_cv_rows)
            del profile_collections
            gc.collect()
        if args.dry_run or args.collect_only:
            return 0
        winner = max(
            screen_candidates,
            key=lambda candidate: (
                float(candidate["best_value"]),
                -float(candidate["lib_f1_std"]),
            ),
        )
        screen_best_profile_id = str(winner["profile_id"])
        screen_cv_summary = winner["study"]
        finalists = sorted(
            screen_candidates,
            key=lambda candidate: (
                -float(candidate["best_value"]),
                float(candidate["lib_f1_std"]),
                str(candidate["profile_id"]),
            ),
        )[: args.structural_finalists]
        finalist_ids = {str(candidate["profile_id"]) for candidate in finalists}
        final_structural_profiles = [
            profile
            for profile in structural_profile_selection
            if str(profile["profile_id"]) in finalist_ids
        ]
        if len(final_structural_profiles) != len(finalist_ids):
            raise RuntimeError(
                f"Unknown structural screen finalists: {sorted(finalist_ids)}"
            )
        screen_report = {
            "best_profile_id": screen_best_profile_id,
            "best_profile": next(
                profile
                for profile in final_structural_profiles
                if str(profile["profile_id"]) == screen_best_profile_id
            ),
            "finalist_profile_ids": sorted(finalist_ids),
            "finalists": finalists,
            "elf_count": len(structural_screen_cases),
            "program_count": len(
                {case["program"] for case in structural_screen_cases}
            ),
            "archives": structural_screen_archives,
            "cross_validation": screen_cv_summary,
            "trials_per_profile": trials_per_screen_profile,
            "profile_candidates": screen_candidates,
            "input_signature": screen_input_signature,
        }
        (screen_args.output_dir / "best_profile.json").write_text(
            json.dumps(screen_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(screen_args.output_dir / "optuna_trials.csv", screen_history)
        write_csv(
            screen_args.output_dir / "structural_profiles.csv",
            screen_profile_rows,
        )
        write_csv(
            screen_args.output_dir / "cross_validation_results.csv",
            screen_cv_rows,
        )
        selection_report["staged_structural_search"]["best_profile_id"] = (
            screen_best_profile_id
        )
        selection_report["staged_structural_search"]["best_profile"] = (
            next(
                profile
                for profile in final_structural_profiles
                if str(profile["profile_id"]) == screen_best_profile_id
            )
        )
        selection_report["staged_structural_search"]["finalist_profile_ids"] = (
            sorted(finalist_ids)
        )
        (args.output_dir / "input_selection.json").write_text(
            json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        panel_description = (
            "development ELF files" if args.thesis_protocol else "balanced ELF files"
        )
        print(
            f"Structural screen finalists: {', '.join(sorted(finalist_ids))}; "
            f"collecting on {len(optimization_cases)} {panel_description} and "
            f"{len(selected_archives)} archives"
        )

    collections = collect_features(
        args,
        optimization_cases,
        elf_dir,
        lib_dir,
        input_signature,
        selected_archives,
        final_structural_profiles,
    )
    if args.dry_run or args.collect_only:
        return 0

    cv_result_rows: list[dict[str, Any]] = []
    cv_summary: dict[str, Any] | None = None
    if args.search_strategy == "optuna":
        (
            best_profile_id,
            params,
            train_metrics,
            validation_metrics,
            history,
            profile_rows,
            cv_result_rows,
            cv_summary,
        ) = search_all_profiles_optuna(
            args,
            collections,
            optimization_cases,
            cv_folds,
            archive_names,
            input_signature,
        )
    elif args.validation_mode == "grouped_cv":
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
    evaluation_split = (
        "thesis_development"
        if args.thesis_protocol
        else "staged_unseen_tuning_panel"
        if args.staged_structural_search
        else ("full_dataset" if args.full_dataset_search else "test")
    )
    evaluation_cases = (
        optimization_cases if args.full_dataset_search else test_cases
    )
    evaluation_metrics, predictions = evaluate(
        params, best_collection, evaluation_cases, archive_names
    )
    metadata_by_name = {
        archive["name"]: archive for archive in selected_archives
    }
    for prediction in predictions:
        prediction["split"] = evaluation_split
        metadata = metadata_by_name[prediction["library"]]
        prediction["candidate_kind"] = metadata["kind"]
        prediction["candidate_source"] = metadata["source"]
        prediction["candidate_variant"] = metadata["variant"]
        prediction["candidate_archive"] = metadata.get(
            "archive_name", metadata["name"]
        )
        prediction["candidate_compiler"] = metadata.get(
            "compiler", metadata.get("compiler_root", "")
        )
        prediction["candidate_optimization"] = metadata.get("optimization", "")
        prediction["candidate_build_identity"] = "/".join(
            metadata.get("build_identity", [])
        )

    evaluation_group_rows = grouped_metrics(
        params,
        best_collection,
        evaluation_cases,
        archive_names,
    )
    evaluation_library_rows = metrics_by_library(
        params,
        best_collection,
        evaluation_cases,
        archive_names,
    )
    program_group_names = {case["program"] for case in evaluation_cases}
    program_metric_rows = [
        row for row in evaluation_group_rows if row["group"] in program_group_names
    ]
    evaluation_macro_program_metrics = {
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
        "thesis_protocol": args.thesis_protocol,
        "best_profile_id": best_profile_id,
        "parameters": tuned_parameters,
        "metrics": evaluation_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "evaluation_metrics": evaluation_metrics,
        "evaluation_macro_program_metrics": evaluation_macro_program_metrics,
        "test_metrics": (
            None if args.full_dataset_search else evaluation_metrics
        ),
        "test_macro_program_metrics": (
            None
            if args.full_dataset_search
            else evaluation_macro_program_metrics
        ),
        "full_dataset_metrics": (
            evaluation_metrics
            if args.full_dataset_search and not args.staged_structural_search
            else None
        ),
        "staged_unseen_tuning_metrics": (
            evaluation_metrics
            if args.staged_structural_search and not args.thesis_protocol
            else None
        ),
        "thesis_development_metrics": (
            evaluation_metrics if args.thesis_protocol else None
        ),
        "available_elf_count": len(cases),
        "source_elf_count": len(search_cases),
        "searched_elf_count": len(optimization_cases),
        "train_elf_count": len(optimization_cases),
        "validation_elf_count": len(validation_cases),
        "test_elf_count": len(test_cases),
        "development_elf_count": len(optimization_cases),
        "splits": selection_report["splits"],
        "archive_selection_split": selection_report["archive_selection_split"],
        "validation_mode": args.validation_mode,
        "search_strategy": args.search_strategy,
        "full_dataset_search": args.full_dataset_search,
        "staged_structural_search": selection_report[
            "staged_structural_search"
        ],
        "cross_validation": cv_summary,
        "threshold_tuning_split": (
            "thesis_development_grouped_cv_folds"
            if args.thesis_protocol
            else "balanced_all_program_grouped_cv_folds"
            if args.staged_structural_search
            else "all_grouped_cv_folds"
            if args.search_strategy == "optuna"
            else (
                "grouped_cv_train_folds_then_full_development"
                if args.validation_mode == "grouped_cv"
                else "train"
            )
        ),
        "profile_selection_split": (
            "bounded_archive_structural_screen_grouped_cv"
            if args.staged_structural_search
            else "grouped_cv_validation_folds"
            if args.validation_mode == "grouped_cv"
            else "validation"
        ),
        "final_evaluation_split": evaluation_split,
        "evaluation_is_unbiased_holdout": not args.full_dataset_search,
        "test_thresholds_frozen": not args.full_dataset_search,
        "test_program_count": len({case["program"] for case in test_cases}),
        "test_library_labels": split_label_counts["test"],
        "device": args.device,
        "asm_normalization": args.asm_normalization,
        "palmtree_pooling": args.palmtree_pooling,
        "library_score_aggregator": params["library_score_aggregator"],
        "ground_truth_unit": args.ground_truth_unit,
        "required_decision_components": REQUIRED_DECISION_COMPONENTS,
        "library_build_diversity": build_diversity,
        "matching_code_signature": matching_code_signature(),
        "seed": args.seed,
        "objective_order": args.objective_order,
        "input_signature": input_signature,
        "positive_archives": positive_archives,
        "negative_archives": negative_archives,
        "pipeline": "current block/.rodata matching only",
        "ground_truth_rule": (
            "candidate archive basename, source/version, full compiler and "
            "library optimization all match a build with at least one included "
            "compilation unit in the randomized linker-map ground truth"
            if args.ground_truth_unit == "exact_build"
            else "archive family has at least one included compilation unit "
            "according to the randomized linker-map ground truth"
        ),
        "not_tuned": {
            "asm_model": "model choice, not a matching threshold",
            "asm_normalization": "feature preprocessing, not a matching threshold",
            "palmtree_pooling": "embedding adapter, not a matching threshold",
            "device": "execution device, not matching behavior",
            "timeout": "execution control, not matching behavior",
        },
    }
    (args.output_dir / "best_thresholds.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "search_history.csv", history)
    if args.search_strategy == "optuna":
        write_csv(args.output_dir / "optuna_trials.csv", history)
    else:
        write_csv(args.output_dir / "greedy_history.csv", history)
    write_csv(args.output_dir / "structural_profiles.csv", profile_rows)
    write_csv(args.output_dir / "cross_validation_results.csv", cv_result_rows)
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_csv(
        args.output_dir / "metrics_by_group.csv",
        [{"split": evaluation_split, **row} for row in evaluation_group_rows],
    )
    write_csv(
        args.output_dir / "metrics_by_library.csv",
        [{"split": evaluation_split, **row} for row in evaluation_library_rows],
    )
    if not args.full_dataset_search:
        write_csv(
            args.output_dir / "test_metrics_by_library.csv",
            [{"split": "test", **row} for row in evaluation_library_rows],
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
    evaluation_label = (
        "Full dataset (fitted)" if args.full_dataset_search else "Test"
    )
    print(
        f"{evaluation_label} CU_F1={float(evaluation_metrics['cu_f1']):.4f} "
        f"CU_precision={float(evaluation_metrics['cu_precision']):.4f} "
        f"lib_F1={float(evaluation_metrics['lib_f1']):.4f}"
    )
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
