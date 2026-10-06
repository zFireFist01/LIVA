#!/usr/bin/env python3
"""Compare the current block pipeline with libseeker's function matching."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from match import LIBRARY_SCORE_AGGREGATORS
from analysis_cache import DEFAULT_CACHE_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR / "Test"
REPO_ROOT = SCRIPT_DIR.parent
CURRENT_PIPELINE_DIR = SCRIPT_DIR
LIBSEEKER_PIPELINE_DIR = REPO_ROOT / "FunctionMatching" / "libseeker" / "thesis_code"
DEFAULT_DATASET_DIR = REPO_ROOT / "Dataset" / "datasets" / "libseeker" / "binaries"
DEFAULT_LIBS_DIR = REPO_ROOT / "Exploration" / "libseeker_repo" / "build_lib" / "libs"
DEFAULT_OUTPUT_DIR = TEST_DIR / "libseeker_batch_results"

AUTO_JOB_MEMORY_BYTES = 1536 * 1024**2
AUTO_JOB_MEMORY_FRACTION = 0.95
AUTO_JOB_MAX_WORKERS = 15
ADAPTIVE_MEMORY_RESERVE_MIN_BYTES = 2 * 1024**3
ADAPTIVE_WORKER_ESTIMATE_MAX_BYTES = 4 * 1024**3
ADAPTIVE_MEMORY_POLL_SECONDS = 2.0
ADAPTIVE_SCALE_UP_SECONDS = 60.0
ADAPTIVE_RETRY_RETURN_CODE = -1000
THREAD_LIMIT_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

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


def parse_jobs(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        jobs = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "jobs must be 'auto' or a positive integer"
        ) from error
    if jobs < 1:
        raise argparse.ArgumentTypeError(
            "jobs must be 'auto' or a positive integer"
        )
    return jobs


def parse_positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def available_memory_bytes() -> int | None:
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size) * int(available_pages)
    except (OSError, ValueError, TypeError):
        return None


def total_memory_bytes() -> int | None:
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_bytes(pid: int) -> int:
    try:
        resident_pages = int(
            Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()[1]
        )
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, TypeError):
        return 0


def portable_cache_is_complete(cache_dir: Path) -> bool:
    inventory_path = cache_dir / "shard_inventory.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        inventory.get("valid") is True
        and inventory.get("errors") == []
        and inventory.get("full_library_catalog") is True
        and inventory.get("experiment_cache_records")
        == inventory.get("elf_records")
        and inventory.get("library_archives_cached")
        == inventory.get("selected_library_archives_total")
    )


def automatic_job_count(
    elf_count: int,
    cache_dir: Path,
    *,
    cpu_count: int | None = None,
    memory_bytes: int | None = None,
    max_workers: int = AUTO_JOB_MAX_WORKERS,
) -> tuple[int, str]:
    cpus = max(1, cpu_count if cpu_count is not None else available_cpu_count())
    available = (
        memory_bytes if memory_bytes is not None else available_memory_bytes()
    )
    cache_complete = portable_cache_is_complete(cache_dir)

    if not cache_complete:
        return 1, "cache inventory missing or incomplete"

    memory_jobs = cpus
    memory_description = "unknown"
    if available is not None:
        usable = int(max(0, available) * AUTO_JOB_MEMORY_FRACTION)
        memory_jobs = max(1, usable // AUTO_JOB_MEMORY_BYTES)
        memory_description = f"{available / 1024**3:.1f} GiB available"

    jobs = max(
        1,
        min(max(1, elf_count), cpus, memory_jobs, max(1, max_workers)),
    )
    reason = (
        f"{cpus} available CPUs, {memory_description}, "
        f"{AUTO_JOB_MEMORY_BYTES / 1024**3:.1f} GiB/worker, "
        f"adaptive ceiling {max(1, max_workers)}, validated cache"
    )
    return jobs, reason


def matching_worker_environment(worker_count: int) -> tuple[dict[str, str], int]:
    threads_per_worker = max(1, available_cpu_count() // max(1, worker_count))
    environment = os.environ.copy()
    for name in THREAD_LIMIT_ENVIRONMENT:
        environment[name] = str(threads_per_worker)
    return environment, threads_per_worker


class AdaptiveJobController:
    """Admission control and low-memory recovery for independent ELF runs.

    A fixed-size thread pool supplies the maximum CPU concurrency.  This
    controller decides how many subprocesses may actually be resident.  Under
    memory pressure it stops admitting work and asks the largest resident
    worker to retry later, after reducing the concurrency limit.  Once memory
    has remained healthy, it raises that limit one worker at a time.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        initial_workers: int | None = None,
        memory_reader=available_memory_bytes,
        rss_reader=process_rss_bytes,
        total_memory: int | None = None,
        poll_seconds: float = ADAPTIVE_MEMORY_POLL_SECONDS,
        scale_up_seconds: float = ADAPTIVE_SCALE_UP_SECONDS,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self._memory_reader = memory_reader
        self._rss_reader = rss_reader
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._scale_up_seconds = max(0.0, float(scale_up_seconds))
        detected_total = (
            total_memory if total_memory is not None else total_memory_bytes()
        )
        fractional_reserve = (
            int(detected_total * (1.0 - AUTO_JOB_MEMORY_FRACTION))
            if detected_total is not None
            else 0
        )
        self.memory_reserve_bytes = max(
            ADAPTIVE_MEMORY_RESERVE_MIN_BYTES,
            fractional_reserve,
        )
        self._condition = threading.Condition()
        self._slots: dict[int, dict[str, object]] = {}
        self._next_token = 1
        self._limit = max(
            1,
            min(
                self.max_workers,
                self.max_workers if initial_workers is None else initial_workers,
            ),
        )
        self._last_pressure = 0.0
        self._last_scale_up = (
            time.monotonic() if self._limit < self.max_workers else 0.0
        )
        self._peak_samples: list[int] = []

    @property
    def concurrency_limit(self) -> int:
        with self._condition:
            return self._limit

    @property
    def active_workers(self) -> int:
        with self._condition:
            return len(self._slots)

    def _worker_estimate_locked(self) -> int:
        if not self._peak_samples:
            return AUTO_JOB_MEMORY_BYTES
        ordered = sorted(self._peak_samples[-64:])
        percentile_index = max(0, ((len(ordered) * 4 + 4) // 5) - 1)
        observed = int(ordered[min(percentile_index, len(ordered) - 1)] * 1.15)
        return min(
            ADAPTIVE_WORKER_ESTIMATE_MAX_BYTES,
            max(AUTO_JOB_MEMORY_BYTES, observed),
        )

    def _refresh_rss_locked(self) -> None:
        for slot in self._slots.values():
            pid = slot.get("pid")
            if not isinstance(pid, int):
                continue
            rss = max(0, int(self._rss_reader(pid)))
            slot["rss"] = rss
            slot["peak"] = max(int(slot.get("peak", 0)), rss)

    def _memory_allows_start_locked(self) -> bool:
        available = self._memory_reader()
        if available is None:
            return True
        self._refresh_rss_locked()
        estimate = self._worker_estimate_locked()
        # Budget the observed cost of the worker being admitted, but do not
        # reserve the worst-case growth of every already-running worker. The
        # two-second pressure monitor remains responsible for shedding the
        # largest worker if MemAvailable falls below the hard reserve.
        return available >= self.memory_reserve_bytes + estimate

    def _maybe_scale_up_locked(self) -> None:
        if self._limit >= self.max_workers or any(
            bool(slot.get("abort")) for slot in self._slots.values()
        ):
            return
        now = time.monotonic()
        stable_since = max(self._last_pressure, self._last_scale_up)
        if now - stable_since < self._scale_up_seconds:
            return
        if not self._memory_allows_start_locked():
            return
        self._limit += 1
        self._last_scale_up = now
        print(
            f"[adaptive] memory stable; concurrency raised to {self._limit}",
            flush=True,
        )

    def acquire(self, label: str) -> int:
        with self._condition:
            while True:
                self._maybe_scale_up_locked()
                if (
                    len(self._slots) < self._limit
                    and self._memory_allows_start_locked()
                ):
                    token = self._next_token
                    self._next_token += 1
                    self._slots[token] = {
                        "label": label,
                        "pid": None,
                        "rss": 0,
                        "peak": 0,
                        "abort": False,
                    }
                    return token
                self._condition.wait(timeout=self._poll_seconds)

    def register_process(self, token: int, pid: int) -> None:
        with self._condition:
            if token in self._slots:
                self._slots[token]["pid"] = pid
                self._refresh_rss_locked()

    def inspect(self, token: int) -> bool:
        """Return True when this worker should be terminated and retried."""
        with self._condition:
            slot = self._slots.get(token)
            if slot is None:
                return False
            self._refresh_rss_locked()
            available = self._memory_reader()
            aborting = any(
                bool(candidate.get("abort")) for candidate in self._slots.values()
            )
            if (
                available is not None
                and available < self.memory_reserve_bytes
                and len(self._slots) > 1
                and not aborting
            ):
                candidates = [
                    (candidate_token, candidate)
                    for candidate_token, candidate in self._slots.items()
                    if isinstance(candidate.get("pid"), int)
                ]
                if candidates:
                    victim_token, victim = max(
                        candidates,
                        key=lambda item: int(item[1].get("rss", 0)),
                    )
                    victim["abort"] = True
                    self._limit = max(
                        1,
                        min(self._limit - 1, len(self._slots) - 1),
                    )
                    self._last_pressure = time.monotonic()
                    print(
                        "[adaptive] low memory "
                        f"({available / 1024**3:.1f} GiB available); "
                        f"retrying {victim.get('label')} later and reducing "
                        f"concurrency to {self._limit}",
                        flush=True,
                    )
                    if victim_token == token:
                        slot = victim
            return bool(slot.get("abort"))

    def release(self, token: int) -> None:
        with self._condition:
            slot = self._slots.pop(token, None)
            if slot is not None:
                peak = int(slot.get("peak", 0))
                if peak > 0:
                    self._peak_samples.append(peak)
            self._condition.notify_all()

SUMMARY_FIELDS = [
    "pipeline",
    "binary",
    "binary_path",
    "library",
    "asm_normalization",
    "palmtree_pooling",
    "library_score_aggregator",
    "library_min_score",
    "status",
    "score",
    "base_score",
    "block_best",
    "block_best_any",
    "block_best_matched",
    "cross_cu_matched_edges",
    "cross_cu_evaluable_edges",
    "cross_cu_expected_edges",
    "cross_cu_ratio",
    "cross_cu_coverage",
    "cross_cu_anchored_matched_edges",
    "cross_cu_anchored_evaluable_edges",
    "cross_cu_anchored_expected_edges",
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
    "binary_path",
    "library",
    "outcome",
    "current_asm_normalization",
    "current_palmtree_pooling",
    "current_library_score_aggregator",
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

GROUND_TRUTH_FIELDS = [
    "pipeline", "asm_normalization", "palmtree_pooling",
    "library_score_aggregator", "binary",
    "binary_path", "library", "library_family",
    "ground_truth_variant", "expected_present", "predicted_present",
    "classification", "correct", "pipeline_status", "pipeline_score",
    "block_best", "rodata_best", "matched_cu", "total_cu",
    "ground_truth_source", "ground_truth_optimization",
    "ground_truth_included_cu", "ground_truth_total_cu",
]

GROUND_TRUTH_METRIC_FIELDS = [
    "pipeline", "asm_normalization", "palmtree_pooling",
    "library_score_aggregator", "binary",
    "evaluated", "missing", "tp", "tn", "fp", "fn",
    "accuracy", "precision", "recall", "specificity", "f1",
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
    parser.add_argument(
        "--library-matrix",
        type=Path,
        help="Use every status=selected archive in library_matrix.tsv.",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        help="Override the root used to resolve --library-matrix paths.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Dataset manifest used to add portable ELF provenance.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--result-log",
        type=Path,
        help=(
            "Portable gzip JSONL result bundle. Defaults to "
            "<output-dir>/results.jsonl.gz."
        ),
    )
    parser.add_argument(
        "--include-rejected-cu",
        action="store_true",
        help=(
            "Include rejected CU diagnostics in the portable result log. "
            "Accepted CU/function mappings are always included."
        ),
    )
    parser.add_argument(
        "--offline-ablation-features",
        action="store_true",
        help=(
            "Current pipeline only: retain replay-complete evidence for every "
            "B-qualified CU so B/H/S/X/R ablations can run after the batch."
        ),
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        help=(
            "Directory containing ground_truth.json files. Generates "
            "ground_truth_scores.csv and ground_truth_metrics.csv."
        ),
    )
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
        help=(
            "Skip a run only when its complete outputs exist and the current "
            "pipeline report uses the requested library score threshold and "
            "PalmTree preprocessing configuration."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-run timeout in seconds. 0 disables subprocess timeout.",
    )
    parser.add_argument(
        "--jobs",
        type=parse_jobs,
        default="auto",
        help=(
            "Number of independent ELF subprocesses to run concurrently, or "
            "'auto'. Auto uses available CPUs and RAM when the packaged cache "
            "inventory is complete, and otherwise falls back to 1; default: auto."
        ),
    )
    parser.add_argument(
        "--adaptive-max-jobs",
        type=parse_positive_integer,
        default=AUTO_JOB_MAX_WORKERS,
        help=(
            "Maximum concurrent ELF subprocesses when --jobs=auto. The actual "
            "count is continuously adjusted for memory pressure; default: "
            f"{AUTO_JOB_MAX_WORKERS}."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for both pipelines. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for the current PalmTree pipeline: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--asm-normalization",
        choices=("legacy", "v2"),
        default="v2",
        help=(
            "Current pipeline only: assembly preprocessing version. "
            "Defaults to stripped-safe v2."
        ),
    )
    parser.add_argument(
        "--palmtree-pooling",
        choices=("mean", "masked_mean"),
        default="masked_mean",
        help=(
            "Current pipeline only: masked_mean excludes padding; mean "
            "reproduces the original PalmTree adapter."
        ),
    )
    parser.add_argument(
        "--analysis-cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Persistent radare2 + PalmTree cache forwarded to the current pipeline.",
    )
    parser.add_argument(
        "--no-analysis-cache",
        action="store_true",
        help="Disable the persistent cache in the current pipeline.",
    )
    parser.add_argument(
        "--analysis-cache-only",
        action="store_true",
        help=(
            "Use only the validated packaged cache, never invoke radare2 or "
            "PalmTree. A failed ELF is reported without stopping the remaining "
            "batch."
        ),
    )
    parser.add_argument(
        "--min-cu",
        type=int,
        default=1,
        help="Legacy libseeker baseline only: minimum matched compilation units.",
    )
    parser.add_argument(
        "--min-functions-cu",
        type=int,
        default=1,
        help="Legacy libseeker baseline only: minimum matched functions per CU.",
    )
    parser.add_argument(
        "--libseeker-threshold",
        type=float,
        default=0.80,
        help="CU similarity threshold used by the libseeker function-matching baseline.",
    )
    parser.add_argument(
        "--library-score-aggregator",
        choices=LIBRARY_SCORE_AGGREGATORS,
        default="top3_noisy_or",
        help="Current pipeline only: CU-to-library score aggregation.",
    )
    parser.add_argument(
        "--library-min-score",
        type=float,
        default=0.975,
        help=(
            "Current pipeline only: minimum aggregated score of accepted CUs "
            "required for a positive library decision."
        ),
    )
    parser.add_argument("--cross-cu-call-bonus-weight", type=float, default=0.20)
    parser.add_argument("--cross-cu-call-penalty-weight", type=float, default=0.0)
    parser.add_argument("--cross-cu-call-saturation-edges", type=int, default=8)
    parser.add_argument("--block-threshold", type=float, default=0.90)
    parser.add_argument("--block-coverage-mean-threshold", type=float, default=0.975)
    parser.add_argument(
        "--block-assignment-quality-threshold",
        "--block-assignment-threshold",
        dest="block_assignment_quality_threshold",
        type=float,
        default=0.875,
    )
    parser.add_argument("--block-min-assignment-ratio", type=float, default=0.40)
    parser.add_argument("--block-min-coverage-ratio", type=float, default=0.65)
    parser.add_argument("--block-locality-window-multiplier", type=float, default=2.5)
    parser.add_argument("--block-locality-window-padding", type=int, default=6)
    parser.add_argument(
        "--block-min-call-edge-ratio",
        "--block-min-edge-locality-ratio",
        dest="block_min_call_edge_ratio",
        type=float,
        default=0.30,
    )
    parser.add_argument("--block-min-instructions", type=int, default=5)
    parser.add_argument("--block-min-function-concentration", type=float, default=0.75)
    parser.add_argument("--block-min-function-spread", type=float, default=0.30)
    parser.add_argument(
        "--cu-min-function-coverage",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of reference-CU functions represented by the "
            "selected block mapping; 0 disables the online gate."
        ),
    )
    parser.add_argument("--disable-block-window-prefilter", action="store_true")
    parser.add_argument("--fast-negative-bound", action="store_true")
    rodata_filter_group = parser.add_mutually_exclusive_group()
    rodata_filter_group.add_argument(
        "--disable-rodata-filter",
        dest="disable_rodata_filter",
        action="store_true",
        help="Disable .rodata filtering.",
    )
    rodata_filter_group.add_argument(
        "--enable-rodata-filter",
        dest="disable_rodata_filter",
        action="store_false",
        help="Enable .rodata filtering.",
    )
    parser.set_defaults(disable_rodata_filter=False)
    parser.add_argument("--rodata-min-bytes", type=int, default=512)
    parser.add_argument("--rodata-min-strings", type=int, default=16)
    parser.add_argument("--rodata-min-ngrams", type=int, default=32)
    parser.add_argument("--rodata-penalty-threshold", type=float, default=0.50)
    parser.add_argument("--rodata-confirm-threshold", type=float, default=0.90)
    parser.add_argument("--rodata-bonus-weight", type=float, default=0.30)
    parser.add_argument("--rodata-string-weight", type=float, default=0.70)
    parser.add_argument("--rodata-byte-only-weight", type=float, default=0.50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected commands without running them.",
    )
    args = parser.parse_args()
    if args.no_analysis_cache and args.analysis_cache_only:
        parser.error("--analysis-cache-only cannot be used with --no-analysis-cache")
    if args.analysis_cache_only and args.pipeline == "libseeker":
        parser.error("--analysis-cache-only requires the current pipeline")
    if not 0.0 <= args.cu_min_function_coverage <= 1.0:
        parser.error("--cu-min-function-coverage must be in [0, 1]")
    if not 0.0 <= args.rodata_string_weight <= 1.0:
        parser.error("--rodata-string-weight must be in [0, 1]")
    if not 0.0 <= args.rodata_byte_only_weight <= 1.0:
        parser.error("--rodata-byte-only-weight must be in [0, 1]")
    return args


def resolve_patterns(base_dir: Path, patterns: list[str], default_patterns: list[str]) -> list[Path]:
    selected: list[Path] = []
    seen = set()

    for pattern in patterns or default_patterns:
        matches = sorted(base_dir.glob(pattern))
        if not matches:
            matches = sorted(base_dir.rglob(pattern))
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


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def select_elfs(args: argparse.Namespace) -> list[Path]:
    if args.all_elfs:
        elfs = sorted(
            path.resolve()
            for path in args.dataset_dir.rglob("*")
            if path.is_file() and is_elf(path)
        )
    else:
        elfs = resolve_patterns(args.dataset_dir, args.elf, DEFAULT_ELF_PATTERNS)

    if args.elf_limit > 0:
        elfs = elfs[: args.elf_limit]

    return elfs


def select_libs(args: argparse.Namespace) -> list[Path]:
    if args.library_matrix:
        matrix = args.library_matrix.expanduser().resolve()
        build_root = (
            args.library_root.expanduser().resolve()
            if args.library_root
            else matrix.parents[1] / "builds/libraries"
        )
        selected: set[Path] = set()
        metadata: dict[str, dict[str, str]] = {}
        with matrix.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row.get("status") != "selected" or not row.get("path"):
                    continue
                path = (build_root / row["path"]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Selected library is missing: {path}")
                selected.add(path)
                metadata[path.as_posix()] = {
                    "archive": row.get("archive", path.name),
                    "package": row.get("package", ""),
                    "role": row.get("role", ""),
                    "toolchain": row.get("toolchain", ""),
                    "optimization": row.get("optimization", ""),
                    "project_candidates": row.get("project_candidates", ""),
                    "dataset_path": (Path("Dataset/builds/libraries") / row["path"]).as_posix(),
                }
        args.library_metadata = metadata
        return sorted(selected)
    if args.all_libs:
        # Exact-build searches stage multiple builds of the same archive under
        # collision-free .a.<identity-hash> aliases. Keep the staged path:
        # resolving symlinks here would discard the alias before main.py sees
        # it, and the old *.a glob omitted such candidates entirely.
        libs = sorted(
            path.absolute()
            for path in args.libs_dir.iterdir()
            if path.is_file() and re.fullmatch(r".*\.a(?:\..+)?", path.name)
        )
    else:
        libs = resolve_patterns(args.libs_dir, args.lib, DEFAULT_LIBRARY_NAMES)

    if args.lib_limit > 0:
        libs = libs[: args.lib_limit]

    return libs


def stable_library_labels(
    libraries: list[Path], metadata: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Return collision-free names that remain recognizable as .a files."""
    counts: dict[str, int] = {}
    for path in libraries:
        counts[path.name] = counts.get(path.name, 0) + 1
    labels: dict[str, str] = {}
    used: set[str] = set()
    for path in libraries:
        details = metadata.get(path.as_posix(), {})
        identity = details.get("dataset_path", path.as_posix())
        if counts[path.name] == 1:
            label = path.name
        else:
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            label = f"{path.name}.{suffix}"
        if label in used:
            raise ValueError(f"library staging label collision: {label}")
        labels[path.as_posix()] = label
        used.add(label)
    return labels


def safe_name(value: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def binary_label(binary: Path, dataset_dir: Path) -> str:
    try:
        relative = binary.resolve().relative_to(dataset_dir.resolve())
        return safe_name(relative.as_posix())
    except ValueError:
        return safe_name(binary.resolve())


def make_libraries_dir(
    libraries: list[Path], labels: dict[str, str] | None = None
) -> tempfile.TemporaryDirectory:
    tmpdir = tempfile.TemporaryDirectory(prefix="libseeker-batch-libs.")
    tmp_path = Path(tmpdir.name)

    for library in libraries:
        link_path = tmp_path / (labels or {}).get(
            library.as_posix(), library.name
        )
        if link_path.exists() or link_path.is_symlink():
            tmpdir.cleanup()
            raise FileExistsError(f"duplicate staged library: {link_path.name}")
        try:
            link_path.symlink_to(library)
        except OSError:
            shutil.copy2(library, link_path)

    return tmpdir


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout: int,
    dry_run: bool,
    environment: dict[str, str] | None = None,
    adaptive_controller: AdaptiveJobController | None = None,
    adaptive_token: int | None = None,
) -> int:
    print(" ".join(command))
    if dry_run:
        return 0

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()
        if adaptive_controller is not None and adaptive_token is not None:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            adaptive_controller.register_process(adaptive_token, process.pid)
            deadline = started + timeout if timeout > 0 else None
            while True:
                wait_seconds = ADAPTIVE_MEMORY_POLL_SECONDS
                if deadline is not None:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.05, deadline - time.time()),
                    )
                try:
                    return process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    pass

                if deadline is not None and time.time() >= deadline:
                    _terminate_process_group(process)
                    elapsed = time.strftime(
                        "%H:%M:%S", time.gmtime(time.time() - started)
                    )
                    log_file.write(f"\n[TIMEOUT] after {elapsed}\n")
                    return 124

                if adaptive_controller.inspect(adaptive_token):
                    log_file.write(
                        "\n[ADAPTIVE-RETRY] worker stopped before system OOM\n"
                    )
                    log_file.flush()
                    _terminate_process_group(process)
                    return ADAPTIVE_RETRY_RETURN_CODE

        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
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


def _terminate_process_group(
    process: subprocess.Popen,
    grace_seconds: float = 10.0,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def current_command(
    args: argparse.Namespace,
    binary: Path,
    libraries_dir: Path,
    output_path: Path,
    features_output_path: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        "main.py",
        "--path_to_binary",
        binary.as_posix(),
        "--libraries_dir",
        libraries_dir.as_posix(),
        "--output",
        output_path.as_posix(),
        "--device",
        args.device,
        "--asm_normalization",
        args.asm_normalization,
        "--palmtree_pooling",
        args.palmtree_pooling,
    ]
    if args.no_analysis_cache:
        command.append("--no_analysis_cache")
    else:
        command.extend(
            ["--analysis_cache_dir", args.analysis_cache_dir.as_posix()]
        )
        if args.analysis_cache_only:
            command.append("--analysis_cache_only")
    if features_output_path is not None:
        command.extend(["--features_output", features_output_path.as_posix()])
    if args.offline_ablation_features:
        command.append("--offline_ablation_features")

    command.extend(
        [
            "--block_threshold",
            str(args.block_threshold),
            "--library_min_score",
            str(args.library_min_score),
            "--library_score_aggregator",
            args.library_score_aggregator,
            "--cross_cu_call_bonus_weight",
            str(args.cross_cu_call_bonus_weight),
            "--cross_cu_call_penalty_weight",
            str(args.cross_cu_call_penalty_weight),
            "--cross_cu_call_saturation_edges",
            str(args.cross_cu_call_saturation_edges),
            "--block_assignment_quality_threshold",
            str(args.block_assignment_quality_threshold),
            "--block_min_assignment_ratio",
            str(args.block_min_assignment_ratio),
            "--block_min_coverage_ratio",
            str(args.block_min_coverage_ratio),
            "--block_locality_window_multiplier",
            str(args.block_locality_window_multiplier),
            "--block_locality_window_padding",
            str(args.block_locality_window_padding),
            "--block_min_call_edge_ratio",
            str(args.block_min_call_edge_ratio),
            "--block_min_instructions",
            str(args.block_min_instructions),
            "--block_min_function_concentration",
            str(args.block_min_function_concentration),
            "--block_min_function_spread",
            str(args.block_min_function_spread),
            "--cu_min_function_coverage",
            str(args.cu_min_function_coverage),
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
            "--rodata_bonus_weight",
            str(args.rodata_bonus_weight),
            "--rodata_string_weight",
            str(args.rodata_string_weight),
            "--rodata_byte_only_weight",
            str(args.rodata_byte_only_weight),
        ]
    )
    if args.disable_rodata_filter:
        command.append("--disable_rodata_filter")
    if args.disable_block_window_prefilter:
        command.append("--disable_block_window_prefilter")
    if args.fast_negative_bound:
        command.append("--fast_negative_bound")
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


def parse_edge_triplet(raw: str | None) -> tuple[str, str, str]:
    if raw is None:
        return "", "", ""
    parts = raw.split("/")
    if len(parts) != 3:
        return raw, "", ""
    return parts[0], parts[1], parts[2]


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


def parse_library_min_score(report_path: Path) -> float:
    """Read the current-pipeline library threshold; old reports imply 0.0."""
    if not report_path.exists():
        return 0.0

    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^Library minimum score:\s*"
        r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
        r"\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(matches[-1]) if matches else 0.0


def parse_float_setting(report_path: Path, label: str) -> float | None:
    """Read a numeric provenance setting, returning None for old reports."""
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        rf"^{re.escape(label)}:\s*"
        r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
        r"\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(matches[-1]) if matches else None


def parse_asm_normalization(report_path: Path) -> str:
    """Read preprocessing version; reports created before v2 are legacy."""
    if not report_path.exists():
        return "legacy"

    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^Assembly normalization:\s*(legacy|v2)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "legacy"


def parse_palmtree_pooling(report_path: Path) -> str:
    """Read pooling provenance; reports created before this option used mean."""
    if not report_path.exists():
        return "mean"

    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^PalmTree pooling:\s*(mean|masked_mean)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "mean"


def parse_library_score_aggregator(report_path: Path) -> str:
    """Read CU aggregation provenance; reports created before it used mean."""
    if not report_path.exists():
        return "mean"

    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^Library score aggregator:\s*([a-z0-9_]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "mean"


def parse_feature_collection_mode(report_path: Path) -> str:
    """Read feature detail provenance; old reports used standard records."""
    if not report_path.exists():
        return "standard"
    text = report_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^Feature collection mode:\s*(standard|replay_complete)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else "standard"


def parse_report(
    pipeline: str,
    report_path: Path,
    binary: Path,
    label: str,
    log_path: Path,
) -> list[dict[str, str]]:
    if not report_path.exists():
        return []

    rows = []
    total_time = parse_total_time(log_path)
    asm_normalization = (
        parse_asm_normalization(report_path) if pipeline == "current" else ""
    )
    palmtree_pooling = (
        parse_palmtree_pooling(report_path) if pipeline == "current" else ""
    )
    library_score_aggregator = (
        parse_library_score_aggregator(report_path)
        if pipeline == "current"
        else ""
    )
    library_min_score = (
        str(parse_library_min_score(report_path)) if pipeline == "current" else ""
    )
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
        cross_matched, cross_evaluable, cross_expected = parse_edge_triplet(
            values.get("cross_cu_edges")
        )
        anchored_matched, anchored_evaluable, anchored_expected = (
            parse_edge_triplet(values.get("cross_cu_anchored_edges"))
        )
        row = {
            "pipeline": pipeline,
            "binary": label,
            "binary_path": str(binary.resolve()),
            "library": values.get("library", ""),
            "asm_normalization": asm_normalization,
            "palmtree_pooling": palmtree_pooling,
            "library_score_aggregator": library_score_aggregator,
            "library_min_score": library_min_score,
            "status": match.group("status").strip(),
            "score": parse_percent(values.get("score")),
            "base_score": parse_percent(values.get("base_score")),
            "block_best": parse_percent(
                values.get(
                    "block_best",
                    values.get("block_best_any", values.get("block_best_matched")),
                )
            ),
            "block_best_any": parse_percent(values.get("block_best_any")),
            "block_best_matched": parse_percent(
                values.get("block_best_matched")
            ),
            "cross_cu_matched_edges": cross_matched,
            "cross_cu_evaluable_edges": cross_evaluable,
            "cross_cu_expected_edges": cross_expected,
            "cross_cu_ratio": values.get("cross_cu_ratio", ""),
            "cross_cu_coverage": values.get("cross_cu_coverage", ""),
            "cross_cu_anchored_matched_edges": anchored_matched,
            "cross_cu_anchored_evaluable_edges": anchored_evaluable,
            "cross_cu_anchored_expected_edges": anchored_expected,
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
        (row["pipeline"], row["binary_path"], row["library"]): row
        for row in summary_rows
    }
    cases = sorted(
        (row["binary"], row["binary_path"], row["library"])
        for row in summary_rows
        if row["pipeline"] == "current"
    )
    cases = sorted(
        set(cases)
        | {
            (row["binary"], row["binary_path"], row["library"])
            for row in summary_rows
            if row["pipeline"] == "libseeker"
        }
    )

    comparison_rows = []
    for binary, binary_path, library in cases:
        current_row = indexed_rows.get(("current", binary_path, library))
        libseeker_row = indexed_rows.get(("libseeker", binary_path, library))
        comparison_rows.append(
            {
                "binary": binary,
                "binary_path": binary_path,
                "library": library,
                "outcome": detection_outcome(current_row, libseeker_row),
                "current_asm_normalization": (
                    current_row["asm_normalization"] if current_row else ""
                ),
                "current_palmtree_pooling": (
                    current_row["palmtree_pooling"] if current_row else ""
                ),
                "current_library_score_aggregator": (
                    current_row["library_score_aggregator"] if current_row else ""
                ),
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


def library_family(name: str) -> str:
    lowered = Path(name).name.lower()
    patterns = (
        (r"^libc\.a(?:\.|$)", "glibc"),
        (r"^libpcre2-8\.a(?:\.|$)", "pcre2"),
        (r"^libpcre2-posix\.a(?:\.|$)", "pcre2-posix"),
        (r"^libiconv\.a(?:\.|$)", "iconv"),
        (r"^libcharset\.a(?:\.|$)", "charset"),
        (r"^libgcc_eh\.a(?:\.|$)", "libgcc_eh"),
        (r"^libgcc\.a(?:\.|$)", "libgcc"),
    )
    for pattern, family in patterns:
        if re.match(pattern, lowered):
            return family
    match = re.match(r"^(lib[^.]+)", lowered)
    return match.group(1) if match else lowered


def ground_truth_archive_family(archive: dict[str, object]) -> str:
    library = str(archive.get("library", "")).lower()
    source = str(archive.get("source", "")).lower()
    if library == "compiler-runtime":
        return library_family(source)
    if library == "glibc":
        return "glibc"
    if library == "pcre2":
        return "pcre2"
    if library in {"iconv", "libiconv"}:
        return "iconv"
    return library_family(str(archive.get("archive", library)))


def load_ground_truths(directory: Path | None) -> dict[str, dict[str, object]]:
    if directory is None:
        return {}
    if not directory.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {directory}")

    # Official ground-truth paths are portable and relative to the dataset
    # output root: <root>/ground_truth/<profile>/...
    artifact_root = directory.parent.parent
    ground_truths = {}
    for path in sorted(directory.rglob("ground_truth.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("binary"):
            declared = Path(str(data["binary"]))
            binary = declared if declared.is_absolute() else artifact_root / declared
            ground_truths[str(binary.resolve())] = data
        else:
            print(f"[WARN] Ground truth without binary path: {path}")
    return ground_truths


def expected_families(ground_truth: dict[str, object]) -> dict[str, dict[str, object]]:
    expected = {}
    for archive in ground_truth.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) > 0:
            expected[ground_truth_archive_family(archive)] = archive
    return expected


def build_ground_truth_rows(
    summary_rows: list[dict[str, str]],
    ground_truths: dict[str, dict[str, object]],
    cases: list[tuple[str, Path, str]],
    libraries: list[Path],
    library_labels: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    indexed = {
        (row["pipeline"], row["binary_path"], row["library"]): row
        for row in summary_rows
    }
    rows = []
    for pipeline, binary, label in cases:
        binary_path = str(binary.resolve())
        ground_truth = ground_truths.get(binary_path)
        if ground_truth is None:
            continue
        expected = expected_families(ground_truth)

        for library in libraries:
            library_label = (library_labels or {}).get(
                library.as_posix(), library.name
            )
            family = library_family(library_label)
            truth_archive = expected.get(family)
            expected_present = truth_archive is not None
            result = indexed.get((pipeline, binary_path, library_label))
            if result is None:
                predicted_present = ""
                classification = "MISSING"
            else:
                predicted = result["status"].startswith("YES")
                predicted_present = str(predicted)
                classification = (
                    "TP" if expected_present and predicted
                    else "FN" if expected_present
                    else "FP" if predicted
                    else "TN"
                )

            result = result or {}
            rows.append({
                "pipeline": pipeline,
                "asm_normalization": result.get("asm_normalization", ""),
                "palmtree_pooling": result.get("palmtree_pooling", ""),
                "library_score_aggregator": result.get(
                    "library_score_aggregator", ""
                ),
                "binary": label,
                "binary_path": binary_path,
                "library": library_label,
                "library_family": family,
                "ground_truth_variant": str(ground_truth.get("variant", "")),
                "expected_present": str(expected_present),
                "predicted_present": predicted_present,
                "classification": classification,
                "correct": str(classification in {"TP", "TN"}),
                "pipeline_status": result.get("status", ""),
                "pipeline_score": result.get("score", ""),
                "block_best": result.get("block_best", ""),
                "rodata_best": result.get("rodata_best", ""),
                "matched_cu": result.get("matched_cu", ""),
                "total_cu": result.get("total_cu", ""),
                "ground_truth_source": str(truth_archive.get("source", "")) if truth_archive else "",
                "ground_truth_optimization": str(truth_archive.get("optimization", "")) if truth_archive else "",
                "ground_truth_included_cu": str(truth_archive.get("included_compilation_units", 0)) if truth_archive else "0",
                "ground_truth_total_cu": str(truth_archive.get("total_compilation_units", 0)) if truth_archive else "0",
            })
    return rows


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.4f}" if denominator else ""


def build_ground_truth_metrics(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = {}
    for row in rows:
        groups.setdefault((row["pipeline"], row["binary"]), []).append(row)
        groups.setdefault((row["pipeline"], "ALL"), []).append(row)

    metrics = []
    for (pipeline, binary), group in sorted(groups.items()):
        provenance = next(
            (
                row
                for row in group
                if row.get("asm_normalization") or row.get("palmtree_pooling")
                or row.get("library_score_aggregator")
            ),
            {},
        )
        counts = {
            key: sum(row["classification"] == key for row in group)
            for key in ("TP", "TN", "FP", "FN", "MISSING")
        }
        evaluated = sum(counts[key] for key in ("TP", "TN", "FP", "FN"))
        metrics.append({
            "pipeline": pipeline,
            "asm_normalization": provenance.get("asm_normalization", ""),
            "palmtree_pooling": provenance.get("palmtree_pooling", ""),
            "library_score_aggregator": provenance.get(
                "library_score_aggregator", ""
            ),
            "binary": binary,
            "evaluated": str(evaluated),
            "missing": str(counts["MISSING"]),
            "tp": str(counts["TP"]),
            "tn": str(counts["TN"]),
            "fp": str(counts["FP"]),
            "fn": str(counts["FN"]),
            "accuracy": ratio(counts["TP"] + counts["TN"], evaluated),
            "precision": ratio(counts["TP"], counts["TP"] + counts["FP"]),
            "recall": ratio(counts["TP"], counts["TP"] + counts["FN"]),
            "specificity": ratio(counts["TN"], counts["TN"] + counts["FP"]),
            "f1": ratio(2 * counts["TP"], 2 * counts["TP"] + counts["FP"] + counts["FN"]),
        })
    return metrics


def persist_results(
    output_dir: Path,
    summary_rows: list[dict[str, str]],
    pipelines: list[tuple[str, Path]],
    ground_truths: dict[str, dict[str, object]],
    cases: list[tuple[str, Path, str]],
    libraries: list[Path],
    library_labels: dict[str, str] | None = None,
) -> None:
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    if {name for name, _ in pipelines} == {"current", "libseeker"}:
        write_csv(
            output_dir / "comparison.csv",
            COMPARISON_FIELDS,
            build_comparison_rows(summary_rows),
        )
    if ground_truths:
        score_rows = build_ground_truth_rows(
            summary_rows, ground_truths, cases, libraries, library_labels
        )
        write_csv(
            output_dir / "ground_truth_scores.csv",
            GROUND_TRUTH_FIELDS,
            score_rows,
        )
        write_csv(
            output_dir / "ground_truth_metrics.csv",
            GROUND_TRUTH_METRIC_FIELDS,
            build_ground_truth_metrics(score_rows),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def result_summary_path(result_log: Path) -> Path:
    name = result_log.name
    for suffix in (".jsonl.gz", ".gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return result_log.with_name(f"{name}.summary.json")


def dataset_record_map(
    dataset_dir: Path, manifest_path: Path | None
) -> tuple[Path | None, dict[str, dict]]:
    manifest = manifest_path or dataset_dir.parent / "manifest.json"
    if not manifest.is_file():
        return None, {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_root = manifest.parents[2]
    records: dict[str, dict] = {}
    for record in payload.get("records", []):
        binary = Path(str(record.get("binary", "")))
        binary = binary if binary.is_absolute() else artifact_root / binary
        records[binary.resolve().as_posix()] = record
    return manifest, records


def matching_configuration(args: argparse.Namespace) -> dict:
    fields = (
        "min_cu", "min_functions_cu", "libseeker_threshold",
        "asm_normalization", "palmtree_pooling", "library_score_aggregator",
        "library_min_score", "cross_cu_call_bonus_weight",
        "cross_cu_call_penalty_weight", "cross_cu_call_saturation_edges",
        "block_threshold", "block_coverage_mean_threshold",
        "block_assignment_quality_threshold", "block_min_assignment_ratio",
        "block_min_coverage_ratio", "block_locality_window_multiplier",
        "block_locality_window_padding", "block_min_call_edge_ratio",
        "block_min_instructions", "block_min_function_concentration",
        "block_min_function_spread", "cu_min_function_coverage",
        "disable_block_window_prefilter", "offline_ablation_features",
        "disable_rodata_filter", "rodata_min_bytes", "rodata_min_strings",
        "rodata_min_ngrams", "rodata_penalty_threshold",
        "rodata_confirm_threshold", "rodata_bonus_weight",
        "rodata_string_weight", "rodata_byte_only_weight",
    )
    return {field: getattr(args, field) for field in fields}


def feature_records(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid feature JSON in {path}:{line_number}: {error}"
                ) from error


def autonomous_feature_log(
    path: Path, expected_mode: str | None = None
) -> bool:
    """Return true only for logs that carry the schema-2 ELF catalog."""
    if not path.is_file():
        return False
    try:
        first = next(feature_records(path), None)
    except (OSError, EOFError, ValueError):
        return False
    autonomous = bool(
        first
        and first.get("type") == "source_call_targets"
        and int(first.get("schema_version", 0)) >= 2
        and isinstance(first.get("functions"), list)
    )
    if not autonomous:
        return False
    if expected_mode == "replay_complete":
        return bool(
            int(first.get("schema_version", 0)) >= 3
            and first.get("feature_mode") == "replay_complete"
            and isinstance(first.get("matching_configuration"), dict)
        )
    return True


def autonomous_replay_cu_record(
    payload: dict,
    elf_functions: dict[int, dict],
    elf_calls: dict[int, list[int]],
) -> dict:
    """Expand indexed schema-3 mappings for the portable result bundle.

    Local replay logs deduplicate function descriptors. The result-only shard
    remains autonomous, so its selected mapping is expanded from the per-ELF
    and per-CU catalogs before it is emitted.
    """
    if (
        int(payload.get("schema_version", 0)) < 3
        or payload.get("record_detail") != "replay"
    ):
        return payload

    expanded = dict(payload)
    target_functions = expanded.get("target_functions", [])
    if not isinstance(target_functions, list):
        raise ValueError("Replay CU target_functions is not a list")
    target_by_index = {
        int(function["index"]): function
        for function in target_functions
        if isinstance(function, dict) and "index" in function
    }

    def expand_match(match: object) -> dict:
        if not isinstance(match, dict):
            raise ValueError("Replay function match is not an object")
        if "reference_function" in match and "elf_function" in match:
            return dict(match)
        target_index = int(match["target_function_index"])
        source_index = int(match["source_function_index"])
        reference = target_by_index.get(target_index)
        elf_function = elf_functions.get(source_index)
        if reference is None or elf_function is None:
            raise ValueError(
                "Replay function mapping refers to an absent function "
                f"target={target_index}, source={source_index}"
            )
        return {
            **match,
            "reference_function": reference,
            "elf_function": elf_function,
            "source_call_targets": elf_calls.get(source_index, []),
        }

    selected = dict(expanded.get("selected_match", {}))
    selected_matches = selected.get("function_matches", [])
    if not isinstance(selected_matches, list):
        raise ValueError("Replay selected function matches is not a list")
    selected["function_matches"] = [
        expand_match(match) for match in selected_matches
    ]
    expanded["selected_match"] = selected
    expanded.setdefault("target_functions", target_functions)

    best = expanded.get("best_function_match")
    if isinstance(best, dict) and "elf_function" not in best:
        source_index = int(best["source_function_index"])
        elf_function = elf_functions.get(source_index)
        if elf_function is not None:
            expanded["best_function_match"] = {
                **best,
                "elf_function": elf_function,
                "source_call_targets": elf_calls.get(source_index, []),
            }
    return expanded


def write_result_bundle(
    *,
    args: argparse.Namespace,
    elfs: list[Path],
    libraries: list[Path],
    library_labels: dict[str, str],
    pipelines: list[tuple[str, Path]],
) -> dict:
    """Write one portable, compressed shard log and verify its coverage."""
    result_log = args.result_log
    result_log.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_log.with_name(f".{result_log.name}.{os.getpid()}.tmp")
    manifest_path, dataset_records = dataset_record_map(
        args.dataset_dir, args.dataset_manifest
    )
    labels = {
        library_labels[path.as_posix()]: path for path in libraries
    }
    if len(labels) != len(libraries):
        raise ValueError("Library labels are not one-to-one")
    library_catalog: dict[str, dict] = {}
    for label, path in sorted(labels.items()):
        details = dict(args.library_metadata.get(path.as_posix(), {}))
        library_catalog[label] = {
            "type": "library",
            "library_id": label,
            "archive": path.name,
            "archive_sha256": sha256_file(path),
            "archive_size": path.stat().st_size,
            **details,
        }

    pipeline_names = [name for name, _path in pipelines]
    run_record = {
        "type": "run",
        "schema_version": 1,
        "pipelines": pipeline_names,
        "elf_count": len(elfs),
        "library_count": len(libraries),
        "expected_library_matches": (
            len(elfs) * len(libraries) * len(pipeline_names)
        ),
        "dataset_manifest": (
            manifest_path.name if manifest_path else None
        ),
        "dataset_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path else None
        ),
        "library_matrix": (
            args.library_matrix.name if args.library_matrix else None
        ),
        "library_matrix_sha256": (
            sha256_file(args.library_matrix) if args.library_matrix else None
        ),
        "matching_configuration": matching_configuration(args),
        "accepted_cu_details": True,
        "rejected_cu_details": bool(args.include_rejected_cu),
    }
    counts = {
        "elf_records": 0,
        "library_records": len(library_catalog),
        "library_matches": 0,
        "cu_matches": 0,
        "accepted_cu_matches": 0,
        "function_matches": 0,
        "failures": 0,
    }

    def emit(stream, record: dict) -> None:
        stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )

    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            emit(stream, run_record)
            for record in library_catalog.values():
                emit(stream, record)

            for binary in sorted(elfs):
                resolved = binary.resolve()
                binary_sha256 = sha256_file(resolved)
                source_record = dataset_records.get(resolved.as_posix(), {})
                declared_sha = source_record.get("binary_sha256")
                if declared_sha and declared_sha != binary_sha256:
                    raise ValueError(f"Dataset/result ELF hash mismatch: {binary}")
                coordinate = (
                    str(source_record.get("program", binary.name)),
                    str(
                        source_record.get("compiler_command")
                        or source_record.get("compiler", "")
                    ),
                    str(source_record.get("program_optimization", "")),
                )
                elf_id = hashlib.sha256(
                    "\0".join((*coordinate, binary_sha256)).encode("utf-8")
                ).hexdigest()
                elf_record = {
                    "type": "elf",
                    "elf_id": elf_id,
                    "binary_sha256": binary_sha256,
                    "binary_size": resolved.stat().st_size,
                    "program": source_record.get("program", binary.name),
                    "source_project": source_record.get("source_project", ""),
                    "compiler": source_record.get("compiler", ""),
                    "compiler_command": source_record.get(
                        "compiler_command", ""
                    ),
                    "program_optimization": source_record.get(
                        "program_optimization", ""
                    ),
                    "dataset_binary": source_record.get("binary", ""),
                    "ground_truth": source_record.get("ground_truth", ""),
                    "ground_truth_complete": bool(
                        source_record.get("ground_truth_complete", False)
                    ),
                    "current_feature_log": (
                        (
                            Path("reports")
                            / "current"
                            / f"{binary_label(binary, args.dataset_dir)}.features.jsonl.gz"
                        ).as_posix()
                        if "current" in pipeline_names
                        else None
                    ),
                }
                emit(stream, elf_record)
                counts["elf_records"] += 1
                stem = binary_label(binary, args.dataset_dir)
                current_rows: dict[str, dict[str, str]] = {}
                for pipeline_name in pipeline_names:
                    reports = args.output_dir / "reports" / pipeline_name
                    report = reports / f"{stem}.report.txt"
                    raw_log = args.output_dir / "raw" / pipeline_name / f"{stem}.raw.log"
                    rows = parse_report(
                        pipeline_name, report, binary, stem, raw_log
                    )
                    by_library = {row["library"]: row for row in rows}
                    missing = sorted(set(labels) - set(by_library))
                    extra = sorted(set(by_library) - set(labels))
                    if missing or extra or len(rows) != len(labels):
                        raise ValueError(
                            f"Incomplete result for {binary} pipeline={pipeline_name}: "
                            f"rows={len(rows)}, missing={len(missing)}, extra={len(extra)}"
                        )
                    for label in sorted(labels):
                        row = dict(by_library[label])
                        # These values are already carried by the run, ELF and
                        # library records. Removing only the duplicates keeps
                        # the 15M+ pair matrix substantially smaller without
                        # discarding any experiment information.
                        for redundant in (
                            "binary_path", "binary", "log_path", "library",
                            "asm_normalization", "palmtree_pooling",
                            "library_score_aggregator", "library_min_score",
                        ):
                            row.pop(redundant, None)
                        row.update({
                            "type": "library_match",
                            "elf_id": elf_id,
                            "library_id": label,
                        })
                        emit(stream, row)
                        counts["library_matches"] += 1
                    if pipeline_name == "current":
                        current_rows = by_library

                if "current" not in pipeline_names:
                    continue
                features = (
                    args.output_dir / "reports/current"
                    / f"{stem}.features.jsonl.gz"
                )
                if not features.is_file():
                    features = features.with_suffix("")
                if not features.is_file():
                    raise FileNotFoundError(features)
                pass_counts: dict[str, int] = {}
                source_catalogs = 0
                elf_functions: dict[int, dict] = {}
                elf_calls: dict[int, list[int]] = {}
                for payload in feature_records(features):
                    record_type = payload.get("type")
                    if record_type == "source_call_targets":
                        elf_functions = {
                            int(function["index"]): function
                            for function in payload.get("functions", [])
                            if isinstance(function, dict) and "index" in function
                        }
                        elf_calls = {
                            int(source_index): [
                                int(target) for target in targets
                            ]
                            for source_index, targets in payload.get("calls", [])
                        }
                        payload["type"] = "elf_function_catalog"
                        payload["elf_id"] = elf_id
                        payload.pop("binary_path", None)
                        emit(stream, payload)
                        source_catalogs += 1
                        continue
                    if record_type != "block_cu":
                        continue
                    if int(payload.get("schema_version", 0)) < 2:
                        raise ValueError(
                            f"CU feature schema is not autonomous: {features}"
                        )
                    label = str(payload.get("library", ""))
                    if label not in labels:
                        raise ValueError(
                            f"Unknown feature library {label!r} in {features}"
                        )
                    status = str(payload.get("cu_status", ""))
                    if status == "PASS":
                        pass_counts[label] = pass_counts.get(label, 0) + 1
                    if status != "PASS" and not args.include_rejected_cu:
                        continue
                    payload = autonomous_replay_cu_record(
                        payload, elf_functions, elf_calls
                    )
                    payload["type"] = "cu_match"
                    payload["elf_id"] = elf_id
                    payload["library_id"] = label
                    payload.pop("binary_path", None)
                    payload.pop("library", None)
                    # Pareto/rejected windows remain in the local replay file.
                    # The portable log keeps the selected mapping in full.
                    payload.pop("windows", None)
                    emit(stream, payload)
                    counts["cu_matches"] += 1
                    if status == "PASS":
                        counts["accepted_cu_matches"] += 1
                    counts["function_matches"] += len(
                        payload.get("selected_match", {}).get(
                            "function_matches", []
                        )
                    )
                if source_catalogs != 1:
                    raise ValueError(
                        f"Expected one ELF function catalog in {features}, "
                        f"found {source_catalogs}"
                    )
                for label, row in current_rows.items():
                    expected = int(row.get("matched_cu") or 0)
                    actual = pass_counts.get(label, 0)
                    if actual != expected:
                        raise ValueError(
                            f"CU detail mismatch for {binary} {label}: "
                            f"PASS={actual}, report matched_cu={expected}"
                        )

        with gzip.open(temporary, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        os.replace(temporary, result_log)
    finally:
        temporary.unlink(missing_ok=True)

    if counts["elf_records"] != len(elfs):
        raise AssertionError(counts)
    if counts["library_matches"] != run_record["expected_library_matches"]:
        raise AssertionError(counts)
    summary = {
        "schema_version": 1,
        "valid": True,
        "result_log": result_log.as_posix(),
        "result_log_sha256": sha256_file(result_log),
        "result_log_size": result_log.stat().st_size,
        **counts,
        "expected_library_matches": run_record["expected_library_matches"],
    }
    summary_path = result_summary_path(result_log)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_batch(args: argparse.Namespace) -> int:
    args.dataset_dir = args.dataset_dir.resolve()
    args.libs_dir = args.libs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.library_matrix is not None:
        args.library_matrix = args.library_matrix.expanduser().resolve()
    if args.library_root is not None:
        args.library_root = args.library_root.expanduser().resolve()
    if args.dataset_manifest is not None:
        args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
    args.result_log = (
        args.result_log.expanduser().resolve()
        if args.result_log
        else args.output_dir / "results.jsonl.gz"
    )
    args.analysis_cache_dir = args.analysis_cache_dir.expanduser().resolve()
    if args.ground_truth_dir is not None:
        args.ground_truth_dir = args.ground_truth_dir.resolve()
    args.libseeker_pipeline_dir = args.libseeker_pipeline_dir.resolve()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if args.library_matrix is None and not args.libs_dir.is_dir():
        raise FileNotFoundError(f"Libraries directory not found: {args.libs_dir}")
    if args.library_matrix is not None and not args.library_matrix.is_file():
        raise FileNotFoundError(f"Library matrix not found: {args.library_matrix}")

    pipelines = selected_pipelines(args)
    for pipeline_name, pipeline_dir in pipelines:
        if not (pipeline_dir / "main.py").is_file():
            raise FileNotFoundError(
                f"{pipeline_name} pipeline main.py not found in: {pipeline_dir}"
            )

    elfs = select_elfs(args)
    args.library_metadata = {}
    libraries = select_libs(args)
    library_labels = stable_library_labels(
        libraries, args.library_metadata
    )
    ground_truths = load_ground_truths(args.ground_truth_dir)
    if not elfs:
        raise ValueError("No ELF files selected")
    if not libraries:
        raise ValueError("No libraries selected")
    if args.analysis_cache_only and not portable_cache_is_complete(
        args.analysis_cache_dir
    ):
        raise ValueError(
            "--analysis-cache-only requires a complete, validated shard inventory"
        )

    adaptive_jobs = args.jobs == "auto"
    if adaptive_jobs:
        initial_matching_jobs, auto_job_reason = automatic_job_count(
            len(elfs),
            args.analysis_cache_dir,
            max_workers=args.adaptive_max_jobs,
        )
        matching_jobs = initial_matching_jobs
        if portable_cache_is_complete(args.analysis_cache_dir):
            matching_jobs = min(
                len(elfs),
                available_cpu_count(),
                args.adaptive_max_jobs,
            )
        print(
            f"Auto-selected {initial_matching_jobs} initial and up to "
            f"{matching_jobs} adaptive matching job(s): "
            f"{auto_job_reason}"
        )
    elif args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    else:
        matching_jobs = args.jobs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(elfs)} ELF file(s)")
    print(f"Selected {len(libraries)} librar(y/ies)")
    print(f"Pipelines: {', '.join(name for name, _ in pipelines)}")
    print(f"Output directory: {args.output_dir}")

    portable_matrix_run = args.library_matrix is not None
    summary_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str, int]] = []
    cases = [
        (pipeline_name, binary, binary_label(binary, args.dataset_dir))
        for pipeline_name, _ in pipelines
        for binary in elfs
    ]

    with make_libraries_dir(libraries, library_labels) as libraries_tmp:
        libraries_dir = Path(libraries_tmp)
        for pipeline_name, pipeline_dir in pipelines:
            worker_count = min(matching_jobs, len(elfs))
            worker_environment, threads_per_worker = matching_worker_environment(
                worker_count
            )
            adaptive_controller = (
                AdaptiveJobController(
                    worker_count,
                    initial_workers=min(initial_matching_jobs, worker_count),
                )
                if adaptive_jobs
                else None
            )
            if adaptive_controller is not None:
                print(
                    f"{pipeline_name}: adaptive 1..{worker_count} worker(s), "
                    f"{adaptive_controller.memory_reserve_bytes / 1024**3:.1f} "
                    "GiB memory reserve"
                )
            else:
                print(
                    f"{pipeline_name}: {worker_count} worker(s), "
                    f"{threads_per_worker} numerical thread(s) per worker"
                )
            raw_dir = args.output_dir / "raw" / pipeline_name
            reports_dir = args.output_dir / "reports" / pipeline_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            def process_binary(
                binary: Path,
            ) -> tuple[str, int, list[dict[str, str]]]:
                stem = binary_label(binary, args.dataset_dir)
                raw_log = raw_dir / f"{stem}.raw.log"
                report = reports_dir / f"{stem}.report.txt"
                features = (
                    reports_dir / f"{stem}.features.jsonl.gz"
                    if pipeline_name == "current"
                    else None
                )
                existing_rows = parse_report(
                    pipeline_name, report, binary, stem, raw_log
                )
                complete_outputs = (
                    raw_log.exists()
                    and report.exists()
                    and (
                        features is None
                        or autonomous_feature_log(
                            features,
                            (
                                "replay_complete"
                                if args.offline_ablation_features
                                else "standard"
                            ),
                        )
                    )
                    and bool(parse_total_time(raw_log))
                    and len(existing_rows) == len(libraries)
                    and (
                        pipeline_name != "current"
                        or abs(
                            parse_library_min_score(report)
                            - args.library_min_score
                        )
                        < 1e-12
                    )
                    and (
                        pipeline_name != "current"
                        or parse_asm_normalization(report)
                        == args.asm_normalization
                    )
                    and (
                        pipeline_name != "current"
                        or parse_palmtree_pooling(report)
                        == args.palmtree_pooling
                    )
                    and (
                        pipeline_name != "current"
                        or parse_library_score_aggregator(report)
                        == args.library_score_aggregator
                    )
                    and (
                        pipeline_name != "current"
                        or parse_feature_collection_mode(report)
                        == (
                            "replay_complete"
                            if args.offline_ablation_features
                            else "standard"
                        )
                    )
                    and (
                        pipeline_name != "current"
                        or (
                            parse_float_setting(
                                report, "CU minimum function coverage"
                            ) or 0.0
                        ) == args.cu_min_function_coverage
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(
                            report, "Rodata penalty threshold"
                        ) == args.rodata_penalty_threshold
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(
                            report, "Rodata confirm threshold"
                        ) == args.rodata_confirm_threshold
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(
                            report, "Rodata bonus weight"
                        ) == args.rodata_bonus_weight
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(report, "Rodata string weight")
                        == args.rodata_string_weight
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(report, "Rodata byte-only weight")
                        == args.rodata_byte_only_weight
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(
                            report, "Cross-CU call bonus weight"
                        ) == args.cross_cu_call_bonus_weight
                    )
                    and (
                        pipeline_name != "current"
                        or parse_float_setting(
                            report, "Cross-CU call penalty weight"
                        ) == args.cross_cu_call_penalty_weight
                    )
                )

                if args.resume and complete_outputs:
                    print(f"[resume:{pipeline_name}] {binary.name}")
                    return stem, 0, existing_rows
                else:
                    if pipeline_name == "current":
                        command = current_command(
                            args,
                            binary,
                            libraries_dir,
                            report,
                            features,
                        )
                    else:
                        command = libseeker_command(args, binary, libraries_dir, report)

                    while True:
                        adaptive_token = None
                        current_environment = worker_environment
                        if adaptive_controller is not None:
                            adaptive_token = adaptive_controller.acquire(stem)
                            current_environment, _threads = (
                                matching_worker_environment(
                                    adaptive_controller.concurrency_limit
                                )
                            )
                        try:
                            returncode = run_command(
                                command,
                                cwd=pipeline_dir,
                                log_path=raw_log,
                                timeout=args.timeout,
                                dry_run=args.dry_run,
                                environment=current_environment,
                                adaptive_controller=adaptive_controller,
                                adaptive_token=adaptive_token,
                            )
                        finally:
                            if (
                                adaptive_controller is not None
                                and adaptive_token is not None
                            ):
                                adaptive_controller.release(adaptive_token)
                        if returncode != ADAPTIVE_RETRY_RETURN_CODE:
                            break
                        print(
                            f"[adaptive:{pipeline_name}] retry queued for "
                            f"{binary.name}",
                            flush=True,
                        )
                rows = parse_report(
                    pipeline_name, report, binary, stem, raw_log
                )
                return stem, returncode, rows

            if worker_count == 1:
                results = map(process_binary, elfs)
            else:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                results = executor.map(process_binary, elfs)
            try:
                for binary, (stem, returncode, rows) in zip(elfs, results):
                    if returncode != 0:
                        failures.append(
                            (pipeline_name, binary.name, returncode)
                        )
                        if args.analysis_cache_only and pipeline_name == "current":
                            print(
                                "Cache-only matching failed for "
                                f"{binary.name}; continuing with the remaining "
                                "ELFs"
                            )
                    if not portable_matrix_run:
                        summary_rows.extend(rows)
                        persist_results(
                            args.output_dir,
                            summary_rows,
                            pipelines,
                            ground_truths,
                            cases,
                            libraries,
                            library_labels,
                        )
            finally:
                if worker_count > 1:
                    executor.shutdown(wait=True, cancel_futures=True)

    if failures:
        print("Failures:")
        for pipeline_name, binary_name, returncode in failures:
            print(f"  {pipeline_name} {binary_name}: return code {returncode}")
        return 1

    if args.dry_run:
        print("Dry run: no portable result log was written.")
        return 0

    if portable_matrix_run:
        write_result_bundle(
            args=args,
            elfs=elfs,
            libraries=libraries,
            library_labels=library_labels,
            pipelines=pipelines,
        )
        print(f"Wrote {args.result_log}")
    else:
        persist_results(
            args.output_dir,
            summary_rows,
            pipelines,
            ground_truths,
            cases,
            libraries,
            library_labels,
        )
        print(f"Wrote {args.output_dir / 'summary.csv'}")
        if args.pipeline == "both":
            print(f"Wrote {args.output_dir / 'comparison.csv'}")
        if ground_truths:
            print(f"Wrote {args.output_dir / 'ground_truth_scores.csv'}")
            print(f"Wrote {args.output_dir / 'ground_truth_metrics.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_batch(parse_args()))
