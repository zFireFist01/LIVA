#!/usr/bin/env python3
"""Evaluate a fixed decision configuration at logical-library-family level.

By default every valid report/feature-log/ground-truth tuple is evaluated for
each compiler independently.  A positive is one (ELF, logical-family) pair:
all build labels belonging to the same archive family are collapsed before
computing TP, TN, FP and FN.  Use --elfs-per-compiler to request a smaller,
balanced panel shared by the selected compilers.

The JSONL checkpoint is append-only and makes an interrupted run resumable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402


ENGINE = family.engine
COMPILERS = family.COMPILERS
OPTIMIZATIONS = family.OPTIMIZATIONS

# Edit this dictionary to test another fixed configuration.
PARAMS: dict[str, Any] = {
    "library_score_aggregator": "top3_mean",
    "library_min_score": 0.905,
    "block_coverage_mean_threshold": 0.975,
    "block_min_coverage_ratio": 0.75,
    "block_assignment_quality_threshold": 0.999,
    "block_min_assignment_ratio": 0.9,
    "block_min_call_edge_ratio": 0.3,
    "block_min_function_concentration": 0.75,
    "block_min_function_spread": 0.3,
    "cu_min_function_coverage": 0.6,
    "cross_cu_call_bonus_weight": 0.2,
    "cross_cu_call_penalty_weight": 0.0,
    "cross_cu_call_saturation_edges": 8,
    "rodata_filter_enabled": 1,
    "rodata_penalty_threshold": 0.35,
    "rodata_confirm_threshold": 0.9,
    "rodata_bonus_weight": 0.3,
    "rodata_min_bytes": 512,
    "rodata_min_ngrams": 32,
    "rodata_min_strings": 16,
}

SOURCE_CALLS_MARKER = b'"type":"source_call_targets"'
BLOCK_CU_MARKER = b'"type":"block_cu"'

_CANDIDATE_FAMILIES: set[str] = set()


def canonical_hash(payload: Any) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def metrics(counts: Counter[str]) -> dict[str, int | float]:
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    return {
        "TP": tp,
        "TN": counts["TN"],
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def balanced_panel(
    common: Iterable[tuple[str, str]], size: int, seed: int
) -> list[tuple[str, str]]:
    coordinates = sorted(common)
    if size <= 0 or size >= len(coordinates):
        return coordinates
    base, remainder = divmod(size, len(OPTIMIZATIONS))
    quotas = {
        optimization: base + (index < remainder)
        for index, optimization in enumerate(OPTIMIZATIONS)
    }
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for optimization in OPTIMIZATIONS:
        available = [c for c in coordinates if c[1] == optimization]
        quota = quotas[optimization]
        if len(available) < quota:
            raise ValueError(
                f"Only {len(available)} common {optimization} coordinates; "
                f"cannot select {quota}"
            )
        selected.extend(rng.sample(available, quota))
    return sorted(selected)


def compact_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    windows = [
        family.compact_window(window)
        for window in payload.get("windows", [])
        if float(window.get("coverage_mean", 0.0)) >= float(
            PARAMS["block_coverage_mean_threshold"]
        )
    ]
    if not windows:
        return None
    return {
        "name": str(payload.get("name", "")),
        "target_cu_index": int(payload["target_cu_index"]),
        "target_function_count": int(payload.get("target_function_count", 0)),
        "windows": windows,
        "inter_cu_calls": [
            {
                "caller_function_index": int(edge["caller_function_index"]),
                "callee_cu_index": int(edge["callee_cu_index"]),
                "callee_function_index": int(edge["callee_function_index"]),
            }
            for edge in payload.get("inter_cu_calls", [])
        ],
        "rodata": float(payload.get("rodata", 0.0)),
        "rodata_has_rodata": bool(payload.get("rodata_has_rodata", False)),
        "rodata_strings": int(payload.get("rodata_strings", 0)),
        "rodata_ngrams": int(payload.get("rodata_ngrams", 0)),
        "rodata_bytes": int(payload.get("rodata_bytes", 0)),
    }


def parse_relevant_features(
    path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[int, list[int]]]:
    header: dict[str, Any] | None = None
    libraries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_call_targets: dict[int, list[int]] = {}
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            is_source_calls = SOURCE_CALLS_MARKER in raw
            if not is_source_calls:
                windows_at = raw.find(b'"windows":[')
                if (
                    BLOCK_CU_MARKER not in raw
                    or windows_at < 0
                ):
                    continue
            decoded = raw.decode("utf-8", "replace")
            for payload in family.iter_json_objects((decoded,)):
                kind = payload.get("type")
                if kind == "source_call_targets":
                    header = payload
                    source_call_targets = {
                        int(source): [int(target) for target in targets]
                        for source, targets in payload.get("calls", [])
                    }
                elif kind == "block_cu":
                    record = compact_record(payload)
                    if record is not None:
                        libraries[str(payload["library"])].append(record)
    if header is None or header.get("feature_mode") != "replay_complete":
        raise ValueError(f"Not a replay_complete feature log: {path}")
    return header, dict(libraries), source_call_targets


def init_worker(candidate_families: set[str]) -> None:
    global _CANDIDATE_FAMILIES
    _CANDIDATE_FAMILIES = candidate_families


def evaluate_case(task: dict[str, str]) -> dict[str, Any]:
    feature = Path(task["feature"])
    truth = Path(task["truth"])
    header, libraries, calls = parse_relevant_features(feature)
    params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **PARAMS,
    }
    predicted: set[str] = set()
    accepted_labels = 0
    for label, records in libraries.items():
        accepted, score = ENGINE.library_match_evidence(records, params, calls)
        if accepted and score >= float(params["library_min_score"]):
            predicted.add(family.family_from_label(label))
            accepted_labels += 1
    expected = family.expected_families(truth, _CANDIDATE_FAMILIES)
    true_positives = predicted & expected
    false_positives = predicted - expected
    false_negatives = expected - predicted
    true_negatives = _CANDIDATE_FAMILIES - predicted - expected
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tn = len(true_negatives)
    return {
        "run_signature": task["run_signature"],
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "program": task["program"],
        "optimization": task["optimization"],
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "expected_families": len(expected),
        "predicted_families": len(predicted),
        "accepted_build_labels": accepted_labels,
        "labels_with_perfect_coverage": len(libraries),
        "true_positive_families": sorted(true_positives),
        "true_negative_families": sorted(true_negatives),
        "false_positive_families": sorted(false_positives),
        "false_negative_families": sorted(false_negatives),
    }


def evaluate_case_safe(task: dict[str, str]) -> dict[str, Any]:
    """Keep the pool alive when one input artifact is unreadable."""
    try:
        return {"ok": True, "row": evaluate_case(task)}
    except Exception as exc:  # recorded verbatim in the run error log
        return {
            "ok": False,
            "error": {
                "run_signature": task["run_signature"],
                "case_id": task["case_id"],
                "compiler": task["compiler"],
                "program": task["program"],
                "optimization": task["optimization"],
                "feature": task["feature"],
                "truth": task["truth"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }


def load_checkpoint(
    path: Path,
    run_signature: str,
    compatible_signatures: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load per-case results and migrate compatible legacy signatures.

    Older checkpoints used a signature that included the complete corpus
    selection.  Adding one ELF therefore invalidated every otherwise
    independent case.  A compatible signature is accepted only after its
    report metadata has been checked by ``compatible_report_signatures``.
    Migrated rows are returned separately so callers can persist the stable
    signature once and make subsequent runs independent of the old report.
    """
    compatible = set(compatible_signatures or ())
    stable: dict[str, dict[str, Any]] = {}
    legacy: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return stable, []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == sum(1 for _ in path.open(encoding="utf-8")):
                    break
                raise
            signature = str(row.get("run_signature", ""))
            case_id = str(row["case_id"])
            if signature == run_signature:
                stable[case_id] = row
            elif signature in compatible:
                legacy[case_id] = row
    migrated = []
    for case_id, row in legacy.items():
        if case_id in stable:
            continue
        migrated_row = {**row, "run_signature": run_signature}
        stable[case_id] = migrated_row
        migrated.append(migrated_row)
    return stable, migrated


def compatible_report_signatures(
    path: Path,
    candidate_family_count: int,
) -> set[str]:
    """Return the legacy signature when its evaluation semantics match."""
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    signature = payload.get("run_signature")
    if (
        not isinstance(signature, str)
        or payload.get("configuration") != PARAMS
        or int(payload.get("candidate_families", -1)) != candidate_family_count
        or payload.get("family_definition")
        != (
            "logical archive basename without .a/build hash; predictions are "
            "collapsed to one decision per ELF-family"
        )
    ):
        return set()
    return {signature}


def summarize(
    rows: Iterable[dict[str, Any]], compilers: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(rows)
    results: list[dict[str, Any]] = []
    combined: Counter[str] = Counter()
    for compiler in compilers:
        compiler_rows = [row for row in rows if row["compiler"] == compiler]
        counts: Counter[str] = Counter()
        for row in compiler_rows:
            counts.update({key: int(row[key]) for key in ("TP", "TN", "FP", "FN")})
        combined.update(counts)
        results.append(
            {
                "compiler": compiler,
                "evaluated_elfs": len(compiler_rows),
                **metrics(counts),
            }
        )
    macro_f1 = sum(float(result["f1"]) for result in results) / len(results)
    return results, {**metrics(combined), "macro_compiler_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elfs-per-compiler",
        type=int,
        default=0,
        help="0 (default) evaluates every valid ELF independently per compiler",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--compiler", action="append", choices=COMPILERS,
        help="Repeat to restrict the run; the default uses all four compilers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "libseeker-unified/family_f1_fixed_200x4.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    compilers = tuple(args.compiler or COMPILERS)
    checkpoint = args.checkpoint or args.output.with_suffix(".cases.jsonl")

    index = family.report_index()
    labels, family_by_label = family.candidate_metadata(index)
    candidate_families = set(family_by_label.values())
    valid_by_compiler: dict[str, list[tuple[str, str]]] = {}
    for compiler in compilers:
        valid_by_compiler[compiler] = sorted(
            coordinate
            for coordinate, report in index[compiler].items()
            if coordinate[1] in OPTIMIZATIONS
            and report.with_name(
                report.name.replace(".report.txt", ".features.jsonl.gz")
            ).is_file()
            and family.truth_path(coordinate[0], compiler, coordinate[1]).is_file()
        )

    if args.elfs_per_compiler > 0:
        common = set.intersection(
            *(set(valid_by_compiler[compiler]) for compiler in compilers)
        )
        panel = balanced_panel(common, args.elfs_per_compiler, args.seed)
        selection_by_compiler = {compiler: panel for compiler in compilers}
        selection_kind = "seeded_balanced_common_coordinates"
    else:
        selection_by_compiler = valid_by_compiler
        selection_kind = "all_valid_coordinates_per_compiler"

    selection = {
        compiler: [
            f"{program}__{optimization}"
            for program, optimization in selection_by_compiler[compiler]
        ]
        for compiler in compilers
    }
    # This signature deliberately excludes the corpus selection.  Each ELF is
    # evaluated independently, so adding reports must schedule only new cases.
    run_signature = canonical_hash(
        {
            "schema_version": 3,
            "protocol": "task1_fixed_family_per_case",
            "configuration": PARAMS,
            "candidate_families": sorted(candidate_families),
        }
    )
    tasks: list[dict[str, str]] = []
    for compiler in compilers:
        for program, optimization in selection_by_compiler[compiler]:
            report = index[compiler][(program, optimization)]
            feature = report.with_name(
                report.name.replace(".report.txt", ".features.jsonl.gz")
            )
            tasks.append(
                {
                    "run_signature": run_signature,
                    "case_id": f"{program}__{compiler}__{optimization}",
                    "compiler": compiler,
                    "program": program,
                    "optimization": optimization,
                    "feature": str(feature),
                    "truth": str(family.truth_path(program, compiler, optimization)),
                }
            )

    legacy_signatures = compatible_report_signatures(
        args.output, len(candidate_families)
    )
    completed, migrated = load_checkpoint(
        checkpoint, run_signature, legacy_signatures
    )
    if migrated:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as stream:
            for row in migrated:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"migrated {len(migrated)} compatible checkpoint case(s) "
            "to the selection-independent signature",
            flush=True,
        )
    pending = [task for task in tasks if task["case_id"] not in completed]
    error_log = args.output.with_suffix(".errors.jsonl")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "run_signature": run_signature,
                "compilers": compilers,
                "elfs_per_compiler": {
                    compiler: len(selection_by_compiler[compiler])
                    for compiler in compilers
                },
                "total_cases": len(tasks),
                "resumed_cases": len(completed),
                "pending_cases": len(pending),
                "workers": args.workers,
                "checkpoint": str(checkpoint),
                "error_log": str(error_log),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )

    started = time.monotonic()
    run_errors: dict[str, dict[str, Any]] = {}
    if pending:
        with (
            checkpoint.open("a", encoding="utf-8") as checkpoint_stream,
            error_log.open("a", encoding="utf-8") as error_stream,
        ):
            with multiprocessing.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(candidate_families,),
            ) as pool:
                for index_value, result in enumerate(
                    pool.imap_unordered(
                        evaluate_case_safe, pending, chunksize=1
                    ),
                    start=1,
                ):
                    if result["ok"]:
                        row = result["row"]
                        completed[row["case_id"]] = row
                        checkpoint_stream.write(
                            json.dumps(row, sort_keys=True) + "\n"
                        )
                        checkpoint_stream.flush()
                    else:
                        error = result["error"]
                        run_errors[error["case_id"]] = error
                        error_stream.write(
                            json.dumps(error, sort_keys=True) + "\n"
                        )
                        error_stream.flush()
                        print(
                            f"ERROR {error['case_id']}: "
                            f"{error['error_type']}: {error['error']}",
                            flush=True,
                        )
                    if index_value % 10 == 0 or index_value == len(pending):
                        print(
                            f"progress {len(completed)}/{len(tasks)} "
                            f"elapsed={time.monotonic() - started:.1f}s",
                            flush=True,
                        )

    rows = [
        completed[task["case_id"]]
        for task in tasks
        if task["case_id"] in completed
    ]
    missing = [
        {
            "case_id": task["case_id"],
            "compiler": task["compiler"],
            "program": task["program"],
            "optimization": task["optimization"],
            "feature": task["feature"],
            "error": run_errors.get(task["case_id"]),
        }
        for task in tasks
        if task["case_id"] not in completed
    ]
    per_compiler, combined = summarize(rows, compilers)
    payload = {
        "schema_version": 2,
        "run_signature": run_signature,
        "complete": not missing,
        "evaluated_cases": len(rows),
        "total_cases": len(tasks),
        "missing_cases": missing,
        "family_definition": (
            "logical archive basename without .a/build hash; predictions are "
            "collapsed to one decision per ELF-family"
        ),
        "configuration": PARAMS,
        "selection": {
            "kind": selection_kind,
            "seed": args.seed,
            "elfs_per_compiler": {
                compiler: len(selection_by_compiler[compiler])
                for compiler in compilers
            },
            "optimization_counts": {
                compiler: dict(
                    Counter(value[1] for value in selection_by_compiler[compiler])
                )
                for compiler in compilers
            },
            "coordinates": selection,
        },
        "candidate_builds": len(labels),
        "candidate_families": len(candidate_families),
        "per_compiler": per_compiler,
        "pooled": combined,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
