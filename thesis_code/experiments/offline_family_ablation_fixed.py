#!/usr/bin/env python3
"""Fixed-parameter offline B/H/S/X/R ablation for Task-1 family F1.

Every configuration is evaluated on the same (ELF, logical archive family)
universe used by evaluate_fixed_family_f1.py.  All version/compiler/build
labels are collapsed to one of 164 families before TP/TN/FP/FN are counted.
The expensive binary matching is not repeated: schema-3 replay feature logs
are consumed offline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
import evaluate_fixed_family_f1 as fixed  # noqa: E402


ENGINE = family.engine
COMPILERS = family.COMPILERS
OPTIMIZATIONS = family.OPTIMIZATIONS
COMPONENTS = ("H", "S", "X", "R")

# Increment this whenever the replay/evaluation semantics change in a way that
# can alter a per-ELF ablation row.  Historical checkpoints predate this
# explicit revision and cannot safely be inferred compatible from their
# parameter dictionary alone.
EVALUATION_SEMANTICS_VERSION = 2

_CANDIDATE_FAMILIES: set[str] = set()


def configurations() -> list[tuple[str, frozenset[str]]]:
    result: list[tuple[str, frozenset[str]]] = []
    for size in range(len(COMPONENTS) + 1):
        for enabled in itertools.combinations(COMPONENTS, size):
            modules = frozenset(enabled)
            name = "B" if not modules else "B_" + "_".join(enabled)
            if modules == frozenset(COMPONENTS):
                name = "FULL"
            result.append((name, modules))
    return result


SPECS = configurations()


def ablated_params(
    full: dict[str, Any], modules: frozenset[str]
) -> dict[str, Any]:
    params = dict(full)
    if "H" not in modules:
        params["block_assignment_quality_threshold"] = 0.0
        params["block_min_assignment_ratio"] = 0.0
    if "S" not in modules:
        params["block_min_call_edge_ratio"] = 0.0
        params["block_min_function_concentration"] = 0.0
        params["block_min_function_spread"] = 0.0
        params["cu_min_function_coverage"] = 0.0
    if "X" not in modules:
        params["cross_cu_call_bonus_weight"] = 0.0
        params["cross_cu_call_penalty_weight"] = 0.0
    if "R" not in modules:
        params["rodata_filter_enabled"] = 0
        params["rodata_bonus_weight"] = 0.0
    return params


def init_worker(candidate_families: set[str]) -> None:
    global _CANDIDATE_FAMILIES
    _CANDIDATE_FAMILIES = candidate_families


def evaluate_case(task: dict[str, str]) -> list[dict[str, Any]]:
    feature = Path(task["feature"])
    truth = Path(task["truth"])
    header, libraries, calls = fixed.parse_relevant_features(feature)
    full_params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **fixed.PARAMS,
    }
    expected = family.expected_families(truth, _CANDIDATE_FAMILIES)
    rows: list[dict[str, Any]] = []
    for configuration, modules in SPECS:
        params = ablated_params(full_params, modules)
        predicted: set[str] = set()
        accepted_labels = 0
        for label, records in libraries.items():
            accepted, score = ENGINE.library_match_evidence(records, params, calls)
            if accepted and score >= float(params["library_min_score"]):
                predicted.add(family.family_from_label(label))
                accepted_labels += 1
        tp_families = predicted & expected
        fp_families = predicted - expected
        fn_families = expected - predicted
        tn = len(_CANDIDATE_FAMILIES - predicted - expected)
        rows.append(
            {
                "run_signature": task["run_signature"],
                "case_id": task["case_id"],
                "compiler": task["compiler"],
                "program": task["program"],
                "optimization": task["optimization"],
                "configuration": configuration,
                "modules": ["B", *[c for c in COMPONENTS if c in modules]],
                "TP": len(tp_families),
                "TN": tn,
                "FP": len(fp_families),
                "FN": len(fn_families),
                "expected_families": len(expected),
                "predicted_families": len(predicted),
                "accepted_build_labels": accepted_labels,
                "true_positive_families": sorted(tp_families),
                "false_positive_families": sorted(fp_families),
                "false_negative_families": sorted(fn_families),
            }
        )
    return rows


def evaluate_case_safe(task: dict[str, str]) -> dict[str, Any]:
    try:
        return {"ok": True, "rows": evaluate_case(task)}
    except Exception as exc:
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
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load rows and migrate selection-bound legacy signatures once."""
    compatible = set(compatible_signatures or ())
    stable: dict[tuple[str, str], dict[str, Any]] = {}
    legacy: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return stable, []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            signature = str(row.get("run_signature", ""))
            key = (str(row["case_id"]), str(row["configuration"]))
            if signature == run_signature:
                stable[key] = row
            elif signature in compatible:
                legacy[key] = row
    migrated = []
    for key, row in legacy.items():
        if key in stable:
            continue
        migrated_row = {**row, "run_signature": run_signature}
        stable[key] = migrated_row
        migrated.append(migrated_row)
    return stable, migrated


def compatible_report_signatures(
    path: Path,
    candidate_family_count: int,
) -> set[str]:
    """Accept a prior report only when all evaluation semantics match."""
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    signature = payload.get("run_signature")
    if (
        not isinstance(signature, str)
        or int(payload.get("evaluation_semantics_version", -1))
        != EVALUATION_SEMANTICS_VERSION
        or payload.get("configuration_full") != fixed.PARAMS
        or int(payload.get("candidate_families", -1)) != candidate_family_count
        or int(payload.get("configuration_count", -1)) != len(SPECS)
        or payload.get("protocol")
        != "fixed-parameter pure factorial B/H/S/X/R ablation"
    ):
        return set()
    return {signature}


def summarize(
    rows: Iterable[dict[str, Any]], compilers: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(rows)
    summary: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, Any]] = {}
    for configuration, modules in SPECS:
        config_rows = [r for r in rows if r["configuration"] == configuration]
        combined: Counter[str] = Counter()
        compiler_f1: list[float] = []
        for compiler in compilers:
            selected = [r for r in config_rows if r["compiler"] == compiler]
            counts: Counter[str] = Counter()
            for row in selected:
                counts.update(
                    {key: int(row[key]) for key in ("TP", "TN", "FP", "FN")}
                )
            values = fixed.metrics(counts)
            compiler_f1.append(float(values["f1"]))
            summary.append(
                {
                    "configuration": configuration,
                    "modules": "+".join(
                        ["B", *[c for c in COMPONENTS if c in modules]]
                    ),
                    "scope": compiler,
                    "evaluated_elfs": len(selected),
                    **values,
                    "macro_compiler_f1": "",
                }
            )
            combined.update(counts)
        pooled_values = fixed.metrics(combined)
        pooled[configuration] = {
            **pooled_values,
            "macro_compiler_f1": sum(compiler_f1) / len(compiler_f1),
        }
        summary.append(
            {
                "configuration": configuration,
                "modules": "+".join(
                    ["B", *[c for c in COMPONENTS if c in modules]]
                ),
                "scope": "pooled",
                "evaluated_elfs": len(config_rows),
                **pooled[configuration],
            }
        )
    full_f1 = float(pooled["FULL"]["f1"])
    for row in summary:
        if row["scope"] == "pooled":
            row["delta_f1_vs_full"] = float(row["f1"]) - full_f1
        else:
            row["delta_f1_vs_full"] = ""
    return summary, pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "libseeker-unified/ablation_task1_fixed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional deterministic per-compiler limit for a smoke test",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "ablation_cases.jsonl"
    error_log = output_dir / "ablation_errors.jsonl"
    report_path = output_dir / "ablation_report.json"
    summary_path = output_dir / "ablation_summary.csv"

    index = family.report_index()
    labels, family_by_label = family.candidate_metadata(index)
    candidate_families = set(family_by_label.values())
    selection: dict[str, list[tuple[str, str]]] = {}
    for compiler in COMPILERS:
        valid = sorted(
            coordinate
            for coordinate, report in index[compiler].items()
            if coordinate[1] in OPTIMIZATIONS
            and report.with_name(
                report.name.replace(".report.txt", ".features.jsonl.gz")
            ).is_file()
            and family.truth_path(coordinate[0], compiler, coordinate[1]).is_file()
        )
        selection[compiler] = valid[: args.limit] if args.limit else valid

    # The corpus selection is intentionally excluded.  Per-ELF factorial
    # rows remain valid when additional matching results are imported.
    signature = fixed.canonical_hash(
        {
            "schema_version": 3,
            "protocol": "task1_fixed_family_factorial_ablation",
            "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
            "configuration": fixed.PARAMS,
            "specifications": [
                [name, sorted(modules)] for name, modules in SPECS
            ],
            "candidate_families": sorted(candidate_families),
        }
    )
    tasks: list[dict[str, str]] = []
    for compiler in COMPILERS:
        for program, optimization in selection[compiler]:
            report = index[compiler][(program, optimization)]
            tasks.append(
                {
                    "run_signature": signature,
                    "case_id": f"{program}__{compiler}__{optimization}",
                    "compiler": compiler,
                    "program": program,
                    "optimization": optimization,
                    "feature": str(
                        report.with_name(
                            report.name.replace(
                                ".report.txt", ".features.jsonl.gz"
                            )
                        )
                    ),
                    "truth": str(
                        family.truth_path(program, compiler, optimization)
                    ),
                }
            )

    legacy_signatures = compatible_report_signatures(
        report_path, len(candidate_families)
    )
    completed, migrated = load_checkpoint(
        checkpoint, signature, legacy_signatures
    )
    if migrated:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as stream:
            for row in migrated:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"migrated {len(migrated)} compatible checkpoint row(s) "
            "to the selection-independent signature",
            flush=True,
        )
    spec_names = {name for name, _ in SPECS}
    completed_cases = {
        task["case_id"]
        for task in tasks
        if {
            configuration
            for case_id, configuration in completed
            if case_id == task["case_id"]
        } == spec_names
    }
    pending = [task for task in tasks if task["case_id"] not in completed_cases]
    print(
        json.dumps(
            {
                "run_signature": signature,
                "total_elfs": len(tasks),
                "resumed_elfs": len(completed_cases),
                "pending_elfs": len(pending),
                "configurations": len(SPECS),
                "candidate_builds": len(labels),
                "candidate_families": len(candidate_families),
                "workers": args.workers,
                "checkpoint": str(checkpoint),
                "error_log": str(error_log),
                "report": str(report_path),
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
            multiprocessing.Pool(
                args.workers,
                initializer=init_worker,
                initargs=(candidate_families,),
            ) as pool,
        ):
            for number, result in enumerate(
                pool.imap_unordered(evaluate_case_safe, pending, chunksize=1),
                start=1,
            ):
                if result["ok"]:
                    for row in result["rows"]:
                        key = (row["case_id"], row["configuration"])
                        completed[key] = row
                        checkpoint_stream.write(
                            json.dumps(row, sort_keys=True) + "\n"
                        )
                    checkpoint_stream.flush()
                else:
                    error = result["error"]
                    run_errors[error["case_id"]] = error
                    error_stream.write(json.dumps(error, sort_keys=True) + "\n")
                    error_stream.flush()
                    print(
                        f"ERROR {error['case_id']}: "
                        f"{error['error_type']}: {error['error']}",
                        flush=True,
                    )
                if number % 10 == 0 or number == len(pending):
                    done = len(
                        {
                            case_id
                            for case_id, configuration in completed
                            if configuration == "FULL"
                        }
                    )
                    print(
                        f"progress {done}/{len(tasks)} ELF "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )

    all_rows = list(completed.values())
    completed_after = {
        case_id for case_id, configuration in completed if configuration == "FULL"
    }
    missing = [
        {
            **{key: task[key] for key in (
                "case_id", "compiler", "program", "optimization", "feature"
            )},
            "error": run_errors.get(task["case_id"]),
        }
        for task in tasks
        if task["case_id"] not in completed_after
    ]
    summary, pooled = summarize(all_rows, COMPILERS)
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    report = {
        "schema_version": 1,
        "run_signature": signature,
        "protocol": "fixed-parameter pure factorial B/H/S/X/R ablation",
        "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
        "metric_protocol": (
            "Task 1: one decision per ELF and logical archive family; all "
            "build/version labels collapse before TP/TN/FP/FN"
        ),
        "retuned": False,
        "complete": not missing,
        "evaluated_elfs": len(completed_after),
        "total_elfs": len(tasks),
        "missing_cases": missing,
        "candidate_builds": len(labels),
        "candidate_families": len(candidate_families),
        "configuration_full": fixed.PARAMS,
        "components": {
            "B": "base block evidence and coverage thresholds (always enabled)",
            "H": "assignment quality and assignment ratio",
            "S": "call-edge, function concentration/spread, and CU function coverage",
            "X": "cross-CU call evidence",
            "R": ".rodata filter and bonus",
        },
        "configuration_count": len(SPECS),
        "pooled": pooled,
        "summary": summary,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "errors": str(error_log),
            "summary_csv": str(summary_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("POOLED RESULTS", flush=True)
    for name, _modules in SPECS:
        row = pooled[name]
        print(
            f"{name:11} F1={float(row['f1']):.6f} "
            f"P={float(row['precision']):.6f} "
            f"R={float(row['recall']):.6f} "
            f"FP={int(row['FP'])} FN={int(row['FN'])}",
            flush=True,
        )
    print(f"Report written to {report_path}", flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
