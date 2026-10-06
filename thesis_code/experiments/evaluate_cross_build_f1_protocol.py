#!/usr/bin/env python3
"""Unbiased offline F1 evaluation for the Section 5.4 cross-build protocol.

For every ELF, a deterministic candidate corpus is selected for *all* library
families before looking at matcher decisions.  The baseline selector uses the
dataset's seeded package metadata; absent packages use declared `current`
version, the ELF compiler and the ELF optimization.  Cross-build scenarios
change exactly one selector dimension globally:

* optimization target: O0, O2, O3, Os;
* compiler target: GCC 11, GCC 13, Clang 14, Clang 18;
* version-role target: current, minor-alternative, major-alternative.

Existing replay-complete feature logs are used.  No radare2, PalmTree, cache
analysis or binary matching is rerun.  The append-only checkpoint permits
incremental execution when new ELF reports arrive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
from evaluate_fixed_family_f1 import PARAMS, canonical_hash, parse_relevant_features  # noqa: E402
from evaluate_fixed_two_tasks import report_index_all  # noqa: E402

ENGINE = family.engine
COMPILERS = tuple(family.COMPILERS)
OPTIMIZATIONS = tuple(family.OPTIMIZATIONS)
VERSION_ROLES = ("current", "minor-alternative", "major-alternative")

_META: dict[str, dict[str, str]] = {}
_FAMILIES: set[str] = set()


def candidate_metadata() -> dict[str, dict[str, str]]:
    rows = [
        row for row in csv.DictReader(
            (UNIFIED / "library_matrix.tsv").open(encoding="utf-8"),
            delimiter="\t",
        ) if row["status"] == "selected" and row["path"]
    ]
    counts = Counter(row["archive"] for row in rows)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row["archive"]
        if counts[label] > 1:
            identity = "Dataset/builds/libraries/" + row["path"]
            label += "." + hashlib.sha256(identity.encode()).hexdigest()[:16]
        parts = Path(row["path"]).parts
        result[label] = {
            "family": family.family_from_label(label),
            "package": row["package"],
            "role": row["role"],
            "compiler": row["toolchain"],
            "optimization": row["optimization"],
            "source_build": parts[1] if len(parts) > 1 else "unknown",
            "path": row["path"],
        }
    return result


def scenario_names() -> list[str]:
    return (
        ["same_build_configuration"]
        + [f"cross_optimization__to_{value}" for value in OPTIMIZATIONS]
        + [f"cross_compiler__to_{value}" for value in COMPILERS]
        + [f"cross_version__to_{value}" for value in VERSION_ROLES]
    )


def baseline_selectors(truth: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Return the pre-matching package build choices recorded by generation.

    Ground truth retains one archive row for every seeded package even when the
    archive contributed no CU.  Consequently this selector is uniform for
    present and absent families and does not inspect matcher predictions.
    """
    result: dict[str, tuple[str, str, str]] = {}
    for archive in truth.get("archives", []):
        package = str(archive.get("library") or "")
        compiler = str(archive.get("compiler") or "")
        optimization = str(archive.get("optimization") or "")
        role = str(archive.get("version_role") or "")
        if not package or not compiler or not optimization or not role:
            continue
        value = (compiler, optimization, role)
        previous = result.setdefault(package, value)
        if previous != value:
            raise ValueError(
                f"Inconsistent seeded build selection for {package}: "
                f"{previous} versus {value}"
            )
    return result


def selector_for(
    scenario: str,
    package: str,
    truth: dict[str, Any],
    baselines: dict[str, tuple[str, str, str]],
) -> tuple[str, str, str]:
    compiler, optimization, role = baselines.get(
        package,
        (
            str(truth["compiler"]),
            str(truth.get("elf_optimization", "")),
            "current",
        ),
    )
    if scenario.startswith("cross_optimization__to_"):
        optimization = scenario.split("__to_", 1)[1]
    elif scenario.startswith("cross_compiler__to_"):
        compiler = scenario.split("__to_", 1)[1]
    elif scenario.startswith("cross_version__to_"):
        role = scenario.split("__to_", 1)[1]
    return compiler, optimization, role


def expected_families(truth: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    for archive in truth.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) <= 0:
            continue
        archive_name = str(
            archive.get("archive_basename")
            or Path(str(archive.get("archive", ""))).name
        )
        normalized = family.normalize_archive_family(archive_name, _FAMILIES)
        if normalized is not None:
            expected.add(normalized)
    return expected


def exact_linked_labels(truth: dict[str, Any]) -> set[str]:
    by_path = {value["path"]: label for label, value in _META.items()}
    result: set[str] = set()
    marker = "Dataset/builds/libraries/"
    for archive in truth.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) <= 0:
            continue
        path = str(archive.get("archive", ""))
        relative = path.split(marker, 1)[1] if marker in path else path
        label = by_path.get(relative)
        if label:
            result.add(label)
    return result


def init_worker(meta: dict[str, dict[str, str]], families: set[str]) -> None:
    global _META, _FAMILIES
    _META, _FAMILIES = meta, families


def evaluate_case(task: dict[str, str]) -> dict[str, Any]:
    truth = json.loads(Path(task["truth"]).read_text(encoding="utf-8"))
    header, libraries, calls = parse_relevant_features(Path(task["feature"]))
    params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **PARAMS,
    }
    accepted: set[str] = set()
    scores: dict[str, float] = {}
    for label, records in libraries.items():
        if label not in _META:
            continue
        successful, score = ENGINE.library_match_evidence(records, params, calls)
        scores[label] = float(score)
        if successful and score >= float(params["library_min_score"]):
            accepted.add(label)

    expected = expected_families(truth)
    exact = exact_linked_labels(truth)
    baselines = baseline_selectors(truth)
    scenario_rows: dict[str, dict[str, Any]] = {}
    for scenario in scenario_names():
        corpus = {
            label for label, meta in _META.items()
            if (
                meta["compiler"], meta["optimization"], meta["role"]
            ) == selector_for(scenario, meta["package"], truth, baselines)
        }
        predicted = {_META[label]["family"] for label in accepted & corpus}
        available = {_META[label]["family"] for label in corpus}
        tp = expected & predicted
        fp = predicted - expected
        fn = expected - predicted
        tn = _FAMILIES - expected - predicted
        eligible = expected & available
        scenario_rows[scenario] = {
            "TP": len(tp), "TN": len(tn), "FP": len(fp), "FN": len(fn),
            "expected": len(expected), "candidate_builds": len(corpus),
            "candidate_families": len(available),
            "dr_detected": len(eligible & predicted),
            "dr_eligible": len(eligible),
            "missing_positive_candidates": len(expected - available),
        }

    exact_detected = len(exact & accepted)
    return {
        "run_signature": task["run_signature"],
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "program": task["program"],
        "optimization": task["optimization"],
        "exact_linked_builds": len(exact),
        "exact_linked_detected": exact_detected,
        "scenarios": scenario_rows,
    }


def safe_worker(task: dict[str, str]) -> dict[str, Any]:
    try:
        return {"ok": True, "row": evaluate_case(task)}
    except Exception as exc:
        return {"ok": False, "error": {
            "case_id": task["case_id"], "feature": task["feature"],
            "error_type": type(exc).__name__, "error": str(exc),
        }}


def load_checkpoint(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_signature") == signature:
            completed[str(row["case_id"])] = row
    return completed


def metrics(counts: Counter[str]) -> dict[str, int | float]:
    tp, tn, fp, fn = (counts[k] for k in ("TP", "TN", "FP", "FN"))
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "detection_rate": (
            counts["dr_detected"] / counts["dr_eligible"]
            if counts["dr_eligible"] else 0.0
        ),
        "dr_detected": counts["dr_detected"],
        "dr_eligible": counts["dr_eligible"],
        "missing_positive_candidates": counts["missing_positive_candidates"],
        "candidate_build_observations": counts["candidate_builds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=UNIFIED / "cross_build_f1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cross_build_cases.jsonl"
    errors = args.output_dir / "cross_build_errors.jsonl"

    meta = candidate_metadata()
    families = {value["family"] for value in meta.values()}
    signature = canonical_hash({
        "schema_version": 2,
        "protocol": "unbiased_global_candidate_corpus",
        "params": PARAMS,
        "candidate_metadata": meta,
        "scenarios": scenario_names(),
        "baseline": "ground-truth seeded archive selection for present and absent packages",
        "fallback": "current role + ELF compiler + ELF optimization only for unseeded packages",
    })
    reports = report_index_all()
    tasks: list[dict[str, str]] = []
    for compiler in COMPILERS:
        for (program, optimization), report in sorted(reports[compiler].items()):
            tasks.append({
                "run_signature": signature,
                "case_id": f"{program}__{compiler}__{optimization}",
                "compiler": compiler, "program": program,
                "optimization": optimization,
                "feature": str(report.with_name(report.name.replace(
                    ".report.txt", ".features.jsonl.gz"))),
                "truth": str(family.truth_path(program, compiler, optimization)),
            })
    if args.limit > 0:
        tasks = tasks[:args.limit]
    completed = load_checkpoint(checkpoint, signature)
    pending = [task for task in tasks if task["case_id"] not in completed]
    print(json.dumps({
        "run_signature": signature, "total_elfs": len(tasks),
        "resumed_elfs": len(completed), "pending_elfs": len(pending),
        "candidate_builds": len(meta), "candidate_families": len(families),
        "workers": args.workers, "checkpoint": str(checkpoint),
    }, indent=2), flush=True)

    started = time.monotonic()
    if pending:
        with checkpoint.open("a", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err:
            with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(meta, families)) as pool:
                for index, result in enumerate(pool.imap_unordered(safe_worker, pending), 1):
                    if result["ok"]:
                        row = result["row"]
                        completed[row["case_id"]] = row
                        out.write(json.dumps(row, sort_keys=True) + "\n"); out.flush()
                    else:
                        err.write(json.dumps(result["error"], sort_keys=True) + "\n"); err.flush()
                        print(f"ERROR {result['error']['case_id']}: {result['error']['error']}", flush=True)
                    if index % 10 == 0 or index == len(pending):
                        print(f"progress {len(completed)}/{len(tasks)} elapsed={time.monotonic()-started:.1f}s", flush=True)

    pooled = {name: Counter() for name in scenario_names()}
    exact = Counter()
    compiler_rows: dict[str, dict[str, Counter[str]]] = {
        c: {name: Counter() for name in scenario_names()} for c in COMPILERS
    }
    for row in completed.values():
        exact["linked"] += int(row["exact_linked_builds"])
        exact["detected"] += int(row["exact_linked_detected"])
        for name, values in row["scenarios"].items():
            pooled[name].update(values)
            compiler_rows[row["compiler"]][name].update(values)

    report = {
        "schema_version": 2, "run_signature": signature,
        "complete": len(completed) == len(tasks),
        "evaluated_elfs": len(completed), "requested_elfs": len(tasks),
        "configuration": PARAMS,
        "candidate_corpus_rule": {
            "baseline": "compiler/optimization/version-role from every seeded ground-truth archive row, including absent libraries",
            "unseeded_package_fallback": "current role + ELF optimization + ELF compiler",
            "cross_build": "replace exactly one dimension globally before observing predictions",
        },
        "same_exact_secondary": {
            "detected": exact["detected"], "linked": exact["linked"],
            "detection_rate": exact["detected"] / exact["linked"] if exact["linked"] else 0.0,
        },
        "pooled": {name: metrics(values) for name, values in pooled.items()},
        "by_elf_compiler": {
            compiler: {name: metrics(values) for name, values in scenarios.items()}
            for compiler, scenarios in compiler_rows.items()
        },
    }
    report_path = args.output_dir / "cross_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "cross_build_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "TP", "TN", "FP", "FN", "precision", "recall", "f1", "detection_rate", "dr_eligible", "missing_positive_candidates"])
        writer.writeheader()
        for name, values in report["pooled"].items():
            writer.writerow({"scenario": name, **{key: values[key] for key in writer.fieldnames if key != "scenario"}})
    print(json.dumps({"complete": report["complete"], "evaluated_elfs": len(completed), "report": str(report_path), "summary": str(csv_path), "same_exact_secondary": report["same_exact_secondary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
