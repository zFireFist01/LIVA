#!/usr/bin/env python3
"""Explain every Task-2 FN without changing matching parameters."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_fixed_family_f1 as fixed  # noqa: E402
import tune_family_f1_optuna as family  # noqa: E402
from audit_cu_source_equivalence import enrich_label_metadata  # noqa: E402
from evaluate_fixed_two_tasks import (  # noqa: E402
    BLOCK_CU_MARKER, LABEL_RE, NAME_RE, report_index_all,
)


Identity = tuple[str, str, str]
_IDENTITY_BY_LABEL_CU: dict[tuple[str, str], Identity] = {}
_CODE_ALIAS: dict[tuple[str, str], set[str]] = {}


def parse_identity(value: str) -> Identity:
    archive_family, cu, source = value.split("|", 2)
    return archive_family, cu, source


def init_worker(
    identity_by_label_cu: dict[tuple[str, str], Identity],
    code_alias: dict[tuple[str, str], set[str]],
) -> None:
    global _IDENTITY_BY_LABEL_CU, _CODE_ALIAS
    _IDENTITY_BY_LABEL_CU = identity_by_label_cu
    _CODE_ALIAS = code_alias


GATES = (
    ("coverage_mean", "block_coverage_mean_threshold"),
    ("coverage_ratio", "block_min_coverage_ratio"),
    ("assignment_quality", "block_assignment_quality_threshold"),
    ("assignment_ratio", "block_min_assignment_ratio"),
    ("call_edge_ratio", "block_min_call_edge_ratio"),
    ("function_concentration", "block_min_function_concentration"),
    ("function_spread", "block_min_function_spread"),
    ("function_coverage", "cu_min_function_coverage"),
)


def evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    false_negatives = {parse_identity(value) for value in task["fn"]}
    observations = {
        identity: {
            "records": 0,
            "target_function_counts": set(),
            "windows": 0,
            "deepest_gate": -1,
            "structural_pass": False,
            "rodata_pass": False,
            "aliases": set(),
        }
        for identity in false_negatives
    }
    aliases_wanted: dict[tuple[str, str], list[Identity]] = defaultdict(list)
    for identity in false_negatives:
        if identity[2].startswith("sha256:"):
            aliases_wanted[(identity[0], identity[2])].append(identity)

    params = {**family.engine.DECISION_DEFAULTS, **fixed.PARAMS}
    with gzip.open(task["feature"], "rb") as stream:
        for raw in stream:
            if BLOCK_CU_MARKER not in raw:
                continue
            label_match = LABEL_RE.search(raw)
            # ``best_function_match.reference_function.name`` precedes the
            # top-level CU name in serialized records. Search only after the
            # top-level library field to avoid mistaking a function for a CU.
            name_match = (
                NAME_RE.search(raw, label_match.end())
                if label_match is not None else None
            )
            if label_match is None or name_match is None:
                continue
            label = label_match.group(1).decode("utf-8", "replace")
            name = family.engine.normalize_cu_name(
                name_match.group(1).decode("utf-8", "replace")
            )
            identity = _IDENTITY_BY_LABEL_CU.get((label, name))
            if identity is None:
                continue
            observation = observations.get(identity)
            alias_targets = aliases_wanted.get((identity[0], identity[2]), [])
            if observation is None and not alias_targets:
                continue
            payload = json.loads(raw)
            if observation is not None:
                observation["records"] += 1
                observation["target_function_counts"].add(
                    int(payload.get("target_function_count", 0))
                )
                windows = list(payload.get("windows", []))
                observation["windows"] += len(windows)
                for window in windows:
                    deepest = -1
                    for index, (field, threshold) in enumerate(GATES):
                        if float(window.get(field, 0.0)) < float(params[threshold]):
                            break
                        deepest = index
                    observation["deepest_gate"] = max(
                        observation["deepest_gate"], deepest
                    )
                    if deepest == len(GATES) - 1:
                        observation["structural_pass"] = True
                        compact = fixed.compact_record(payload)
                        if compact is not None and not family.engine.rodata_is_penalty(
                            compact, params
                        ):
                            observation["rodata_pass"] = True
            for expected in alias_targets:
                if identity[1] != expected[1]:
                    observations[expected]["aliases"].add(identity[1])

    categories = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for identity, observation in observations.items():
        if observation["records"] == 0:
            category = (
                "record_absent_alias_available"
                if observation["aliases"] else "record_absent"
            )
        elif observation["target_function_counts"] and max(
            observation["target_function_counts"]
        ) < 2:
            category = "candidate_below_two_functions"
        elif observation["windows"] == 0:
            category = "record_present_no_windows"
        elif observation["deepest_gate"] < 0:
            category = "failed_coverage_mean"
        elif not observation["structural_pass"]:
            category = "failed_" + GATES[observation["deepest_gate"] + 1][0]
        elif not observation["rodata_pass"]:
            category = "failed_rodata"
        else:
            # This should have been accepted and is therefore an evaluator
            # consistency failure, not a matcher false negative.
            category = "unexpected_record_pass"
        categories[category] += 1
        families[category][identity[0]] += 1
        if len(examples[category]) < 20:
            suffix = ""
            if observation["aliases"]:
                suffix = " aliases=" + ",".join(sorted(observation["aliases"]))
            examples[category].append("|".join(identity) + suffix)
    return {
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "fn": len(false_negatives),
        "categories": dict(categories),
        "families": {
            category: dict(counts) for category, counts in families.items()
        },
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--cases", type=Path,
        default=UNIFIED / "task2_evaluable/task2_evaluable_cases.jsonl",
    )
    parser.add_argument(
        "--source-audit", type=Path,
        default=UNIFIED / "cu_source_equivalence_audit.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=UNIFIED / "task2_evaluable/fn_audit",
    )
    args = parser.parse_args()

    label_meta = enrich_label_metadata()
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    identity_by_label_cu: dict[tuple[str, str], Identity] = {}
    code_alias: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in source_audit["mappings"]:
        source = (
            f"sha256:{row['source_code_sha256']}"
            if row["status"] == "mapped"
            else f"unverified-source-version:{row['source_version']}"
        )
        identity = str(row["family"]), str(row["cu"]), source
        for label in row["build_labels"]:
            identity_by_label_cu[(str(label), str(row["cu"]))] = identity
        if source.startswith("sha256:"):
            code_alias[(identity[0], source)].add(identity[1])

    reports = report_index_all()
    feature_by_coordinate = {
        (compiler, program, optimization): report.with_name(
            report.name.replace(".report.txt", ".features.jsonl.gz")
        )
        for compiler, compiler_reports in reports.items()
        for (program, optimization), report in compiler_reports.items()
    }
    tasks = []
    with args.cases.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row["fn"]:
                continue
            tasks.append({
                **row,
                "feature": str(feature_by_coordinate[
                    (row["compiler"], row["program"], row["optimization"])
                ]),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "fn_audit_cases.jsonl"
    total = Counter()
    family_total: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    completed_ids: set[str] = set()
    if cases_path.is_file():
        with cases_path.open(encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                result = json.loads(line)
                completed_ids.add(str(result["case_id"]))
                total.update(result["categories"])
                for category, counts in result["families"].items():
                    family_total[category].update(counts)
                for category, values in result["examples"].items():
                    remaining = 30 - len(examples[category])
                    if remaining > 0:
                        examples[category].extend(values[:remaining])
    pending = [task for task in tasks if task["case_id"] not in completed_ids]
    completed = len(completed_ids)
    print(
        f"resume {completed}/{len(tasks)} pending={len(pending)}",
        flush=True,
    )
    with (
        multiprocessing.Pool(
            args.workers,
            initializer=init_worker,
            initargs=(identity_by_label_cu, dict(code_alias)),
        ) as pool,
        cases_path.open("a", encoding="utf-8") as output,
    ):
        for result in pool.imap_unordered(evaluate_task, pending, chunksize=1):
            completed += 1
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            total.update(result["categories"])
            for category, counts in result["families"].items():
                family_total[category].update(counts)
            for category, values in result["examples"].items():
                remaining = 30 - len(examples[category])
                if remaining > 0:
                    examples[category].extend(values[:remaining])
            if completed % 25 == 0 or completed == len(tasks):
                print(f"progress {completed}/{len(tasks)}", flush=True)

    report = {
        "schema_version": 1,
        "audited_cases": len(tasks),
        "audited_false_negatives": sum(total.values()),
        "configuration": fixed.PARAMS,
        "categories": dict(total),
        "categories_by_family": {
            category: dict(counts.most_common())
            for category, counts in family_total.items()
        },
        "examples": dict(examples),
        "checks": {
            "record_presence": True,
            "candidate_function_count": True,
            "normalized_name_alias_within_family_and_code_hash": True,
            "fixed_gate_progression": [field for field, _threshold in GATES],
            "rodata_penalty": True,
        },
        "artifacts": {"cases": str(cases_path)},
    }
    report_path = args.output_dir / "fn_audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(report_path),
        "audited_false_negatives": report["audited_false_negatives"],
        "categories": report["categories"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
