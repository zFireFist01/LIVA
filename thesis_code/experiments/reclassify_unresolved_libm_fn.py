#!/usr/bin/env python3
"""Reclassify unresolved linked libm as FN in same-build configuration.

This is a read-only replay of existing per-ELF classifications. It does not
rerun matching or alter the v4 source report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import evaluate_full_corpus_constrained_positive as protocol


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "libseeker-unified/full_corpus_constrained_positive_v4"
TARGET = ROOT / "libseeker-unified/full_corpus_constrained_positive_v5"
TASK1 = ROOT / "libseeker-unified"


def main() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {TARGET}")
    old_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    task1_report = json.loads((TASK1 / "family_f1_fixed_all.json").read_text(encoding="utf-8"))
    task1_fn: dict[str, set[str]] = {}
    with (TASK1 / "family_f1_fixed_all.cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["run_signature"] == task1_report["run_signature"]:
                task1_fn[row["case_id"]] = set(row["false_negative_families"])

    signature = hashlib.sha256(json.dumps({
        "parent_signature": old_report["run_signature"],
        "rule": "unresolved linked libm counts as family-only FN in same build configuration",
        "task1_signature": task1_report["run_signature"],
    }, sort_keys=True).encode()).hexdigest()
    completed: dict[str, dict] = {}
    affected = []
    with (SOURCE / "cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["run_signature"] != old_report["run_signature"]:
                continue
            row["run_signature"] = signature
            if "libm" in row["excluded_out_of_scope_families"]:
                if "libm" not in task1_fn.get(row["case_id"], set()):
                    raise ValueError(f"libm is not an FN in Task 1: {row['case_id']}")
                same = row["scenarios"]["same_build_configuration"]
                if ("libm" in same["true_positive_families"]
                        or "libm" in same["false_negative_families"]):
                    raise ValueError(f"libm already classified: {row['case_id']}")
                same["FN"] += 1
                same["false_negative_families"].append("libm")
                same["false_negative_families"].sort()
                same["family_only_fn"] = 1
                same["family_only_fn_families"] = ["libm"]
                row["same_build_exception_families"] = ["libm"]
                affected.append(row["case_id"])
            completed[row["case_id"]] = row

    if len(completed) != old_report["evaluated_elfs"] or len(affected) != 15:
        raise ValueError(f"Unexpected corpus size: cases={len(completed)}, libm={len(affected)}")
    summaries, diagnostics, transitions = protocol.aggregate(completed)
    old_same = old_report["scenarios"]["same_build_configuration"]["strict"]
    new_same = summaries["same_build_configuration"]["strict"]
    if not (new_same["TP"] == old_same["TP"]
            and new_same["FP"] == old_same["FP"]
            and new_same["TN"] == old_same["TN"]
            and new_same["FN"] == old_same["FN"] + 15
            and new_same["off_scenario_only"] == old_same["off_scenario_only"]):
        raise ValueError("Unexpected same-configuration metric change")
    for scenario in protocol.SCENARIOS:
        if scenario != "same_build_configuration":
            for key in ("TP", "TN", "FP", "FN"):
                if summaries[scenario]["strict"][key] != old_report["scenarios"][scenario]["strict"][key]:
                    raise ValueError(f"Unexpected change in {scenario}.{key}")

    report = dict(old_report)
    report.update({
        "schema_version": 5,
        "run_signature": signature,
        "parent_run_signature": old_report["run_signature"],
        "scenarios": summaries,
        "diagnostics": diagnostics,
        "transitions": transitions,
    })
    report["protocol"] = {
        **old_report["protocol"],
        "family_only_fn_exception": "15 unresolved system libm links are counted as FN in same build configuration, following Task 1 family-level classification",
        "out_of_scope": "unresolved linked archives remain excluded, except the 15 libm family-only FN in same build configuration",
    }
    report["diagnostics"]["family_only_fn_families"] = [
        {"family": "libm", "elf_count": len(affected)}
    ]
    TARGET.mkdir(parents=True)
    with (TARGET / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case_id in sorted(completed):
            stream.write(json.dumps(completed[case_id], sort_keys=True) + "\n")
    (TARGET / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (TARGET / "thesis_report.md").write_text(
        protocol.render_markdown(report), encoding="utf-8"
    )
    fields = [
        "scenario", "view", "TP", "TN", "FP", "FN", "precision", "recall",
        "f1", "coverage", "eligible_positives", "unavailable_positives",
        "family_only_fn", "off_scenario_only", "undetected_on_any_build",
        "eligible_positive_builds",
    ]
    with (TARGET / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario, views in summaries.items():
            for view, values in views.items():
                writer.writerow({"scenario": scenario, "view": view, **values})
    print(json.dumps({
        "output": str(TARGET),
        "reclassified_libm": len(affected),
        "same_build_configuration": new_same,
    }, indent=2))


if __name__ == "__main__":
    main()
