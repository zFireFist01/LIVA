#!/usr/bin/env python3
"""Reclassify Task 2 over the domain the matcher can actually evaluate.

This consumes the existing fixed-parameter matching checkpoint.  It changes
neither matching nor thresholds.  Expected CUs are evaluable only when the
exact linked object has at least two functions and a source-equivalent CU is
present in the candidate universe.  C/C++ source equivalence uses normalized
code hashes from the source audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
from audit_cu_source_equivalence import enrich_label_metadata  # noqa: E402


Identity = tuple[str, str, str]


def render(identity: Identity) -> str:
    return "|".join(identity)


def metric(counts: Counter[str]) -> dict[str, int | float]:
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cases", type=Path,
        default=UNIFIED / "task2_source_verified/task2_cases.jsonl",
    )
    parser.add_argument(
        "--input-report", type=Path,
        default=UNIFIED / "task2_source_verified/task2_report.json",
    )
    parser.add_argument(
        "--source-audit", type=Path,
        default=UNIFIED / "cu_source_equivalence_audit.json",
    )
    parser.add_argument(
        "--function-counts", type=Path,
        default=UNIFIED / "ground_truth/ground_truth_function_counts.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=UNIFIED / "task2_evaluable",
    )
    args = parser.parse_args()

    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    function_audit = json.loads(args.function_counts.read_text(encoding="utf-8"))
    previous_report = json.loads(args.input_report.read_text(encoding="utf-8"))
    function_counts = {
        row["archive_sha256"]: row["members"]
        for row in function_audit["archives"]
    }
    candidate_families = {
        metadata["family"] for metadata in enrich_label_metadata().values()
    }

    source_hashes: dict[tuple[str, str, str], str] = {}
    raw_to_code: dict[Identity, str] = {}
    universe: set[Identity] = set()
    for row in source_audit["mappings"]:
        archive_family = str(row["family"])
        cu = str(row["cu"])
        source = str(row["source_version"])
        if row["status"] == "mapped":
            digest = str(row["source_code_sha256"])
            source_hashes[(archive_family, cu, source)] = digest
            source_identity = f"sha256:{digest}"
            for raw_digest in row.get("candidates_by_file_sha256", {}):
                raw_to_code[(archive_family, cu, str(raw_digest))] = digest
        else:
            source_identity = f"unverified-source-version:{source}"
        universe.add((archive_family, cu, source_identity))

    def convert_old(value: str) -> Identity:
        archive_family, cu, source_identity = value.split("|", 2)
        if source_identity.startswith("sha256:"):
            raw_digest = source_identity.removeprefix("sha256:")
            code_digest = raw_to_code.get((archive_family, cu, raw_digest))
            if code_digest:
                source_identity = f"sha256:{code_digest}"
        return archive_family, cu, source_identity

    def source_identity(archive_family: str, cu: str, source: str) -> Identity:
        normalized_cu = family.engine.normalize_cu_name(cu)
        digest = source_hashes.get((archive_family, normalized_cu, source))
        identity = (
            f"sha256:{digest}" if digest
            else f"unverified-source-version:{source}"
        )
        return archive_family, normalized_cu, identity

    def expected_statuses(truth: Path) -> dict[Identity, str]:
        payload = json.loads(truth.read_text(encoding="utf-8"))
        observations: dict[Identity, list[str]] = defaultdict(list)
        for archive in payload.get("archives", []):
            if archive.get("confirmed_compilation_units_method") != "linker_map":
                continue
            archive_name = str(
                archive.get("archive_basename")
                or Path(str(archive.get("archive", ""))).name
            )
            archive_family = family.normalize_archive_family(
                archive_name, candidate_families
            )
            if archive_family is None:
                continue
            source = str(archive.get("source") or "unknown")
            member_counts = function_counts.get(
                str(archive.get("archive_sha256") or ""), {}
            )
            for member in archive.get("confirmed_compilation_units", []):
                values = member_counts.get(str(member), [])
                if not values:
                    status = "function_count_unknown"
                elif any(value >= 2 for value in values) and any(
                    value < 2 for value in values
                ):
                    status = "function_count_ambiguous"
                elif any(value >= 2 for value in values):
                    status = "eligible_functions"
                else:
                    status = "monofunction_or_zero"
                observations[source_identity(
                    archive_family, str(member), source
                )].append(status)
        result: dict[Identity, str] = {}
        for identity, statuses in observations.items():
            if "eligible_functions" in statuses:
                result[identity] = "eligible_functions"
            elif "function_count_ambiguous" in statuses:
                result[identity] = "function_count_ambiguous"
            elif "function_count_unknown" in statuses:
                result[identity] = "function_count_unknown"
            else:
                result[identity] = "monofunction_or_zero"
        return result

    # The input checkpoint is append-only and can contain rows produced by
    # older parameter configurations.  Only consume rows belonging to the
    # report being reclassified; otherwise a failed current case could be
    # silently replaced by a stale result with the same case_id.
    input_run_signature = str(previous_report.get("run_signature") or "")
    if not input_run_signature:
        raise ValueError("Input Task-2 report has no run_signature")

    old_rows: dict[str, dict[str, Any]] = {}
    with args.input_cases.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if str(row.get("run_signature") or "") != input_run_signature:
                    continue
                old_rows[str(row["case_id"])] = row

    output_rows: list[dict[str, Any]] = []
    pooled: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    exclusion_families: dict[str, Counter[str]] = defaultdict(Counter)
    for old in old_rows.values():
        predicted = {
            convert_old(value) for value in old["tp"] + old["fp"]
        }
        statuses = expected_statuses(
            family.truth_path(
                str(old["program"]), str(old["compiler"]),
                str(old["optimization"]),
            )
        )
        eligible = {
            identity for identity, status in statuses.items()
            if status == "eligible_functions" and identity in universe
        }
        mono = {
            identity for identity, status in statuses.items()
            if status == "monofunction_or_zero"
        }
        count_uncertain = {
            identity for identity, status in statuses.items()
            if status in {"function_count_unknown", "function_count_ambiguous"}
        }
        no_candidate = {
            identity for identity, status in statuses.items()
            if status == "eligible_functions" and identity not in universe
        }
        excluded = mono | count_uncertain
        predicted -= excluded
        tp = predicted & eligible
        fp = predicted - eligible
        fn = eligible - predicted
        case_universe = universe - excluded
        tn = case_universe - predicted - eligible
        counts = Counter(TP=len(tp), TN=len(tn), FP=len(fp), FN=len(fn))
        pooled.update(counts)
        categories = {
            "not_evaluable_monofunction_or_zero": mono,
            "not_evaluable_function_count_uncertain": count_uncertain,
            "not_evaluable_no_source_equivalent_candidate": no_candidate,
        }
        for category, identities in categories.items():
            exclusions[category] += len(identities)
            for identity in identities:
                exclusion_families[category][identity[0]] += 1
        output_rows.append({
            "case_id": old["case_id"],
            "compiler": old["compiler"],
            "program": old["program"],
            "optimization": old["optimization"],
            **counts,
            "expected_evaluable": len(eligible),
            "predicted_evaluable_domain": len(predicted),
            "tp": sorted(render(value) for value in tp),
            "fp": sorted(render(value) for value in fp),
            "fn": sorted(render(value) for value in fn),
            "exclusions": {
                category: sorted(render(value) for value in identities)
                for category, identities in categories.items()
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "task2_evaluable_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as stream:
        for row in sorted(output_rows, key=lambda value: value["case_id"]):
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    per_compiler = []
    for compiler in family.COMPILERS:
        counts: Counter[str] = Counter()
        selected = [row for row in output_rows if row["compiler"] == compiler]
        for row in selected:
            counts.update({key: int(row[key]) for key in ("TP", "TN", "FP", "FN")})
        per_compiler.append({
            "compiler": compiler, "evaluated_elfs": len(selected), **metric(counts)
        })
    report = {
        "schema_version": 1,
        "task": "CU discrimination over the matcher-evaluable domain",
        "complete": previous_report.get("complete", False),
        "evaluated_elfs": len(output_rows),
        "missing_cases": previous_report.get("missing_cases", []),
        "configuration": previous_report["configuration"],
        "input_run_signature": input_run_signature,
        "equivalence_rule": source_audit["equivalence_rule"],
        "family_identity_rule": "family remains part of the Task-2 identity",
        "evaluable_rule": (
            "exact linked archive member has at least two functions and a "
            "source-equivalent candidate identity exists"
        ),
        "function_count_rule": function_audit["pipeline_rule"],
        "candidate_universe": len(universe),
        "pooled": metric(pooled),
        "per_compiler": per_compiler,
        "excluded_occurrences": dict(exclusions),
        "excluded_occurrences_by_family": {
            category: dict(counts.most_common())
            for category, counts in exclusion_families.items()
        },
        "artifacts": {
            "cases": str(cases_path),
            "source_audit": str(args.source_audit),
            "function_counts": str(args.function_counts),
            "input_cases": str(args.input_cases),
        },
    }
    report_path = args.output_dir / "task2_evaluable_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(report_path),
        "cases": str(cases_path),
        "pooled": report["pooled"],
        "excluded_occurrences": report["excluded_occurrences"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
