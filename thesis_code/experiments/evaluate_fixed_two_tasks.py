#!/usr/bin/env python3
"""Replay one fixed configuration on every valid report for two tasks.

Task 1 evaluates logical library-family presence per ELF.
Task 2 evaluates (family, normalized CU name, source variant) per ELF.  Builds
made with a different compiler/optimization but the same recorded source
variant are source-equivalent; an identically named CU from another source
variant is a distinct negative and therefore an FP when predicted.

The detailed JSONL is append-only and resumable.  It records TP/FP/FN
identities (TN is represented by its count because enumerating every negative
for every ELF would make the audit log unnecessarily enormous).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
from evaluate_fixed_family_f1 import (  # noqa: E402
    PARAMS,
    parse_relevant_features,
)


ENGINE = family.engine
COMPILERS = family.COMPILERS
OPTIMIZATIONS = family.OPTIMIZATIONS
LABEL_RE = re.compile(rb'"library":"([^"\\]+)"')
NAME_RE = re.compile(rb'"name":"([^"\\]+)"')
BLOCK_CU_MARKER = b'"type":"block_cu"'

_LABEL_META: dict[str, dict[str, str]] = {}
_CANDIDATE_FAMILIES: set[str] = set()
_CU_UNIVERSE: set[tuple[str, str, str]] = set()


def canonical_hash(payload: Any) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def source_from_matrix_row(row: dict[str, str]) -> str:
    path = Path(row.get("path", ""))
    if len(path.parts) >= 2:
        return path.parts[1]
    candidates = str(row.get("project_candidates", ""))
    return candidates.split("|", 1)[0] if candidates else "unknown"


def candidate_label_metadata() -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in csv.DictReader(
            (UNIFIED / "library_matrix.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
        if row["status"] == "selected" and row["path"]
    ]
    name_counts = Counter(row["archive"] for row in rows)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row["archive"]
        if name_counts[row["archive"]] > 1:
            identity = "Dataset/builds/libraries/" + row["path"]
            label += "." + hashlib.sha256(identity.encode()).hexdigest()[:16]
        result[label] = {
            "family": family.family_from_label(label),
            "source": source_from_matrix_row(row),
            "archive": row["archive"],
            "toolchain": row["toolchain"],
            "optimization": row["optimization"],
        }
    return result


def scan_cu_universe(
    feature: Path, label_meta: dict[str, dict[str, str]]
) -> set[tuple[str, str, str]]:
    universe: set[tuple[str, str, str]] = set()
    with gzip.open(feature, "rb") as stream:
        for raw in stream:
            if BLOCK_CU_MARKER not in raw:
                continue
            label_match = LABEL_RE.search(raw)
            name_match = NAME_RE.search(raw)
            if not label_match or not name_match:
                continue
            label = label_match.group(1).decode("utf-8", "replace")
            metadata = label_meta.get(label)
            if metadata is None:
                continue
            name = name_match.group(1).decode("utf-8", "replace")
            universe.add(
                (
                    metadata["family"],
                    ENGINE.normalize_cu_name(name),
                    metadata["source"],
                )
            )
    if not universe:
        raise ValueError(f"No CU identities found in {feature}")
    return universe


def report_index_all() -> dict[str, dict[tuple[str, str], Path]]:
    result: dict[str, dict[tuple[str, str], Path]] = {}
    for compiler in COMPILERS:
        directory = UNIFIED / compiler / "reports/current"
        current: dict[tuple[str, str], Path] = {}
        for report in sorted(directory.glob("*.report.txt")):
            coordinate = family.coordinate(report)
            if coordinate[1] not in OPTIMIZATIONS:
                continue
            feature = report.with_name(
                report.name.replace(".report.txt", ".features.jsonl.gz")
            )
            truth = family.truth_path(coordinate[0], compiler, coordinate[1])
            if not feature.is_file() or not truth.is_file():
                continue
            if coordinate in current:
                raise ValueError(f"Duplicate report coordinate {compiler} {coordinate}")
            current[coordinate] = report
        result[compiler] = current
    return result


def expected_truth(
    path: Path, candidate_families: set[str]
) -> tuple[set[str], set[tuple[str, str, str]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_families: set[str] = set()
    expected_cus: set[tuple[str, str, str]] = set()
    audit = Counter()
    for archive in payload.get("archives", []):
        confirmed = archive.get("confirmed_compilation_units")
        if confirmed is None:
            units = archive.get("compilation_units") or []
            confirmed = [
                unit["compilation_unit"]
                for unit in units
                if unit.get("included")
                and unit.get("ground_truth_method") == "linker_map"
            ]
        elif archive.get("confirmed_compilation_units_method") != "linker_map":
            confirmed = []

        archive_name = str(
            archive.get("archive_basename")
            or Path(str(archive.get("archive", ""))).name
        )
        archive_family = family.normalize_archive_family(
            archive_name, candidate_families
        )
        if archive_family is None:
            if confirmed:
                audit["confirmed_cus_outside_candidate_families"] += len(confirmed)
            continue
        if int(archive.get("included_compilation_units", 0) or 0) > 0:
            expected_families.add(archive_family)
        source = str(archive.get("source") or "unknown")
        for member in confirmed:
            expected_cus.add(
                (archive_family, ENGINE.normalize_cu_name(str(member)), source)
            )
    audit["expected_families"] = len(expected_families)
    audit["expected_cus"] = len(expected_cus)
    audit["expected_cus_unknown_source"] = sum(
        source == "unknown" for _, _, source in expected_cus
    )
    return expected_families, expected_cus, dict(audit)


def init_worker(
    label_meta: dict[str, dict[str, str]],
    candidate_families: set[str],
    cu_universe: set[tuple[str, str, str]],
) -> None:
    global _LABEL_META, _CANDIDATE_FAMILIES, _CU_UNIVERSE
    _LABEL_META = label_meta
    _CANDIDATE_FAMILIES = candidate_families
    _CU_UNIVERSE = cu_universe


def classify(
    expected: set[Any], predicted: set[Any], universe: set[Any]
) -> tuple[dict[str, int], set[Any], set[Any], set[Any]]:
    tp = predicted & expected
    fp = predicted - expected
    fn = expected - predicted
    evaluated_universe = universe | expected
    tn_count = len(evaluated_universe - predicted - expected)
    return (
        {"TP": len(tp), "FP": len(fp), "FN": len(fn), "TN": tn_count},
        tp,
        fp,
        fn,
    )


def render_cu(identity: tuple[str, str, str]) -> str:
    return "|".join(identity)


def evaluate_case(task: dict[str, str]) -> dict[str, Any]:
    header, libraries, calls = parse_relevant_features(Path(task["feature"]))
    params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **PARAMS,
    }
    predicted_families: set[str] = set()
    predicted_cus: set[tuple[str, str, str]] = set()
    accepted_labels = 0
    for label, records in libraries.items():
        metadata = _LABEL_META.get(label)
        if metadata is None:
            continue
        successful, score = ENGINE.library_match_evidence(records, params, calls)
        if successful and score >= float(params["library_min_score"]):
            predicted_families.add(metadata["family"])
            accepted_labels += 1
        for record in successful:
            name = record.get("name")
            if name:
                predicted_cus.add(
                    (
                        metadata["family"],
                        ENGINE.normalize_cu_name(str(name)),
                        metadata["source"],
                    )
                )

    expected_families, expected_cus, truth_audit = expected_truth(
        Path(task["truth"]), _CANDIDATE_FAMILIES
    )
    task1_counts, task1_tp, task1_fp, task1_fn = classify(
        expected_families, predicted_families, _CANDIDATE_FAMILIES
    )
    task2_counts, task2_tp, task2_fp, task2_fn = classify(
        expected_cus, predicted_cus, _CU_UNIVERSE
    )
    return {
        "run_signature": task["run_signature"],
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "program": task["program"],
        "optimization": task["optimization"],
        "task1": {
            **task1_counts,
            "expected": sorted(expected_families),
            "predicted": sorted(predicted_families),
            "tp": sorted(task1_tp),
            "fp": sorted(task1_fp),
            "fn": sorted(task1_fn),
            "accepted_build_labels": accepted_labels,
        },
        "task2": {
            **task2_counts,
            "expected_count": len(expected_cus),
            "predicted_count": len(predicted_cus),
            "tp": sorted(render_cu(value) for value in task2_tp),
            "fp": sorted(render_cu(value) for value in task2_fp),
            "fn": sorted(render_cu(value) for value in task2_fn),
            "truth_audit": truth_audit,
        },
    }


def metrics(counts: Counter[str]) -> dict[str, int | float]:
    tp, tn, fp, fn = (
        counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    )
    total = tp + tn + fp + fn
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "evaluated_decisions": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def load_checkpoint(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_signature") == signature:
                completed[str(row["case_id"])] = row
    return completed


def summarize_task(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    per_compiler = []
    pooled: Counter[str] = Counter()
    for compiler in COMPILERS:
        selected = [row for row in rows if row["compiler"] == compiler]
        counts: Counter[str] = Counter()
        for row in selected:
            counts.update({key: int(row[task][key]) for key in ("TP", "TN", "FP", "FN")})
        pooled.update(counts)
        per_compiler.append(
            {"compiler": compiler, "evaluated_elfs": len(selected), **metrics(counts)}
        )
    return {
        "per_compiler": per_compiler,
        "pooled": {
            **metrics(pooled),
            "macro_compiler_f1": sum(float(row["f1"]) for row in per_compiler)
            / len(per_compiler),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=UNIFIED / "fixed_two_tasks_all_elfs.summary.json",
    )
    parser.add_argument(
        "--decisions-log",
        type=Path,
        default=UNIFIED / "fixed_two_tasks_all_elfs.decisions.jsonl",
    )
    parser.add_argument(
        "--cu-catalog",
        type=Path,
        default=UNIFIED / "fixed_two_tasks_cu_source_variants.json",
    )
    parser.add_argument("--limit-per-compiler", type=int, default=0)
    args = parser.parse_args()

    label_meta = candidate_label_metadata()
    candidate_families = {value["family"] for value in label_meta.values()}
    reports = report_index_all()
    sample_report = next(iter(reports[COMPILERS[0]].values()))
    sample_feature = sample_report.with_name(
        sample_report.name.replace(".report.txt", ".features.jsonl.gz")
    )
    cu_universe = scan_cu_universe(sample_feature, label_meta)
    args.cu_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.cu_catalog.write_text(
        json.dumps(
            {
                "identity": "family|normalized_cu|recorded_source_variant",
                "source_equivalence": (
                    "same recorded source variant; compiler and optimization ignored"
                ),
                "count": len(cu_universe),
                "values": sorted(render_cu(value) for value in cu_universe),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tasks: list[dict[str, str]] = []
    selected_coordinates: dict[str, list[str]] = {}
    for compiler in COMPILERS:
        compiler_reports = sorted(reports[compiler].items())
        if args.limit_per_compiler > 0:
            compiler_reports = compiler_reports[: args.limit_per_compiler]
        selected_coordinates[compiler] = [
            f"{program}__{optimization}"
            for (program, optimization), _ in compiler_reports
        ]
        for (program, optimization), report in compiler_reports:
            tasks.append(
                {
                    "case_id": f"{program}__{compiler}__{optimization}",
                    "compiler": compiler,
                    "program": program,
                    "optimization": optimization,
                    "feature": str(
                        report.with_name(
                            report.name.replace(".report.txt", ".features.jsonl.gz")
                        )
                    ),
                    "truth": str(family.truth_path(program, compiler, optimization)),
                }
            )
    signature = canonical_hash(
        {
            "configuration": PARAMS,
            "task1": "logical family presence per ELF",
            "task2": "family+normalized CU+recorded source variant per ELF",
            "coordinates": selected_coordinates,
            "cu_universe_hash": canonical_hash(sorted(cu_universe)),
        }
    )
    for task in tasks:
        task["run_signature"] = signature
    completed = load_checkpoint(args.decisions_log, signature)
    pending = [task for task in tasks if task["case_id"] not in completed]
    args.decisions_log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "run_signature": signature,
                "valid_reports": {key: len(value) for key, value in reports.items()},
                "selected_cases": len(tasks),
                "resumed_cases": len(completed),
                "pending_cases": len(pending),
                "candidate_families": len(candidate_families),
                "cu_source_variants": len(cu_universe),
                "workers": args.workers,
                "decisions_log": str(args.decisions_log),
                "summary": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )

    started = time.monotonic()
    if pending:
        with args.decisions_log.open("a", encoding="utf-8") as log_stream:
            with multiprocessing.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(label_meta, candidate_families, cu_universe),
            ) as pool:
                for index, row in enumerate(
                    pool.imap_unordered(evaluate_case, pending, chunksize=1), start=1
                ):
                    completed[row["case_id"]] = row
                    log_stream.write(json.dumps(row, sort_keys=True) + "\n")
                    log_stream.flush()
                    if index % 25 == 0 or index == len(pending):
                        print(
                            f"progress {len(completed)}/{len(tasks)} "
                            f"elapsed={time.monotonic() - started:.1f}s",
                            flush=True,
                        )

    rows = [completed[task["case_id"]] for task in tasks]
    payload = {
        "schema_version": 1,
        "run_signature": signature,
        "configuration": PARAMS,
        "corpus": {
            "all_valid_reports": True,
            "excluded_placeholder_program": None,
            "elfs_per_compiler": Counter(row["compiler"] for row in rows),
            "total_elfs": len(rows),
            "candidate_families": len(candidate_families),
            "cu_source_variants": len(cu_universe),
        },
        "task1_library_family": summarize_task(rows, "task1"),
        "task2_cu_source_variant": {
            "identity": "family|normalized_cu|recorded_source_variant",
            "source_equivalence": (
                "same recorded source variant; compiler and optimization ignored"
            ),
            **summarize_task(rows, "task2"),
        },
        "artifacts": {
            "decisions_log": str(args.decisions_log),
            "cu_catalog": str(args.cu_catalog),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
