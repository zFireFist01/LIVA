#!/usr/bin/env python3
"""Run a fixed-parameter B/H/S/X/R ablation on cached external-test features.

This is a pure ablation: every configuration uses the FULL pipeline's frozen
profile, thresholds, PalmTree features and library panel.  Components are
removed by neutralising only their decision gates; no threshold search or
model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
if str(THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(THESIS_DIR))

import optuna_threshold_search as greedy


TEST_DIR = THESIS_DIR / "Test"
REPO_ROOT = THESIS_DIR.parents[1]
DEFAULT_GREEDY_DIR = TEST_DIR / "greedy_threshold_results_9elf_top3_full"
DEFAULT_TEST_DIR = TEST_DIR / "expanded_test_9elf_top3_final"

CONFIGURATIONS = tuple(
    (
        "FULL" if modules == frozenset("BHSXR") else "_".join(
            component for component in "BHSXR" if component in modules
        ),
        modules,
        "Complete current pipeline" if modules == frozenset("BHSXR")
        else " + ".join(component for component in "BHSXR" if component in modules),
    )
    for size in range(5)
    for optional in itertools.combinations("HSXR", size)
    for modules in (frozenset(("B", *optional)),)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-dir", type=Path, default=DEFAULT_GREEDY_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=REPO_ROOT / "GroundTruth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEST_DIR / "frozen_ablation_results",
    )
    parser.add_argument("--cu-function-coverage", type=float, default=0.60)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def archive_names(report: dict[str, Any]) -> list[str]:
    return sorted(
        str(archive["name"])
        for archive in (
            list(report.get("positive_archives", []))
            + list(report.get("negative_archives", []))
        )
    )


def load_cases_for_variants(
    ground_truth_dir: Path,
    variants: list[str],
) -> list[dict[str, Any]]:
    """Load exact ELF variants without depending on experiment-only helpers."""
    return greedy.load_cases(
        SimpleNamespace(
            all_programs=False,
            ground_truth_dir=ground_truth_dir,
            elf_variants=variants,
            program=None,
            validation_program=[],
            test_program=[],
            compiler=None,
        )
    )


def ablated_params(
    full_params: dict[str, Any], modules: frozenset[str]
) -> dict[str, Any]:
    """Neutralise only components absent from ``modules``."""
    params = dict(full_params)
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
        params["cross_cu_call_saturation_edges"] = 0
    if "R" not in modules:
        params["rodata_filter_enabled"] = 0
        params["rodata_bonus_weight"] = 0.0
    return params


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_metrics(
    predictions: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    rows = []
    values = sorted({str(prediction[key]) for prediction in predictions})
    for value in values:
        selected = [
            prediction
            for prediction in predictions
            if str(prediction[key]) == value
        ]
        counts = {
            label.lower(): sum(
                prediction["classification"] == label
                for prediction in selected
            )
            for label in ("TP", "TN", "FP", "FN")
        }
        metrics = greedy.calculate_metrics(**counts)
        rows.append({key: value, **metrics})
    return rows


def main() -> int:
    args = parse_args()
    greedy_dir = args.greedy_dir.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_json(greedy_dir / "best_thresholds.json")
    test_selection = load_json(test_dir / "test_selection.json")
    profile_id = str(report["best_profile_id"])
    candidates = archive_names(report)
    variants = [str(case["variant"]) for case in test_selection["cases"]]
    cases = load_cases_for_variants(
        args.ground_truth_dir.resolve(), variants
    )
    structural_profile = {
        key: report["parameters"][key]
        for key in (
            "block_threshold",
            "block_locality_window_multiplier",
            "block_locality_window_padding",
        )
    }
    collection = greedy.parse_feature_collection(
        structural_profile,
        test_dir / "reports" / "current",
        len(cases),
        "fixed-parameter-ablation",
        candidates,
    )

    full_params = {
        **report["parameters"],
        "library_score_aggregator": "top3_mean",
        "rodata_penalty_threshold": 0.0,
        "rodata_confirm_threshold": 0.70,
        "rodata_bonus_weight": 0.30,
        "rodata_filter_enabled": 1,
        "cu_min_function_coverage": args.cu_function_coverage,
    }

    results: list[dict[str, Any]] = []
    predictions_output: list[dict[str, Any]] = []
    library_output: list[dict[str, Any]] = []
    program_output: list[dict[str, Any]] = []
    parameters_output: list[dict[str, Any]] = []
    for name, modules, description in CONFIGURATIONS:
        params = ablated_params(full_params, modules)
        metrics, predictions = greedy.evaluate(
            params, collection, cases, candidates
        )
        results.append(
            {
                "configuration": name,
                "modules": "+".join(sorted(modules)),
                "description": description,
                **metrics,
            }
        )
        parameters_output.append(
            {
                "configuration": name,
                **params,
            }
        )
        enriched = [
            {
                "configuration": name,
                "library_family": greedy.archive_family(
                    str(prediction["library"])
                ),
                **prediction,
            }
            for prediction in predictions
        ]
        predictions_output.extend(enriched)
        library_output.extend(
            {
                "configuration": name,
                **row,
            }
            for row in grouped_metrics(enriched, "library_family")
        )
        program_output.extend(
            {
                "configuration": name,
                **row,
            }
            for row in grouped_metrics(enriched, "program")
        )

    full = next(row for row in results if row["configuration"] == "FULL")
    for row in results:
        row["delta_lib_f1_vs_full"] = (
            float(row["lib_f1"]) - float(full["lib_f1"])
        )
        row["delta_lib_precision_vs_full"] = (
            float(row["lib_precision"]) - float(full["lib_precision"])
        )
        row["delta_lib_recall_vs_full"] = (
            float(row["lib_recall"]) - float(full["lib_recall"])
        )

    metadata = {
        "protocol": "fixed-parameter pure ablation",
        "retuned": False,
        "profile_id": profile_id,
        "test_dir": str(test_dir),
        "elf_count": len(cases),
        "library_count": len(candidates),
        "pair_count": len(cases) * len(candidates),
        "full_parameters": full_params,
        "configurations": [
            {
                "name": name,
                "modules": sorted(modules),
                "description": description,
            }
            for name, modules, description in CONFIGURATIONS
        ],
        "summary": results,
    }
    (output_dir / "ablation_results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "ablation_summary.csv", results)
    write_csv(output_dir / "ablation_predictions.csv", predictions_output)
    write_csv(output_dir / "ablation_library_metrics.csv", library_output)
    write_csv(output_dir / "ablation_program_metrics.csv", program_output)
    write_csv(output_dir / "ablation_parameters.csv", parameters_output)

    print(
        f"Fixed ablation: {len(cases)} ELF x {len(candidates)} libraries, "
        f"profile={profile_id}, aggregator=top3_mean"
    )
    for row in results:
        print(
            f"  {row['configuration']:7} "
            f"P={float(row['lib_precision']):.4f} "
            f"R={float(row['lib_recall']):.4f} "
            f"F1={float(row['lib_f1']):.4f} "
            f"FP={int(row['fp']):2d} FN={int(row['fn']):2d} "
            f"delta_F1={float(row['delta_lib_f1_vs_full']):+.4f}"
        )
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
