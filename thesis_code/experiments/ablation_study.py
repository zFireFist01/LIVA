#!/usr/bin/env python3
"""Offline ablation study for the block/.rodata matching pipeline.

The expensive radare2 and PalmTree stage is not repeated.  This script loads
the feature collections produced by ``optuna_threshold_search.py`` and tunes
the active decision thresholds for all combinations of five components:

    B: PalmTree block coverage (always enabled)
    H: lexicographic Hungarian assignment evidence
    S: intra-CU structural consistency, including function coverage
    X: cross-CU call consistency
    R: .rodata negative evidence

For every configuration, thresholds are tuned on train, the structural profile
is selected on validation, and metrics are computed on an untouched test set.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
if str(THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(THESIS_DIR))

import optuna_threshold_search as greedy


TEST_DIR = THESIS_DIR / "Test"
DEFAULT_GREEDY_DIR = TEST_DIR / "greedy_threshold_results_5elf_v4"
RESULT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class AblationSpec:
    name: str
    modules: frozenset[str]
    description: str


ABLATIONS = tuple(
    AblationSpec(
        "FULL" if modules == frozenset("BHSXR") else "_".join(
            component for component in "BHSXR" if component in modules
        ),
        modules,
        "Complete pipeline" if modules == frozenset("BHSXR")
        else " + ".join(sorted(modules)),
    )
    for size in range(4 + 1)
    for optional in itertools.combinations("HSXR", size)
    for modules in (frozenset(("B", *optional)),)
)
ABLATION_BY_NAME = {spec.name: spec for spec in ABLATIONS}

COVERAGE_PARAMETERS = (
    "block_coverage_mean_threshold",
    "block_min_coverage_ratio",
)
LIBRARY_PARAMETERS = ("library_min_score",)
HUNGARIAN_PARAMETERS = (
    "block_assignment_quality_threshold",
    "block_min_assignment_ratio",
)
STRUCTURE_PARAMETERS = (
    "block_min_call_edge_ratio",
    "block_min_function_concentration",
    "block_min_function_spread",
    "cu_min_function_coverage",
)
CROSS_CU_PARAMETERS = (
    "cross_cu_call_bonus_weight",
    "cross_cu_call_penalty_weight",
    "cross_cu_call_saturation_edges",
)
RODATA_PARAMETERS = (
    "rodata_min_bytes",
    "rodata_min_strings",
    "rodata_min_ngrams",
    "rodata_penalty_threshold",
    "rodata_confirm_threshold",
    "rodata_bonus_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the B/H/S/X/R ablation study offline from greedy-search "
            "feature collections."
        )
    )
    parser.add_argument(
        "--greedy-dir",
        type=Path,
        default=DEFAULT_GREEDY_DIR,
        help="Directory containing input_selection.json and features/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to GREEDY_DIR/ablation_results.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=greedy.DEFAULT_GROUND_TRUTH,
    )
    parser.add_argument(
        "--split-mode",
        choices=("original", "rotating"),
        default="original",
        help=(
            "original reuses the greedy train/validation/test split; rotating "
            "is an exploratory fixed-panel sensitivity analysis that tests every "
            "selected program once with the next program as validation."
        ),
    )
    parser.add_argument(
        "--configuration",
        action="append",
        choices=tuple(ABLATION_BY_NAME),
        help="Run only this configuration. Repeatable; defaults to all sixteen.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--objective-order",
        choices=("cu_first", "lib_first"),
        help="Defaults to the objective saved by the greedy search.",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument(
        "--allow-missing-profiles",
        action="store_true",
        help="Use only feature files currently present (for provisional runs).",
    )
    parser.add_argument(
        "--strict-profiles",
        action="store_true",
        help="Fail instead of excluding a present but incomplete profile.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible per-configuration/per-fold checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_specs(names: list[str] | None) -> list[AblationSpec]:
    if not names:
        return list(ABLATIONS)
    requested = set(names)
    return [spec for spec in ABLATIONS if spec.name in requested]


def parameter_space(
    spec: AblationSpec,
) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    """Return valid defaults and tunable grids for one ablation.

    Removed gates are fixed to their neutral value rather than left in the
    coordinate search.  This makes each run a genuine re-tuning of only the
    components that are available to that configuration.
    """
    params = dict(greedy.DECISION_DEFAULTS)
    active = list(LIBRARY_PARAMETERS + COVERAGE_PARAMETERS)

    if "H" in spec.modules:
        active.extend(HUNGARIAN_PARAMETERS)
    else:
        for parameter in HUNGARIAN_PARAMETERS:
            params[parameter] = 0.0

    if "S" in spec.modules:
        active.extend(STRUCTURE_PARAMETERS)
    else:
        for parameter in STRUCTURE_PARAMETERS:
            params[parameter] = 0.0

    if "X" in spec.modules:
        active.extend(CROSS_CU_PARAMETERS)
    else:
        for parameter in CROSS_CU_PARAMETERS:
            params[parameter] = 0.0

    if "R" in spec.modules:
        params["rodata_filter_enabled"] = 1
        active.extend(RODATA_PARAMETERS)
    else:
        params["rodata_filter_enabled"] = 0

    grids = {
        parameter: tuple(greedy.DECISION_GRIDS[parameter])
        for parameter in greedy.DECISION_GRIDS
        if parameter in active
    }
    return params, grids


def random_params(
    defaults: dict[str, Any],
    grids: dict[str, tuple[Any, ...]],
    rng: random.Random,
) -> dict[str, Any]:
    params = dict(defaults)
    for parameter, candidates in grids.items():
        params[parameter] = rng.choice(candidates)
    return params


def tune_profile(
    spec: AblationSpec,
    collection: dict[str, Any],
    train_cases: list[dict[str, Any]],
    archive_names: list[str],
    objective_order: str,
    restarts: int,
    max_rounds: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    defaults, grids = parameter_space(spec)
    metric_cache: dict[tuple[Any, ...], dict[str, float | int]] = {}

    def metrics_for(params: dict[str, Any]) -> dict[str, float | int]:
        key = tuple(params[name] for name in greedy.DECISION_DEFAULTS)
        if key not in metric_cache:
            metric_cache[key] = greedy.evaluate(
                params, collection, train_cases, archive_names
            )[0]
        return metric_cache[key]

    best_params = dict(defaults)
    best_metrics = metrics_for(best_params)

    rng = random.Random(seed)
    for restart in range(restarts):
        params = (
            dict(defaults)
            if restart == 0
            else random_params(defaults, grids, rng)
        )
        current_metrics = metrics_for(params)

        for _ in range(max_rounds):
            changed = False
            for parameter, candidates in grids.items():
                selected_value = params[parameter]
                selected_metrics = current_metrics
                selected_key = greedy.objective(current_metrics, objective_order)

                for candidate in candidates:
                    trial = dict(params)
                    trial[parameter] = candidate
                    metrics = metrics_for(trial)
                    key = greedy.objective(metrics, objective_order)
                    if key > selected_key:
                        selected_value = candidate
                        selected_metrics = metrics
                        selected_key = key

                if selected_value != params[parameter]:
                    params[parameter] = selected_value
                    current_metrics = selected_metrics
                    changed = True
            if not changed:
                break

        if greedy.objective(
            current_metrics, objective_order
        ) > greedy.objective(best_metrics, objective_order):
            best_params = dict(params)
            best_metrics = current_metrics

    return best_params, best_metrics


def load_selected_cases(
    selection: dict[str, Any], ground_truth_dir: Path
) -> list[dict[str, Any]]:
    variants = [str(value) for value in selection["elf_variants"]]
    loader_args = SimpleNamespace(
        all_programs=False,
        elf_variants=variants,
        program=None,
        validation_program=None,
        test_program=None,
        compiler=None,
        ground_truth_dir=ground_truth_dir.resolve(),
    )
    cases = greedy.load_cases(loader_args)
    by_variant = {case["variant"]: case for case in cases}
    missing = sorted(set(variants) - set(by_variant))
    if missing:
        raise ValueError("Ground truth missing selected variants: " + ", ".join(missing))
    return [by_variant[variant] for variant in variants]


def cases_for_variants(
    cases: list[dict[str, Any]], variants: Iterable[str]
) -> list[dict[str, Any]]:
    by_variant = {case["variant"]: case for case in cases}
    selected = []
    for variant in variants:
        if variant not in by_variant:
            raise KeyError(f"Unknown selected variant: {variant}")
        selected.append(by_variant[variant])
    return sorted(selected, key=lambda case: case["variant"])


def build_folds(
    selection: dict[str, Any],
    cases: list[dict[str, Any]],
    split_mode: str,
) -> list[dict[str, Any]]:
    if split_mode == "original":
        splits = selection["splits"]
        return [
            {
                "fold": "original",
                "train": cases_for_variants(cases, splits["train"]["elf_variants"]),
                "validation": cases_for_variants(
                    cases, splits["validation"]["elf_variants"]
                ),
                "test": cases_for_variants(cases, splits["test"]["elf_variants"]),
            }
        ]

    programs = sorted({str(case["program"]) for case in cases})
    if len(programs) < 3:
        raise ValueError("Rotating splits require at least three programs")

    folds = []
    for index, test_program in enumerate(programs):
        validation_program = programs[(index + 1) % len(programs)]
        train_programs = set(programs) - {test_program, validation_program}
        folds.append(
            {
                "fold": f"test_{test_program}__validation_{validation_program}",
                "train": [case for case in cases if case["program"] in train_programs],
                "validation": [
                    case for case in cases if case["program"] == validation_program
                ],
                "test": [case for case in cases if case["program"] == test_program],
            }
        )
    return folds


def validate_folds(
    folds: list[dict[str, Any]], archive_names: list[str]
) -> None:
    for fold in folds:
        for split in ("train", "validation", "test"):
            cases = fold[split]
            if not cases:
                raise ValueError(f"{fold['fold']} has an empty {split} split")
            labels = greedy.split_library_label_counts(cases, archive_names)
            if labels["positive"] == 0 or labels["negative"] == 0:
                raise ValueError(
                    f"{fold['fold']} {split} has a single label class: {labels}"
                )


def missing_collection_pairs(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> list[str]:
    missing = []
    collected_cases = collection.get("cases", {})
    for case in cases:
        binary_path = str(case["binary"])
        libraries = collected_cases.get(binary_path)
        if libraries is None:
            missing.append(f"{case['variant']}:<case>")
            continue
        for archive_name in archive_names:
            found, _ = greedy.find_archive_records(libraries, archive_name)
            if not found:
                missing.append(f"{case['variant']}:{archive_name}")
    return missing


def invalid_collection_records(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> list[str]:
    required_record_fields = {
        "rodata",
        "rodata_has_rodata",
        "rodata_strings",
        "rodata_ngrams",
        "rodata_bytes",
    }
    required_window_fields = {
        "coverage_mean",
        "coverage_ratio",
        "assignment_quality",
        "assignment_ratio",
        "call_edge_ratio",
        "function_concentration",
        "function_spread",
    }
    invalid = []
    collected_cases = collection.get("cases", {})
    for case in cases:
        libraries = collected_cases.get(str(case["binary"]), {})
        for archive_name in archive_names:
            found, records = greedy.find_archive_records(libraries, archive_name)
            if not found:
                continue
            for record_index, record in enumerate(records):
                missing = sorted(required_record_fields - set(record))
                if not record.get("windows"):
                    missing.extend(
                        sorted(required_window_fields - set(record))
                    )
                if missing:
                    invalid.append(
                        f"{case['variant']}:{archive_name}:record[{record_index}] "
                        f"missing {','.join(missing)}"
                    )
                for window_index, window in enumerate(record.get("windows", [])):
                    missing_window = sorted(required_window_fields - set(window))
                    if missing_window:
                        invalid.append(
                            f"{case['variant']}:{archive_name}:record[{record_index}]"
                            f".window[{window_index}] missing {','.join(missing_window)}"
                        )
    return invalid


def load_collections(
    greedy_dir: Path,
    selection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
    allow_missing_profiles: bool,
    strict_profiles: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    collections = {}
    excluded = []
    missing_files = []

    for expected_profile in selection["structural_profiles"]:
        profile_id = str(expected_profile["profile_id"])
        path = greedy_dir / "features" / f"{profile_id}.json.gz"
        if not path.is_file():
            missing_files.append(str(path))
            continue

        try:
            collection = greedy.read_gzip_json(path)
        except (OSError, json.JSONDecodeError) as error:
            reason = f"unreadable: {error}"
            if strict_profiles:
                raise ValueError(f"Invalid feature profile {profile_id}: {reason}") from error
            excluded.append({"profile_id": profile_id, "reason": reason})
            continue

        if collection.get("feature_schema_version") != greedy.FEATURE_SCHEMA_VERSION:
            reason = (
                f"schema={collection.get('feature_schema_version')}, "
                f"expected={greedy.FEATURE_SCHEMA_VERSION}"
            )
        elif collection.get("asm_normalization", "legacy") != selection.get(
            "asm_normalization", "legacy"
        ):
            reason = "assembly normalization metadata mismatch"
        elif collection.get("palmtree_pooling", "mean") != selection.get(
            "palmtree_pooling", "mean"
        ):
            reason = "PalmTree pooling metadata mismatch"
        elif not greedy.structural_profile_matches(expected_profile, collection):
            reason = "structural profile metadata mismatch"
        elif collection.get("input_signature") != greedy.selection_signature(
            [{"input": selection["input_signature"], **expected_profile}]
        ):
            reason = "feature input signature mismatch"
        else:
            missing_pairs = missing_collection_pairs(
                collection, cases, archive_names
            )
            invalid_records = invalid_collection_records(
                collection, cases, archive_names
            )
            if missing_pairs:
                reason = (
                    f"missing {len(missing_pairs)} pair(s): "
                    + ", ".join(missing_pairs[:5])
                    + (" ..." if len(missing_pairs) > 5 else "")
                )
            elif invalid_records:
                reason = (
                    f"invalid {len(invalid_records)} record(s): "
                    + "; ".join(invalid_records[:3])
                    + (" ..." if len(invalid_records) > 3 else "")
                )
            else:
                reason = ""

        if reason:
            if strict_profiles:
                raise ValueError(f"Invalid feature profile {profile_id}: {reason}")
            excluded.append({"profile_id": profile_id, "reason": reason})
            continue

        collections[profile_id] = collection

    if missing_files and not allow_missing_profiles:
        raise FileNotFoundError(
            "Feature collection is not finished; missing:\n  "
            + "\n  ".join(missing_files)
        )
    if not collections:
        raise ValueError("No complete compatible feature profiles are available")
    return collections, excluded


def balanced_accuracy(metrics: dict[str, float | int]) -> float:
    return (
        float(metrics["recall"]) + float(metrics["specificity"])
    ) / 2.0


def aggregate_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, float | int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    cu_counts = {"tp": 0, "fp": 0, "fn": 0}
    cu_evaluated_pairs = 0
    for prediction in predictions:
        counts[str(prediction["classification"]).lower()] += 1
        if prediction.get("cu_ground_truth"):
            cu_evaluated_pairs += 1
            cu_counts["tp"] += int(prediction["cu_tp"])
            cu_counts["fp"] += int(prediction["cu_fp"])
            cu_counts["fn"] += int(prediction["cu_fn"])

    metrics = greedy.calculate_metrics(**counts)
    metrics.update(
        {
            "lib_accuracy": metrics["accuracy"],
            "lib_precision": metrics["precision"],
            "lib_recall": metrics["recall"],
            "lib_specificity": metrics["specificity"],
            "lib_f1": metrics["f1"],
            "lib_balanced_accuracy": balanced_accuracy(metrics),
        }
    )
    metrics.update(
        greedy.calculate_cu_metrics(
            cu_counts["tp"],
            cu_counts["fp"],
            cu_counts["fn"],
            cu_evaluated_pairs,
        )
    )
    return metrics


def result_signature(
    selection: dict[str, Any],
    spec: AblationSpec,
    fold: dict[str, Any],
    profile_ids: list[str],
    objective_order: str,
    seed: int,
    restarts: int,
    max_rounds: int,
) -> str:
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "feature_input_signature": selection.get("input_signature"),
        "asm_normalization": selection.get("asm_normalization", "legacy"),
        "palmtree_pooling": selection.get("palmtree_pooling", "mean"),
        "ablation": spec.name,
        "modules": sorted(spec.modules),
        "fold": fold["fold"],
        "train": [case["variant"] for case in fold["train"]],
        "validation": [case["variant"] for case in fold["validation"]],
        "test": [case["variant"] for case in fold["test"]],
        "profiles": profile_ids,
        "objective_order": objective_order,
        "seed": seed,
        "restarts": restarts,
        "max_rounds": max_rounds,
    }
    return greedy.selection_signature(payload)


def run_fold(
    spec: AblationSpec,
    fold: dict[str, Any],
    collections: dict[str, dict[str, Any]],
    archive_names: list[str],
    objective_order: str,
    seed: int,
    restarts: int,
    max_rounds: int,
) -> dict[str, Any]:
    best_profile_id = ""
    best_params: dict[str, Any] = {}
    best_train_metrics: dict[str, float | int] = {}
    best_validation_metrics: dict[str, float | int] = {}
    profile_rows = []

    for profile_id, collection in sorted(collections.items()):
        params, train_metrics = tune_profile(
            spec,
            collection,
            fold["train"],
            archive_names,
            objective_order,
            restarts,
            max_rounds,
            stable_seed(seed, spec.name, fold["fold"], profile_id),
        )
        validation_metrics, _ = greedy.evaluate(
            params, collection, fold["validation"], archive_names
        )
        profile_rows.append(
            {
                "configuration": spec.name,
                "fold": fold["fold"],
                "profile_id": profile_id,
                **collection["profile"],
                **{f"best_{key}": value for key, value in params.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )

        if (
            not best_validation_metrics
            or greedy.objective(validation_metrics, objective_order)
            > greedy.objective(best_validation_metrics, objective_order)
        ):
            best_profile_id = profile_id
            best_params = params
            best_train_metrics = train_metrics
            best_validation_metrics = validation_metrics

    collection = collections[best_profile_id]
    test_metrics, predictions = greedy.evaluate(
        best_params, collection, fold["test"], archive_names
    )
    test_metrics["lib_balanced_accuracy"] = balanced_accuracy(test_metrics)
    for prediction in predictions:
        prediction["configuration"] = spec.name
        prediction["fold"] = fold["fold"]
        prediction["split"] = "test"

    structural = {
        key: value
        for key, value in collection["profile"].items()
        if key != "profile_id"
    }
    return {
        "configuration": spec.name,
        "modules": sorted(spec.modules),
        "description": spec.description,
        "fold": fold["fold"],
        "train_variants": [case["variant"] for case in fold["train"]],
        "validation_variants": [case["variant"] for case in fold["validation"]],
        "test_variants": [case["variant"] for case in fold["test"]],
        "best_profile_id": best_profile_id,
        "parameters": {**structural, **best_params},
        "train_metrics": best_train_metrics,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "predictions": predictions,
        "profile_rows": profile_rows,
    }


def fold_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["test_metrics"]
    return {
        "configuration": result["configuration"],
        "modules": "+".join(result["modules"]),
        "fold": result["fold"],
        "best_profile_id": result["best_profile_id"],
        "train_variants": ";".join(result["train_variants"]),
        "validation_variants": ";".join(result["validation_variants"]),
        "test_variants": ";".join(result["test_variants"]),
        **metrics,
        "parameters_json": json.dumps(result["parameters"], sort_keys=True),
    }


def aggregate_summary_rows(
    specs: list[AblationSpec], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        selected = [
            result for result in results if result["configuration"] == spec.name
        ]
        predictions = [
            prediction
            for result in selected
            for prediction in result["predictions"]
        ]
        metrics = aggregate_predictions(predictions)
        fold_lib_f1 = [float(result["test_metrics"]["lib_f1"]) for result in selected]
        fold_cu_f1 = [float(result["test_metrics"]["cu_f1"]) for result in selected]
        rows.append(
            {
                "configuration": spec.name,
                "modules": "+".join(sorted(spec.modules)),
                "description": spec.description,
                "folds": len(selected),
                **metrics,
                "macro_fold_lib_f1": statistics.fmean(fold_lib_f1),
                "macro_fold_cu_f1": statistics.fmean(fold_cu_f1),
                "std_fold_lib_f1": (
                    statistics.pstdev(fold_lib_f1) if len(fold_lib_f1) > 1 else 0.0
                ),
                "std_fold_cu_f1": (
                    statistics.pstdev(fold_cu_f1) if len(fold_cu_f1) > 1 else 0.0
                ),
            }
        )

    full = next((row for row in rows if row["configuration"] == "FULL"), None)
    for row in rows:
        row["delta_lib_f1_vs_full"] = (
            float(row["lib_f1"]) - float(full["lib_f1"]) if full else ""
        )
        row["delta_cu_f1_vs_full"] = (
            float(row["cu_f1"]) - float(full["cu_f1"]) if full else ""
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.restarts < 1:
        raise ValueError("--restarts must be at least 1")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")

    greedy_dir = args.greedy_dir.resolve()
    selection_path = greedy_dir / "input_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing greedy selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else greedy_dir / "ablation_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else selection["seed"])
    objective_order = args.objective_order or str(
        selection.get("objective_order", "lib_first")
    )
    specs = selected_specs(args.configuration)
    cases = load_selected_cases(selection, args.ground_truth_dir)
    archive_names = [str(archive["name"]) for archive in selection["archives"]]
    folds = build_folds(selection, cases, args.split_mode)
    validate_folds(folds, archive_names)
    collections, excluded_profiles = load_collections(
        greedy_dir,
        selection,
        cases,
        archive_names,
        args.allow_missing_profiles,
        args.strict_profiles,
    )
    profile_ids = sorted(collections)

    print(
        f"Loaded {len(profile_ids)} complete profiles, {len(cases)} ELF files, "
        f"{len(archive_names)} libraries and {len(folds)} fold(s)."
    )
    for excluded in excluded_profiles:
        print(
            f"[exclude] {excluded['profile_id']}: {excluded['reason']}",
            file=sys.stderr,
        )
    for fold in folds:
        print(
            f"[{fold['fold']}] train={len(fold['train'])} "
            f"validation={len(fold['validation'])} test={len(fold['test'])}"
        )
    print("Configurations: " + ", ".join(spec.name for spec in specs))
    if args.dry_run:
        return 0

    results = []
    checkpoint_dir = output_dir / "checkpoints"
    for spec in specs:
        for fold in folds:
            signature = result_signature(
                selection,
                spec,
                fold,
                profile_ids,
                objective_order,
                seed,
                args.restarts,
                args.max_rounds,
            )
            checkpoint = checkpoint_dir / f"{spec.name}__{fold['fold']}.json"
            if args.resume and checkpoint.is_file():
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                if cached.get("signature") == signature:
                    print(f"[reuse] {spec.name} / {fold['fold']}")
                    results.append(cached["result"])
                    continue

            print(f"[run] {spec.name} / {fold['fold']}", flush=True)
            result = run_fold(
                spec,
                fold,
                collections,
                archive_names,
                objective_order,
                seed,
                args.restarts,
                args.max_rounds,
            )
            write_json(
                checkpoint,
                {"signature": signature, "result": result},
            )
            results.append(result)
            metrics = result["test_metrics"]
            print(
                f"  best={result['best_profile_id']} "
                f"lib_F1={float(metrics['lib_f1']):.4f} "
                f"CU_F1={float(metrics['cu_f1']):.4f}",
                flush=True,
            )

    summary_rows = aggregate_summary_rows(specs, results)
    fold_rows = [fold_summary_row(result) for result in results]
    predictions = [
        prediction
        for result in results
        for prediction in result["predictions"]
    ]
    profile_rows = [
        row
        for result in results
        for row in result["profile_rows"]
    ]
    metadata = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "greedy_dir": str(greedy_dir),
        "feature_input_signature": selection.get("input_signature"),
        "asm_normalization": selection.get("asm_normalization", "legacy"),
        "palmtree_pooling": selection.get("palmtree_pooling", "mean"),
        "split_mode": args.split_mode,
        "archive_panel_rule": "fixed panel from the original greedy selection",
        "seed": seed,
        "objective_order": objective_order,
        "restarts": args.restarts,
        "max_rounds": args.max_rounds,
        "profiles_used": profile_ids,
        "profiles_excluded": excluded_profiles,
        "configurations": [
            {
                "name": spec.name,
                "modules": sorted(spec.modules),
                "description": spec.description,
            }
            for spec in specs
        ],
        "summary": summary_rows,
        "fold_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"predictions", "profile_rows"}
            }
            for result in results
        ],
    }

    write_json(output_dir / "ablation_results.json", metadata)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "ablation_folds.csv", fold_rows)
    write_csv(output_dir / "ablation_predictions.csv", predictions)
    write_csv(output_dir / "ablation_profiles.csv", profile_rows)

    print("\nAggregate out-of-sample results:")
    for row in summary_rows:
        print(
            f"  {row['configuration']:7} "
            f"lib_F1={float(row['lib_f1']):.4f} "
            f"CU_F1={float(row['cu_f1']):.4f} "
            f"FP={int(row['fp'])} FN={int(row['fn'])}"
        )
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
