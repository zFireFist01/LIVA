#!/usr/bin/env python3
"""Replay a fixed-profile B/H/S/X/R ablation from batch feature logs.

The expensive ELF/archive analysis is not repeated.  Input logs must have been
created with ``run_libseeker_batch.py --offline-ablation-features``.  The
default candidate panel follows the LibSeeker-style experiment used in this
repository: one current-version GCC 13/O2 archive per archive family.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = THESIS_DIR.parent
if str(THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(THESIS_DIR))

import optuna_threshold_search as replay
from run_libseeker_batch import ground_truth_archive_family, library_family


COMPONENTS = ("H", "S", "X", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Batch output containing reports/current. Repeat for multiple "
            "shards."
        ),
    )
    parser.add_argument(
        "--library-matrix",
        type=Path,
        default=REPO_ROOT / "Dataset/manifests/library_matrix.tsv",
    )
    parser.add_argument(
        "--candidate-panel",
        choices=("reference", "all-versions-top1"),
        default="reference",
        help=(
            "reference keeps the selected role/toolchain/optimization; "
            "all-versions-top1 compares all builds but emits one decision per "
            "archive family, retaining only its highest-scoring build."
        ),
    )
    parser.add_argument(
        "--reference-role",
        default="current",
        help="Library version role kept in the LibSeeker-style panel.",
    )
    parser.add_argument("--reference-toolchain", default="gcc-13-13.3.0")
    parser.add_argument("--reference-optimization", default="O2")
    parser.add_argument(
        "--cu-function-coverage",
        type=float,
        help=(
            "Override the function-coverage threshold used by configurations "
            "containing S. "
            "By default the value recorded in the feature log is used."
        ),
    )
    parser.add_argument(
        "--configuration",
        action="append",
        help=(
            "Run one configuration such as B, B_H, B_H_S_X or FULL. "
            "Repeatable; defaults to the complete 2^4 factorial study."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "ablation_results/current_pipeline",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze the replay-complete ELF logs currently present.",
    )
    return parser.parse_args()


def configurations(requested: list[str] | None) -> list[tuple[str, frozenset[str]]]:
    result: list[tuple[str, frozenset[str]]] = []
    for size in range(len(COMPONENTS) + 1):
        for enabled in itertools.combinations(COMPONENTS, size):
            modules = frozenset(enabled)
            name = "B" if not modules else "B_" + "_".join(enabled)
            if modules == frozenset(COMPONENTS):
                name = "FULL"
            result.append((name, modules))
    if not requested:
        return result
    aliases = {name: modules for name, modules in result}
    aliases["B_H_S_X_R"] = frozenset(COMPONENTS)
    unknown = sorted(set(requested) - set(aliases))
    if unknown:
        raise ValueError("Unknown configuration(s): " + ", ".join(unknown))
    return [(name, aliases[name]) for name in requested]


def matrix_catalog(
    path: Path,
    role: str | None,
    toolchain: str | None,
    optimization: str | None,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("status") == "selected" and row.get("path"):
                rows.append(dict(row))

    basename_counts: dict[str, int] = {}
    for row in rows:
        basename = str(row["archive"])
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

    catalog: dict[str, dict[str, str]] = {}
    for row in rows:
        basename = str(row["archive"])
        dataset_path = (
            Path("Dataset/builds/libraries") / str(row["path"])
        ).as_posix()
        label = basename
        if basename_counts[basename] > 1:
            suffix = hashlib.sha256(dataset_path.encode("utf-8")).hexdigest()[:16]
            label = f"{basename}.{suffix}"
        catalog[label] = {**row, "label": label, "dataset_path": dataset_path}

    selected = {
        label: row
        for label, row in catalog.items()
        if (role is None or row.get("role") == role)
        and (toolchain is None or row.get("toolchain") == toolchain)
        and (optimization is None or row.get("optimization") == optimization)
    }
    if not selected:
        raise ValueError("The requested reference panel is empty")
    return selected


def manifest_records(batch_output_dir: Path) -> dict[str, dict[str, Any]]:
    shard_root = batch_output_dir.resolve().parents[1]
    manifest = shard_root / "datasets/libseeker/manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Cannot infer dataset manifest for {batch_output_dir}: {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_root = manifest.parents[2]
    records: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        binary = Path(str(record.get("binary", "")))
        binary = binary if binary.is_absolute() else artifact_root / binary
        records[binary.resolve().as_posix()] = {
            **record,
            "_artifact_root": artifact_root.as_posix(),
        }
    return records


def feature_header(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                header = json.loads(line)
                break
        else:
            raise ValueError(f"Empty feature log: {path}")
    if (
        header.get("type") != "source_call_targets"
        or int(header.get("schema_version", 0)) < 3
        or header.get("feature_mode") != "replay_complete"
    ):
        raise ValueError(
            f"{path} is not replay-complete schema 3; rerun the ELF with "
            "--offline-ablation-features"
        )
    return header


def expected_families(case: dict[str, Any]) -> set[str]:
    artifact_root = Path(str(case["_artifact_root"]))
    ground_truth = Path(str(case.get("ground_truth", "")))
    ground_truth = (
        ground_truth if ground_truth.is_absolute() else artifact_root / ground_truth
    )
    payload = json.loads(ground_truth.read_text(encoding="utf-8"))
    return {
        ground_truth_archive_family(archive)
        for archive in payload.get("archives", [])
        if int(archive.get("included_compilation_units", 0) or 0) > 0
    }


def ablated_params(
    full: dict[str, Any],
    modules: frozenset[str],
    function_coverage: float,
) -> dict[str, Any]:
    params = {**replay.DECISION_DEFAULTS, **full}
    params["cu_min_function_coverage"] = function_coverage
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


def update_count(counts: dict[str, int], expected: bool, predicted: bool) -> str:
    label = (
        "tp" if expected and predicted
        else "fn" if expected
        else "fp" if predicted
        else "tn"
    )
    counts[label] += 1
    return label.upper()


def main() -> int:
    args = parse_args()
    if (
        args.cu_function_coverage is not None
        and not 0.0 <= args.cu_function_coverage <= 1.0
    ):
        raise ValueError("--cu-function-coverage must be in [0, 1]")
    specs = configurations(args.configuration)
    reference_panel = args.candidate_panel == "reference"
    candidates = matrix_catalog(
        args.library_matrix.resolve(),
        args.reference_role if reference_panel else None,
        args.reference_toolchain if reference_panel else None,
        args.reference_optimization if reference_panel else None,
    )
    candidate_groups: dict[str, list[str]] = {}
    for label in candidates:
        candidate_groups.setdefault(library_family(label), []).append(label)
    if reference_panel:
        duplicates = {
            family: labels
            for family, labels in candidate_groups.items()
            if len(labels) != 1
        }
        if duplicates:
            family, labels = next(iter(duplicates.items()))
            raise ValueError(
                f"Reference panel is not one archive per family: "
                f"{family} -> {', '.join(labels)}"
            )
    counts = {
        name: {label: 0 for label in ("tp", "tn", "fp", "fn")}
        for name, _modules in specs
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "ablation_predictions.csv.gz"
    prediction_fields = (
        "configuration", "program", "compiler", "optimization", "library",
        "library_family", "expected", "predicted", "classification",
        "score", "matched_cu", "feature_log",
        "candidates_compared",
    )
    seen_binaries: set[str] = set()
    full_configuration: dict[str, Any] | None = None
    effective_function_coverage: float | None = None
    processed = 0

    with gzip.open(prediction_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        for batch_dir_arg in args.batch_output_dir:
            batch_dir = batch_dir_arg.resolve()
            case_by_binary = manifest_records(batch_dir)
            feature_dir = batch_dir / "reports/current"
            paths = sorted(feature_dir.glob("*.features.jsonl.gz"))
            if not paths:
                raise FileNotFoundError(f"No feature logs in {feature_dir}")
            for path in paths:
                header = feature_header(path)
                binary_path = Path(str(header["binary_path"])).resolve().as_posix()
                if binary_path in seen_binaries:
                    raise ValueError(f"Duplicate ELF feature log: {binary_path}")
                case = case_by_binary.get(binary_path)
                if case is None:
                    if args.allow_incomplete:
                        print(f"[skip] ELF is absent from inferred manifest: {binary_path}")
                        continue
                    raise KeyError(f"ELF is absent from inferred manifest: {binary_path}")
                configuration = dict(header.get("matching_configuration", {}))
                if full_configuration is None:
                    full_configuration = configuration
                    effective_function_coverage = float(
                        args.cu_function_coverage
                        if args.cu_function_coverage is not None
                        else configuration.get("cu_min_function_coverage", 0.0)
                    )
                elif configuration != full_configuration:
                    raise ValueError(
                        f"Mixed matching configurations; first differs from {path}"
                    )
                (
                    parsed_binary,
                    libraries,
                    _pooling,
                    source_calls,
                ) = replay.parse_feature_jsonl(path, set(candidates))
                if Path(parsed_binary).resolve().as_posix() != binary_path:
                    raise ValueError(f"Header/parser binary mismatch in {path}")
                expected = expected_families(case)
                missing = sorted(set(candidates) - set(libraries))
                if missing:
                    if args.allow_incomplete:
                        print(f"[skip] incomplete {path.name}: {len(missing)} candidates missing")
                        continue
                    raise ValueError(
                        f"Incomplete candidate panel in {path}: {len(missing)} missing"
                    )
                seen_binaries.add(binary_path)
                processed += 1
                for name, modules in specs:
                    params = ablated_params(
                        configuration,
                        modules,
                        float(effective_function_coverage),
                    )
                    for family, labels in candidate_groups.items():
                        ranked = []
                        for label in labels:
                            accepted, score = replay.library_match_evidence(
                                libraries[label], params, source_calls
                            )
                            ranked.append((score, len(accepted), label, accepted))
                        score, _accepted_count, label, accepted = max(ranked)
                        predicted = bool(accepted) and score >= float(
                            params["library_min_score"]
                        )
                        is_expected = family in expected
                        classification = update_count(
                            counts[name], is_expected, predicted
                        )
                        writer.writerow({
                            "configuration": name,
                            "program": case.get("program", ""),
                            "compiler": case.get("compiler", ""),
                            "optimization": case.get("program_optimization", ""),
                            "library": label,
                            "library_family": family,
                            "expected": is_expected,
                            "predicted": predicted,
                            "classification": classification,
                            "score": score,
                            "matched_cu": len(accepted),
                            "feature_log": path.as_posix(),
                            "candidates_compared": len(labels),
                        })

    if processed == 0:
        raise ValueError("No replay-complete ELF was evaluated")
    summary = []
    full_metrics = None
    for name, modules in specs:
        metrics = replay.calculate_metrics(**counts[name])
        row = {
            "configuration": name,
            "modules": "+".join(
                component for component in ("B", *COMPONENTS)
                if component == "B" or component in modules
            ),
            **metrics,
        }
        summary.append(row)
        if name == "FULL":
            full_metrics = metrics
    if full_metrics is not None:
        for row in summary:
            row["delta_f1_vs_full"] = float(row["f1"]) - float(full_metrics["f1"])

    summary_path = output_dir / "ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "schema_version": 1,
        "protocol": "fixed-profile pure factorial ablation",
        "metric_protocol": "LibSeeker-style archive-family presence",
        "retuned": False,
        "elf_count": processed,
        "library_family_count": len(candidate_groups),
        "candidate_archive_count": len(candidates),
        "pair_count_per_configuration": processed * len(candidate_groups),
        "candidate_panel": args.candidate_panel,
        "reference_panel": {
            "role": args.reference_role,
            "toolchain": args.reference_toolchain,
            "optimization": args.reference_optimization,
        },
        "cu_function_coverage": effective_function_coverage,
        "full_matching_configuration": full_configuration,
        "summary": summary,
    }
    (output_dir / "ablation_results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Offline ablation: {processed} ELF x {len(candidate_groups)} archive families "
        f"x {len(specs)} configurations"
    )
    if not effective_function_coverage:
        print(
            "[warning] The function-coverage sub-gate of S is disabled. "
            "Rerun matching with --cu-min-function-coverage 0.60 or pass an "
            "explicit offline override."
        )
    for row in summary:
        print(
            f"  {row['configuration']:11} "
            f"P={float(row['precision']):.4f} "
            f"R={float(row['recall']):.4f} "
            f"F1={float(row['f1']):.4f} "
            f"FP={int(row['fp'])} FN={int(row['fn'])}"
        )
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
