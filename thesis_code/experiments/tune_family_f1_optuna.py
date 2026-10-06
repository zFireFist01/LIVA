#!/usr/bin/env python3
"""Exploratory 25x4 Optuna study for logical-library-family F1.

This study selects a small panel from ``libseeker-unified`` and makes its own
development/test split.  It is not the Section 5.3 thesis calibration, which
uses a separate 1200-ELF development dataset and 3000 final trials.  For that
protocol use ``thesis_code/optuna_threshold_search.py --thesis-protocol``.

The input is the schema-3 ``replay_complete`` feature log corpus under
``libseeker-unified``.  A family is the logical archive name (for example
``libssl`` and ``libcrypto`` are distinct), never the source project.

The workflow is intentionally split into commands so feature parsing is done
once and the compact cache can be copied to every Optuna worker node:

    select   choose a balanced, reproducible ELF panel
    prepare  compact the selected replay logs
    init     create and seed the shared Optuna study
    serve    expose a local journal study through Optuna gRPC
    worker   contribute trials to the shared study
    status   print study progress without evaluating the test split
    finalize evaluate the best development trial once on the held-out test
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import random
import re
import statistics
import sys
from typing import Any, Iterable
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
CLANG14_TRUTH = REPO_ROOT / "libseeker" / "binaries"
SHARED_TRUTH = UNIFIED / "ground_truth" / "libseeker" / "binaries"

sys.path.insert(0, str(SCRIPT_DIR))
import replay_compiler_f1 as replay  # noqa: E402

engine = replay.replay


COMPILERS = (
    "gcc-11-11.5.0",
    "gcc-13-13.3.0",
    "clang-18-18.1.3",
    "clang-14-14.0.6",
)
OPTIMIZATIONS = ("O0", "O2", "O3", "Os")
STATUS_RE = re.compile(r"^(?:YES \[W\]|YES|NO)\s+\|\s+library=([^ |]+)")
HASHED_ARCHIVE_RE = re.compile(r"^(?P<archive>.+\.a)\.[0-9a-f]{16}$", re.I)

TUNED_PARAMETERS = (
    "library_score_aggregator",
    "library_min_score",
    "block_coverage_mean_threshold",
    "block_min_coverage_ratio",
    "block_assignment_quality_threshold",
    "block_min_assignment_ratio",
    "cu_min_function_coverage",
    "rodata_penalty_threshold",
)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def coordinate(report: Path) -> tuple[str, str]:
    with report.open(encoding="utf-8", errors="replace") as stream:
        stream.readline()
        target = stream.readline().removeprefix("Target binary: ").strip()
    parts = Path(target).parts
    if len(parts) < 4:
        raise ValueError(f"Cannot parse target coordinate from {report}")
    return parts[-4], parts[-2]


def report_index() -> dict[str, dict[tuple[str, str], Path]]:
    result: dict[str, dict[tuple[str, str], Path]] = {}
    for compiler in COMPILERS:
        reports = UNIFIED / compiler / "reports" / "current"
        current: dict[tuple[str, str], Path] = {}
        for report in sorted(reports.glob("*.report.txt")):
            key = coordinate(report)
            if key in current:
                raise ValueError(f"Duplicate report coordinate {compiler} {key}")
            current[key] = report
        result[compiler] = current
    return result


def report_labels(report: Path) -> set[str]:
    labels: set[str] = set()
    with report.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = STATUS_RE.match(line)
            if match:
                labels.add(match.group(1))
    return labels


def family_from_label(label: str) -> str:
    name = Path(label).name.lower()
    match = HASHED_ARCHIVE_RE.match(name)
    archive = match.group("archive") if match else name
    return archive[:-2] if archive.endswith(".a") else archive


def candidate_metadata(index: dict[str, dict[tuple[str, str], Path]]) -> tuple[set[str], dict[str, str]]:
    label_sets = []
    for compiler in COMPILERS:
        try:
            sample = next(iter(index[compiler].values()))
        except StopIteration as error:
            raise ValueError(f"No reports found for {compiler}") from error
        label_sets.append(report_labels(sample))
    first = label_sets[0]
    for compiler, labels in zip(COMPILERS[1:], label_sets[1:]):
        if labels != first:
            raise ValueError(
                f"Candidate label set differs for {compiler}: "
                f"{len(labels)} != {len(first)}"
            )
    mapping = {label: family_from_label(label) for label in sorted(first)}
    return first, mapping


def truth_root(compiler: str) -> Path:
    return CLANG14_TRUTH if compiler == "clang-14-14.0.6" else SHARED_TRUTH


def truth_path(program: str, compiler: str, optimization: str) -> Path:
    return truth_root(compiler) / program / compiler / optimization / "ground_truth.json"


def normalize_archive_family(name: str, candidate_families: set[str]) -> str | None:
    base = Path(name).name.lower()
    match = HASHED_ARCHIVE_RE.match(base)
    if match:
        base = match.group("archive")
    stem = base[:-2] if base.endswith(".a") else base
    if stem in candidate_families:
        return stem
    # Some linker maps expose glibc's internal versioned filename (for
    # example libm-2.39.a), while the searchable archive is libm.a.
    versionless = re.sub(r"-\d+(?:[._+-].*)?$", "", stem)
    if versionless in candidate_families:
        return versionless
    return None


def expected_families(path: Path, candidate_families: set[str]) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for archive in payload.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) <= 0:
            continue
        family = normalize_archive_family(
            str(archive.get("archive", "")), candidate_families
        )
        if family is not None:
            result.add(family)
    return result


def sample_with_optimization_quotas(
    coordinates: list[tuple[str, str]],
    quotas: dict[str, int],
    rng: random.Random,
) -> list[tuple[str, str]] | None:
    by_optimization = {
        optimization: [c for c in coordinates if c[1] == optimization]
        for optimization in OPTIMIZATIONS
    }
    selected: list[tuple[str, str]] = []
    used_programs: set[str] = set()
    slots = [
        optimization
        for optimization, count in quotas.items()
        for _ in range(count)
    ]
    rng.shuffle(slots)
    for optimization in slots:
        choices = [
            c
            for c in by_optimization[optimization]
            if c[0] not in used_programs
        ]
        if not choices:
            return None
        choice = rng.choice(choices)
        selected.append(choice)
        used_programs.add(choice[0])
    return selected


def distribution_score(
    selected: Iterable[tuple[str, str]],
    reference: Iterable[tuple[str, str]],
    positives: dict[tuple[str, str, str], set[str]],
) -> float:
    selected = list(selected)
    reference = list(reference)
    selected_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    selected_compiler_totals: Counter[str] = Counter()
    reference_compiler_totals: Counter[str] = Counter()
    for compiler in COMPILERS:
        for program, optimization in selected:
            families = positives[(compiler, program, optimization)]
            selected_counts.update(families)
            selected_compiler_totals[compiler] += len(families)
        for program, optimization in reference:
            families = positives[(compiler, program, optimization)]
            reference_counts.update(families)
            reference_compiler_totals[compiler] += len(families)
    ratio = len(selected) / max(1, len(reference))
    score = 0.0
    for family, count in reference_counts.items():
        expected = count * ratio
        difference = selected_counts[family] - expected
        score += (difference * difference) / (expected + 1.0)
        if count >= 8 and selected_counts[family] == 0:
            score += 8.0
    for compiler in COMPILERS:
        expected = reference_compiler_totals[compiler] * ratio
        difference = selected_compiler_totals[compiler] - expected
        score += 0.25 * (difference * difference) / (expected + 1.0)
    return score


def assign_folds(
    development: list[tuple[str, str]],
    positives: dict[tuple[str, str, str], set[str]],
    rng: random.Random,
    attempts: int = 4000,
) -> dict[tuple[str, str], int]:
    by_optimization = {
        optimization: [c for c in development if c[1] == optimization]
        for optimization in OPTIMIZATIONS
    }
    if any(len(values) != 5 for values in by_optimization.values()):
        raise ValueError("Development selection must contain five cases per optimization")
    best: tuple[float, dict[tuple[str, str], int]] | None = None
    for _ in range(attempts):
        folds: list[list[tuple[str, str]]] = [[] for _ in range(5)]
        for optimization in OPTIMIZATIONS:
            values = list(by_optimization[optimization])
            rng.shuffle(values)
            for fold, value in enumerate(values):
                folds[fold].append(value)
        score = sum(
            distribution_score(fold, development, positives) for fold in folds
        )
        mapping = {
            coordinate_value: fold_index
            for fold_index, fold in enumerate(folds)
            for coordinate_value in fold
        }
        if best is None or score < best[0]:
            best = score, mapping
    assert best is not None
    return best[1]


def command_select(args: argparse.Namespace) -> None:
    if args.elfs_per_compiler != 25 or args.test_per_compiler != 5:
        raise ValueError("This run is frozen at 25 ELF/compiler with 5 test ELF/compiler")
    index = report_index()
    labels, family_by_label = candidate_metadata(index)
    candidate_families = set(family_by_label.values())
    common = sorted(set.intersection(*(set(index[c]) for c in COMPILERS)))
    common = [coordinate_value for coordinate_value in common if coordinate_value[1] in OPTIMIZATIONS]
    positives: dict[tuple[str, str, str], set[str]] = {}
    for program, optimization in common:
        for compiler in COMPILERS:
            path = truth_path(program, compiler, optimization)
            if not path.is_file():
                raise FileNotFoundError(path)
            positives[(compiler, program, optimization)] = expected_families(
                path, candidate_families
            )

    rng = random.Random(args.seed)
    extra_optimization = rng.choice(OPTIMIZATIONS)
    total_quotas = {optimization: 6 for optimization in OPTIMIZATIONS}
    total_quotas[extra_optimization] = 7
    best_selection: tuple[float, list[tuple[str, str]]] | None = None
    for _ in range(args.balance_attempts):
        candidate = sample_with_optimization_quotas(common, total_quotas, rng)
        if candidate is None:
            continue
        score = distribution_score(candidate, common, positives)
        if best_selection is None or score < best_selection[0]:
            best_selection = score, candidate
    if best_selection is None:
        raise RuntimeError("Could not create a balanced 25-coordinate selection")
    selected = sorted(best_selection[1])

    test_quotas = {optimization: 1 for optimization in OPTIMIZATIONS}
    test_quotas[extra_optimization] = 2
    best_test: tuple[float, list[tuple[str, str]]] | None = None
    for _ in range(args.balance_attempts):
        candidate = sample_with_optimization_quotas(selected, test_quotas, rng)
        if candidate is None:
            continue
        score = distribution_score(candidate, selected, positives)
        if best_test is None or score < best_test[0]:
            best_test = score, candidate
    if best_test is None:
        raise RuntimeError("Could not create a balanced five-coordinate test split")
    test = set(best_test[1])
    development = [coordinate_value for coordinate_value in selected if coordinate_value not in test]
    fold_by_coordinate = assign_folds(development, positives, rng)

    cases = []
    for program, optimization in selected:
        for compiler in COMPILERS:
            report = index[compiler][(program, optimization)]
            feature = report.with_name(
                report.name.replace(".report.txt", ".features.jsonl.gz")
            )
            if not feature.is_file():
                raise FileNotFoundError(feature)
            truth = truth_path(program, compiler, optimization)
            split = "test" if (program, optimization) in test else "development"
            cases.append(
                {
                    "case_id": f"{program}__{compiler}__{optimization}",
                    "compiler": compiler,
                    "program": program,
                    "optimization": optimization,
                    "split": split,
                    "fold": (
                        None
                        if split == "test"
                        else fold_by_coordinate[(program, optimization)]
                    ),
                    "report": str(report.relative_to(REPO_ROOT)),
                    "feature": str(feature.relative_to(REPO_ROOT)),
                    "truth": str(truth.relative_to(REPO_ROOT)),
                    "expected_families": sorted(
                        positives[(compiler, program, optimization)]
                    ),
                }
            )
    selection_core = {
        "schema_version": 1,
        "seed": args.seed,
        "compilers": list(COMPILERS),
        "optimizations": list(OPTIMIZATIONS),
        "elfs_per_compiler": args.elfs_per_compiler,
        "development_elfs_per_compiler": args.elfs_per_compiler - args.test_per_compiler,
        "test_elfs_per_compiler": args.test_per_compiler,
        "candidate_builds": len(labels),
        "candidate_families": sorted(candidate_families),
        "family_definition": "logical archive basename without .a/build hash; source project is ignored",
        "selection_method": {
            "kind": "seeded_random_multilabel_balanced",
            "balance_attempts": args.balance_attempts,
            "unique_programs": True,
            "same_coordinates_for_every_compiler": True,
            "optimization_quotas": total_quotas,
            "test_optimization_quotas": test_quotas,
        },
        "cases": cases,
    }
    selection_core["selection_signature"] = canonical_json_hash(selection_core)
    json_dump(args.output, selection_core)
    print(json.dumps({
        "output": str(args.output),
        "selection_signature": selection_core["selection_signature"],
        "candidate_builds": len(labels),
        "candidate_families": len(candidate_families),
        "common_coordinates": len(common),
        "selected_coordinates": len(selected),
        "development_cases": sum(c["split"] == "development" for c in cases),
        "test_cases": sum(c["split"] == "test" for c in cases),
        "optimization_counts": dict(Counter(c[1] for c in selected)),
    }, indent=2, sort_keys=True))


def iter_json_objects(stream: Iterable[str]) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for line in stream:
        cursor = 0
        while cursor < len(line):
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            if cursor >= len(line):
                break
            payload, cursor = decoder.raw_decode(line, cursor)
            yield payload


def compact_function_matches(raw: list[Any]) -> list[int]:
    flattened: list[int] = []
    for match in raw:
        if isinstance(match, dict):
            flattened.extend(
                [
                    int(match["target_function_index"]),
                    int(match["source_function_index"]),
                ]
            )
        elif isinstance(match, (list, tuple)) and len(match) >= 2:
            flattened.extend([int(match[0]), int(match[1])])
    return flattened


def compact_window(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage_mean": float(window.get("coverage_mean", 0.0)),
        "coverage_ratio": float(window.get("coverage_ratio", 0.0)),
        "assignment_quality": float(window.get("assignment_quality", 0.0)),
        "assignment_ratio": float(window.get("assignment_ratio", 0.0)),
        "call_edge_ratio": float(window.get("call_edge_ratio", 0.0)),
        "function_concentration": float(window.get("function_concentration", 0.0)),
        "function_spread": float(window.get("function_spread", 0.0)),
        "function_coverage": float(window.get("function_coverage", 0.0)),
        "function_matches": compact_function_matches(
            list(window.get("function_matches", []))
        ),
    }


def prepare_one(task: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    case_row, signature, compact_dir_string = task
    compact_dir = Path(compact_dir_string)
    output = compact_dir / f"{case_row['case_id']}.pkl.gz"
    if output.is_file():
        try:
            with gzip.open(output, "rb") as stream:
                existing = pickle.load(stream)
            if existing.get("selection_signature") == signature:
                return {
                    "case_id": case_row["case_id"],
                    "output": str(output),
                    "reused": True,
                    "labels_with_windows": len(existing["libraries"]),
                    "records_with_windows": sum(
                        len(records) for records in existing["libraries"].values()
                    ),
                }
        except Exception:
            pass

    feature = REPO_ROOT / case_row["feature"]
    libraries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_call_targets: dict[int, list[int]] = {}
    header: dict[str, Any] | None = None
    with gzip.open(feature, "rt", encoding="utf-8", errors="replace") as stream:
        for payload in iter_json_objects(stream):
            kind = payload.get("type")
            if kind == "source_call_targets":
                header = payload
                source_call_targets = {
                    int(source): [int(target) for target in targets]
                    for source, targets in payload.get("calls", [])
                }
                continue
            if kind != "block_cu" or not payload.get("windows"):
                continue
            label = str(payload["library"])
            record = {
                "name": str(payload.get("name", "")),
                "target_cu_index": int(payload["target_cu_index"]),
                "target_function_count": int(payload.get("target_function_count", 0)),
                "windows": [compact_window(window) for window in payload["windows"]],
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
            libraries[label].append(record)
    if header is None or header.get("feature_mode") != "replay_complete":
        raise ValueError(f"Not a replay_complete feature log: {feature}")
    family_by_label = {
        label: family_from_label(label) for label in libraries
    }
    result = {
        "schema_version": 1,
        "selection_signature": signature,
        "case_id": case_row["case_id"],
        "compiler": case_row["compiler"],
        "program": case_row["program"],
        "optimization": case_row["optimization"],
        "split": case_row["split"],
        "fold": case_row["fold"],
        "expected_families": set(case_row["expected_families"]),
        "libraries": dict(libraries),
        "family_by_label": family_by_label,
        "source_call_targets": source_call_targets,
        "matching_configuration": header.get("matching_configuration", {}),
    }
    compact_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb", compresslevel=3) as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "case_id": case_row["case_id"],
        "output": str(output),
        "reused": False,
        "labels_with_windows": len(libraries),
        "records_with_windows": sum(len(records) for records in libraries.values()),
    }


def command_prepare(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    signature = selection["selection_signature"]
    tasks = [
        (case, signature, str(args.compact_dir.resolve()))
        for case in selection["cases"]
    ]
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(prepare_one, tasks))
        for result in results:
            print(
                f"[{len(results):03d}] {result['case_id']} "
                f"labels={result['labels_with_windows']} "
                f"records={result['records_with_windows']} "
                f"{'reused' if result['reused'] else 'written'}"
            )
    manifest = {
        "schema_version": 1,
        "selection_signature": signature,
        "cases": sorted(results, key=lambda item: item["case_id"]),
    }
    json_dump(args.compact_dir / "compact_manifest.json", manifest)


def load_cases(selection_path: Path, compact_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    signature = selection["selection_signature"]
    cases = []
    for row in selection["cases"]:
        path = compact_dir / f"{row['case_id']}.pkl.gz"
        with gzip.open(path, "rb") as stream:
            case = pickle.load(stream)
        if case.get("selection_signature") != signature:
            raise ValueError(f"Compact cache signature mismatch: {path}")
        cases.append(case)
    return selection, cases


def suggested_params(trial: Any) -> dict[str, Any]:
    params = dict(engine.DECISION_DEFAULTS)
    params.update(
        {
            "library_score_aggregator": trial.suggest_categorical(
                "library_score_aggregator",
                ["mean", "max", "top3_mean", "top3_noisy_or"],
            ),
            "library_min_score": trial.suggest_float(
                "library_min_score", 0.90, 1.00, step=0.005
            ),
            "block_coverage_mean_threshold": trial.suggest_float(
                "block_coverage_mean_threshold", 0.975, 1.00, step=0.005
            ),
            "block_min_coverage_ratio": trial.suggest_float(
                "block_min_coverage_ratio", 0.65, 1.00, step=0.05
            ),
            "block_assignment_quality_threshold": trial.suggest_float(
                "block_assignment_quality_threshold", 0.875, 1.00, step=0.025
            ),
            "block_min_assignment_ratio": trial.suggest_float(
                "block_min_assignment_ratio", 0.40, 1.00, step=0.05
            ),
            "cu_min_function_coverage": trial.suggest_float(
                "cu_min_function_coverage", 0.0, 1.0, step=0.05
            ),
            "rodata_penalty_threshold": trial.suggest_float(
                "rodata_penalty_threshold", 0.0, 0.50, step=0.05
            ),
        }
    )
    return params


def default_trial_params() -> dict[str, Any]:
    defaults = engine.DECISION_DEFAULTS
    return {
        name: (
            "top3_noisy_or"
            if name == "library_score_aggregator"
            else defaults[name]
        )
        for name in TUNED_PARAMETERS
    }


def family_counts(cases: Iterable[dict[str, Any]], params: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        predicted: set[str] = set()
        for label, records in case["libraries"].items():
            accepted, score = engine.library_match_evidence(
                records, params, case["source_call_targets"]
            )
            if accepted and score >= float(params["library_min_score"]):
                predicted.add(case["family_by_label"][label])
        expected = set(case["expected_families"])
        counts["tp"] += len(predicted & expected)
        counts["fp"] += len(predicted - expected)
        counts["fn"] += len(expected - predicted)
        counts["evaluated_elfs"] += 1
    return counts


def metrics_from_counts(counts: Counter[str]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evaluated_elfs": counts["evaluated_elfs"],
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def storage_from_argument(value: str):
    import optuna

    if value.startswith("grpc://"):
        parsed = urlparse(value)
        return optuna.storages.GrpcStorageProxy(
            host=parsed.hostname or "localhost",
            port=parsed.port or 13000,
        )
    if value.startswith("journal://"):
        from optuna.storages.journal import JournalFileBackend

        path = value.removeprefix("journal://")
        backend = JournalFileBackend(path)
        return optuna.storages.JournalStorage(backend)
    return value


def create_or_load_study(args: argparse.Namespace, *, sampler_seed: int):
    import optuna

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=100,
        multivariate=True,
        constant_liar=True,
    )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage_from_argument(args.storage),
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )


def command_init(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    study = create_or_load_study(args, sampler_seed=args.seed)
    expected = {
        "selection_signature": selection["selection_signature"],
        "objective": "mean_5fold_pooled_family_f1",
        "family_definition": selection["family_definition"],
        "total_trial_budget": args.total_trials,
        "test_policy": "held_out_until_finalize",
    }
    for key, value in expected.items():
        previous = study.user_attrs.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"Incompatible study attribute {key}: {previous!r}")
        study.set_user_attr(key, value)
    if not study.trials:
        study.enqueue_trial(default_trial_params(), user_attrs={"kind": "baseline"})
    print(json.dumps({"study": study.study_name, **expected}, indent=2, sort_keys=True))


def command_serve(args: argparse.Namespace) -> None:
    import optuna
    from optuna.storages.journal import JournalFileBackend

    backend = JournalFileBackend(str(args.journal.resolve()))
    storage = optuna.storages.JournalStorage(backend)
    print(f"Serving Optuna journal {args.journal} on {args.host}:{args.port}", flush=True)
    optuna.storages.run_grpc_proxy_server(
        storage, host=args.host, port=args.port
    )


def command_worker(args: argparse.Namespace) -> None:
    import optuna
    from optuna.trial import TrialState

    selection, cases = load_cases(args.selection, args.compact_dir)
    development = [case for case in cases if case["split"] == "development"]
    folds = {
        fold: [case for case in development if case["fold"] == fold]
        for fold in range(5)
    }
    if any(not fold_cases for fold_cases in folds.values()):
        raise ValueError("Every development fold must contain cases")
    study = create_or_load_study(
        args,
        sampler_seed=args.seed + (args.worker_id * 1_000_003),
    )
    expected_signature = study.user_attrs.get("selection_signature")
    if expected_signature != selection["selection_signature"]:
        raise ValueError("Study and compact cache use different selections")

    def objective(trial: Any) -> float:
        params = suggested_params(trial)
        fold_metrics = []
        compiler_counts: dict[str, Counter[str]] = {
            compiler: Counter() for compiler in COMPILERS
        }
        for fold in range(5):
            current_counts: Counter[str] = Counter()
            for compiler in COMPILERS:
                subset = [
                    case for case in folds[fold] if case["compiler"] == compiler
                ]
                subset_counts = family_counts(subset, params)
                current_counts.update(subset_counts)
                compiler_counts[compiler].update(subset_counts)
            current_metrics = metrics_from_counts(current_counts)
            fold_metrics.append(current_metrics)
            trial.report(float(current_metrics["f1"]), step=fold)
        values = [float(item["f1"]) for item in fold_metrics]
        for compiler, counts in compiler_counts.items():
            trial.set_user_attr(
                f"development_f1_{compiler}",
                metrics_from_counts(counts)["f1"],
            )
        trial.set_user_attr("fold_f1", values)
        trial.set_user_attr("fold_f1_std", statistics.pstdev(values))
        return statistics.fmean(values)

    callback = optuna.study.MaxTrialsCallback(
        args.total_trials,
        states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
    )
    study.optimize(
        objective,
        n_trials=args.total_trials,
        callbacks=[callback],
        gc_after_trial=False,
        show_progress_bar=False,
    )


def study_status(study: Any) -> dict[str, Any]:
    from optuna.trial import TrialState

    counts = Counter(trial.state.name for trial in study.get_trials(deepcopy=False))
    result: dict[str, Any] = {
        "study": study.study_name,
        "states": dict(counts),
        "total_trials": sum(counts.values()),
        "target_trials": study.user_attrs.get("total_trial_budget"),
    }
    complete = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == TrialState.COMPLETE
    ]
    if complete:
        result.update(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
            }
        )
    return result


def command_status(args: argparse.Namespace) -> None:
    import optuna

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_from_argument(args.storage),
    )
    print(json.dumps(study_status(study), indent=2, sort_keys=True))


def command_finalize(args: argparse.Namespace) -> None:
    import optuna
    from optuna.trial import TrialState

    selection, cases = load_cases(args.selection, args.compact_dir)
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_from_argument(args.storage),
    )
    finished = sum(
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        for trial in study.get_trials(deepcopy=False)
    )
    target = int(study.user_attrs.get("total_trial_budget", args.total_trials))
    if finished < target and not args.allow_incomplete:
        raise RuntimeError(f"Study incomplete: {finished}/{target} finished trials")
    params = dict(engine.DECISION_DEFAULTS)
    params.update(study.best_params)
    baseline = dict(engine.DECISION_DEFAULTS)

    def evaluate_split(split: str, current: dict[str, Any]) -> dict[str, Any]:
        selected = [case for case in cases if case["split"] == split]
        overall = metrics_from_counts(family_counts(selected, current))
        by_compiler = {
            compiler: metrics_from_counts(
                family_counts(
                    [case for case in selected if case["compiler"] == compiler],
                    current,
                )
            )
            for compiler in COMPILERS
        }
        return {"overall": overall, "by_compiler": by_compiler}

    result = {
        "selection_signature": selection["selection_signature"],
        "study": study_status(study),
        "best_params": params,
        "baseline": {
            "development": evaluate_split("development", baseline),
            "test": evaluate_split("test", baseline),
        },
        "tuned": {
            "development": evaluate_split("development", params),
            "test": evaluate_split("test", params),
        },
    }
    json_dump(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--elfs-per-compiler", type=int, default=25)
    select.add_argument("--test-per-compiler", type=int, default=5)
    select.add_argument("--seed", type=int, default=20260916)
    select.add_argument("--balance-attempts", type=int, default=10000)
    select.set_defaults(func=command_select)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--compact-dir", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.set_defaults(func=command_prepare)

    for name, function in (("init", command_init), ("worker", command_worker)):
        current = subparsers.add_parser(name)
        current.add_argument("--selection", type=Path, required=True)
        current.add_argument("--compact-dir", type=Path)
        current.add_argument("--storage", required=True)
        current.add_argument("--study-name", default="family_f1_25x4")
        current.add_argument("--total-trials", type=int, default=2000)
        current.add_argument("--seed", type=int, default=20260916)
        if name == "worker":
            current.add_argument("--worker-id", type=int, required=True)
        current.set_defaults(func=function)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--journal", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=13000)
    serve.set_defaults(func=command_serve)

    status = subparsers.add_parser("status")
    status.add_argument("--storage", required=True)
    status.add_argument("--study-name", default="family_f1_25x4")
    status.set_defaults(func=command_status)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--selection", type=Path, required=True)
    finalize.add_argument("--compact-dir", type=Path, required=True)
    finalize.add_argument("--storage", required=True)
    finalize.add_argument("--study-name", default="family_f1_25x4")
    finalize.add_argument("--total-trials", type=int, default=2000)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"init", "worker"} and args.command == "worker" and args.compact_dir is None:
        raise ValueError("worker requires --compact-dir")
    args.func(args)


if __name__ == "__main__":
    main()
