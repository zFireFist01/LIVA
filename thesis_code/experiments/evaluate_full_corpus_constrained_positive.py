#!/usr/bin/env python3
"""Evaluate build transfer while keeping the negative corpus fully open.

This is the stricter, truth-conditioned protocol used as a stress test in the
thesis.  For a family that is present in an ELF, only build labels compatible
with the scenario may produce a TP.  For every absent family, however, *all*
available build labels remain candidates; accepting any of them produces one
family-level FP.

The matcher is replayed from existing ``replay_complete`` feature logs.  No
ELF analysis, disassembly, embeddings, or binary matching is rerun.
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
from evaluate_cross_build_f1_protocol import candidate_metadata  # noqa: E402

ENGINE = family.engine
COMPILERS = tuple(family.COMPILERS)
OPTIMIZATIONS = tuple(family.OPTIMIZATIONS)
VERSION_ROLES = ("current", "minor-alternative", "major-alternative")
SCENARIOS = (
    "same_exact_artifact",
    "same_build_configuration",
    "cross_optimization_any",
    "cross_compiler_any",
    "cross_version_any",
)
TRANSITION_VALUES = {
    "optimization": OPTIMIZATIONS,
    "compiler": COMPILERS,
    "version": VERSION_ROLES,
}

_META: dict[str, dict[str, str]] = {}
_FAMILIES: set[str] = set()
_LABELS_BY_FAMILY: dict[str, set[str]] = {}
_LABEL_BY_PATH: dict[str, str] = {}


def init_worker(meta: dict[str, dict[str, str]], families: set[str]) -> None:
    global _META, _FAMILIES, _LABELS_BY_FAMILY, _LABEL_BY_PATH
    _META = meta
    _FAMILIES = families
    grouped: dict[str, set[str]] = defaultdict(set)
    for label, value in meta.items():
        grouped[value["family"]].add(label)
    _LABELS_BY_FAMILY = dict(grouped)
    _LABEL_BY_PATH = {value["path"]: label for label, value in meta.items()}


def truth_anchors(truth: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Return the actually linked archive build(s), grouped by logical family."""
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    marker = "Dataset/builds/libraries/"
    for archive in truth.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) <= 0:
            continue
        archive_name = str(
            archive.get("archive_basename")
            or Path(str(archive.get("archive", ""))).name
        )
        family_name = family.normalize_archive_family(archive_name, _FAMILIES)
        if family_name is None:
            continue
        absolute = str(archive.get("archive", ""))
        relative = absolute.split(marker, 1)[1] if marker in absolute else absolute
        exact_label = _LABEL_BY_PATH.get(relative, "")
        meta = _META.get(exact_label, {})
        inferred = infer_archive_provenance(relative, family_name)
        result[family_name].append({
            "label": exact_label,
            "package": str(
                meta.get("package", "") or archive.get("library")
                or inferred.get("package", "")
            ),
            "compiler": str(
                meta.get("compiler", "") or archive.get("compiler")
                or inferred.get("compiler", "")
            ),
            "optimization": str(
                meta.get("optimization", "") or archive.get("optimization")
                or inferred.get("optimization", "")
            ),
            "role": str(
                meta.get("role", "") or archive.get("version_role")
                or inferred.get("role", "")
            ),
            "source_build": str(
                meta.get("source_build", "") or inferred.get("source_build", "")
            ),
            "provenance_source": (
                "exact_matrix_path" if exact_label
                else inferred.get("provenance_source", "unresolved")
            ),
            "path": relative,
        })
    return dict(result)


def canonical_source_build(value: str) -> str:
    """Normalize source directory spelling such as util-linux-v2.39.3."""
    value = Path(value).name.lower()
    return value.replace("-v", "-", 1) if "-v" in value else value


def infer_archive_provenance(path: str, family_name: str) -> dict[str, str]:
    """Recover metadata omitted for archives originating in program build trees.

    Ground truth sometimes names an intermediate ``build/.libs/*.a`` archive,
    whereas the candidate matrix names an independently installed archive from
    the same source version/configuration.  Path identity must remain false, but
    package, version role, compiler, and optimization can be reconstructed.
    """
    normalized = path.replace("\\", "/")
    source_build = ""
    compiler = ""
    optimization = ""
    provenance_source = "unresolved"

    library_marker = "builds/libraries/"
    program_marker = "builds/programs/"
    if library_marker in normalized:
        parts = normalized.split(library_marker, 1)[1].split("/")
        if len(parts) >= 3:
            compiler, source_build, optimization = parts[:3]
            provenance_source = "library_path"
    elif program_marker in normalized:
        parts = normalized.split(program_marker, 1)[1].split("/")
        if len(parts) >= 2:
            source_build, build_directory = parts[:2]
            for compiler_value in COMPILERS:
                for optimization_value in OPTIMIZATIONS:
                    if f"_{compiler_value}_{optimization_value}" in build_directory:
                        compiler = compiler_value
                        optimization = optimization_value
                        provenance_source = "program_build_path"
                        break
                if compiler:
                    break

    matching_meta = [
        meta for label, meta in _META.items()
        if (
            meta["family"] == family_name
            and canonical_source_build(meta["source_build"])
            == canonical_source_build(source_build)
        )
    ]
    packages = {meta["package"] for meta in matching_meta}
    roles = {meta["role"] for meta in matching_meta}
    return {
        "package": next(iter(packages)) if len(packages) == 1 else "",
        "role": next(iter(roles)) if len(roles) == 1 else "",
        "compiler": compiler,
        "optimization": optimization,
        "source_build": source_build,
        "provenance_source": provenance_source,
    }


def compatible_labels(
    scenario: str, family_name: str, anchors: list[dict[str, str]]
) -> set[str]:
    candidates = _LABELS_BY_FAMILY.get(family_name, set())
    selected: set[str] = set()
    for anchor in anchors:
        if scenario == "same_exact_artifact":
            if anchor["label"]:
                selected.add(anchor["label"])
            continue
        for label in candidates:
            meta = _META[label]
            same_package = meta["package"] == anchor["package"]
            same_compiler = meta["compiler"] == anchor["compiler"]
            same_optimization = meta["optimization"] == anchor["optimization"]
            same_role = meta["role"] == anchor["role"]
            if scenario == "same_build_configuration":
                valid = (
                    same_package and same_compiler and same_optimization and same_role
                )
            elif scenario == "cross_optimization_any":
                valid = (
                    same_package and same_compiler and same_role
                    and not same_optimization
                )
            elif scenario == "cross_compiler_any":
                valid = (
                    same_package and same_optimization and same_role
                    and not same_compiler
                )
            elif scenario == "cross_version_any":
                valid = (
                    same_package and same_compiler and same_optimization
                    and not same_role
                )
            else:  # guarded by the fixed SCENARIOS tuple
                raise ValueError(f"Unknown scenario: {scenario}")
            if valid:
                selected.add(label)
    return selected


def transition_labels(
    dimension: str,
    target: str,
    family_name: str,
    anchors: list[dict[str, str]],
) -> set[str]:
    """Return candidates for one directed source-to-target transition."""
    selected: set[str] = set()
    for anchor in anchors:
        for label in _LABELS_BY_FAMILY.get(family_name, set()):
            meta = _META[label]
            if meta["package"] != anchor["package"]:
                continue
            if dimension == "optimization":
                valid = (
                    meta["compiler"] == anchor["compiler"]
                    and meta["role"] == anchor["role"]
                    and meta["optimization"] == target
                )
            elif dimension == "compiler":
                valid = (
                    meta["optimization"] == anchor["optimization"]
                    and meta["role"] == anchor["role"]
                    and meta["compiler"] == target
                )
            elif dimension == "version":
                valid = (
                    meta["compiler"] == anchor["compiler"]
                    and meta["optimization"] == anchor["optimization"]
                    and meta["role"] == target
                )
            else:
                raise ValueError(f"Unknown transition dimension: {dimension}")
            if valid:
                selected.add(label)
    return selected


def anchor_dimension(anchor: dict[str, str], dimension: str) -> str:
    return anchor["role" if dimension == "version" else dimension]


def evaluate_case(task: dict[str, str]) -> dict[str, Any]:
    truth = json.loads(Path(task["truth"]).read_text(encoding="utf-8"))
    header, libraries, calls = parse_relevant_features(Path(task["feature"]))
    params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **PARAMS,
    }
    accepted: set[str] = set()
    for label, records in libraries.items():
        if label not in _META:
            continue
        successful, score = ENGINE.library_match_evidence(records, params, calls)
        if successful and score >= float(params["library_min_score"]):
            accepted.add(label)

    all_anchors_by_family = truth_anchors(truth)
    anchor_provenance = Counter(
        anchor["provenance_source"]
        for anchors in all_anchors_by_family.values() for anchor in anchors
    )
    unresolved_anchor_families = sorted({
        family_name
        for family_name, anchors in all_anchors_by_family.items()
        if any(
            not all(anchor.get(key) for key in (
                "package", "compiler", "optimization", "role"
            ))
            for anchor in anchors
        )
    })
    anchors_by_family = {
        family_name: [
            anchor for anchor in anchors
            if all(anchor.get(key) for key in (
                "package", "compiler", "optimization", "role"
            ))
        ]
        for family_name, anchors in all_anchors_by_family.items()
    }
    excluded_out_of_scope_families = sorted(
        family_name for family_name, anchors in anchors_by_family.items()
        if not anchors
    )
    anchors_by_family = {
        family_name: anchors for family_name, anchors in anchors_by_family.items()
        if anchors
    }
    expected = set(anchors_by_family)
    excluded = set(excluded_out_of_scope_families)
    accepted_by_family: dict[str, set[str]] = defaultdict(set)
    for label in accepted:
        accepted_by_family[_META[label]["family"]].add(label)

    false_positive_families = sorted(
        set(accepted_by_family) - expected - excluded
    )
    true_negative_families = sorted(
        _FAMILIES - expected - excluded - set(accepted_by_family)
    )
    scenario_rows: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        tp_families: list[str] = []
        fn_families: list[str] = []
        unavailable_families: list[str] = []
        off_scenario_only_families: list[str] = []
        undetected_families: list[str] = []
        eligible_builds = 0
        for family_name, anchors in anchors_by_family.items():
            compatible = compatible_labels(scenario, family_name, anchors)
            eligible_builds += len(compatible)
            accepted_for_family = accepted_by_family.get(family_name, set())
            if compatible & accepted_for_family:
                tp_families.append(family_name)
            else:
                fn_families.append(family_name)
                if not compatible:
                    unavailable_families.append(family_name)
                elif accepted_for_family:
                    off_scenario_only_families.append(family_name)
                else:
                    undetected_families.append(family_name)
        scenario_rows[scenario] = {
            "TP": len(tp_families),
            "TN": len(true_negative_families),
            "FP": len(false_positive_families),
            "FN": len(fn_families),
            "eligible_positives": len(expected) - len(unavailable_families),
            "eligible_positive_builds": eligible_builds,
            "unavailable_positives": len(unavailable_families),
            "off_scenario_only": len(off_scenario_only_families),
            "undetected_on_any_build": len(undetected_families),
            "true_positive_families": sorted(tp_families),
            "false_negative_families": sorted(fn_families),
            "unavailable_families": sorted(unavailable_families),
            "off_scenario_only_families": sorted(off_scenario_only_families),
            "undetected_families": sorted(undetected_families),
        }

    transition_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, values in TRANSITION_VALUES.items():
        dimension_rows: dict[str, dict[str, Any]] = {}
        for source in values:
            source_anchors = {
                family_name: [
                    anchor for anchor in anchors
                    if anchor_dimension(anchor, dimension) == source
                ]
                for family_name, anchors in anchors_by_family.items()
            }
            source_anchors = {
                family_name: anchors
                for family_name, anchors in source_anchors.items() if anchors
            }
            if not source_anchors:
                continue
            for target in values:
                if target == source:
                    continue
                tp_families: list[str] = []
                fn_families: list[str] = []
                unavailable_families: list[str] = []
                off_target_only_families: list[str] = []
                undetected_families: list[str] = []
                candidate_builds = 0
                for family_name, anchors in source_anchors.items():
                    candidates = transition_labels(
                        dimension, target, family_name, anchors
                    )
                    candidate_builds += len(candidates)
                    accepted_for_family = accepted_by_family.get(family_name, set())
                    if candidates & accepted_for_family:
                        tp_families.append(family_name)
                    else:
                        fn_families.append(family_name)
                        if not candidates:
                            unavailable_families.append(family_name)
                        elif accepted_for_family:
                            off_target_only_families.append(family_name)
                        else:
                            undetected_families.append(family_name)
                key = f"{source}__to__{target}"
                dimension_rows[key] = {
                    "source": source,
                    "target": target,
                    "TP": len(tp_families),
                    "TN": len(true_negative_families),
                    "FP": len(false_positive_families),
                    "FN": len(fn_families),
                    "source_positive_families": len(source_anchors),
                    "eligible_positives": len(source_anchors) - len(unavailable_families),
                    "unavailable_positives": len(unavailable_families),
                    "eligible_positive_builds": candidate_builds,
                    "off_target_only": len(off_target_only_families),
                    "undetected_on_any_build": len(undetected_families),
                    "true_positive_families": sorted(tp_families),
                    "false_negative_families": sorted(fn_families),
                    "unavailable_families": sorted(unavailable_families),
                    "off_target_only_families": sorted(off_target_only_families),
                    "undetected_families": sorted(undetected_families),
                }
        transition_rows[dimension] = dimension_rows
    return {
        "run_signature": task["run_signature"],
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "program": task["program"],
        "optimization": task["optimization"],
        "expected_families": sorted(expected),
        "false_positive_families": false_positive_families,
        "true_negative_families": true_negative_families,
        "accepted_build_labels": len(accepted),
        "anchor_provenance": dict(anchor_provenance),
        "unresolved_anchor_families": unresolved_anchor_families,
        "excluded_out_of_scope_families": excluded_out_of_scope_families,
        "scenarios": scenario_rows,
        "transitions": transition_rows,
    }


def safe_worker(task: dict[str, str]) -> dict[str, Any]:
    try:
        return {"ok": True, "row": evaluate_case(task)}
    except Exception as exc:
        return {"ok": False, "error": {
            "case_id": task["case_id"],
            "feature": task["feature"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }}


def load_checkpoint(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_signature") == signature:
                completed[str(row["case_id"])] = row
    return completed


def metric_row(counts: Counter[str], conditioned: bool = False) -> dict[str, Any]:
    tp, tn, fp = counts["TP"], counts["TN"], counts["FP"]
    fn = counts["FN"] - counts["unavailable_positives"] if conditioned else counts["FN"]
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "coverage": (
            counts["eligible_positives"]
            / (counts["eligible_positives"] + counts["unavailable_positives"]
               + counts["family_only_fn"])
            if (counts["eligible_positives"] + counts["unavailable_positives"]
                + counts["family_only_fn"]) else 0.0
        ),
        "eligible_positives": counts["eligible_positives"],
        "unavailable_positives": counts["unavailable_positives"],
        "family_only_fn": counts["family_only_fn"],
        "off_scenario_only": (
            counts["off_scenario_only"] + counts["off_target_only"]
        ),
        "undetected_on_any_build": counts["undetected_on_any_build"],
        "eligible_positive_builds": counts["eligible_positive_builds"],
    }


def aggregate(
    completed: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    totals = {scenario: Counter() for scenario in SCENARIOS}
    family_rows: dict[str, dict[str, Counter[str]]] = {
        scenario: defaultdict(Counter) for scenario in SCENARIOS
    }
    fp_families: Counter[str] = Counter()
    provenance_totals: Counter[str] = Counter()
    unresolved_provenance_families: Counter[str] = Counter()
    excluded_out_of_scope_families: Counter[str] = Counter()
    transition_totals: dict[str, dict[str, Counter[str]]] = {
        dimension: defaultdict(Counter) for dimension in TRANSITION_VALUES
    }
    transition_families: dict[str, dict[str, dict[str, Counter[str]]]] = {
        dimension: defaultdict(lambda: defaultdict(Counter))
        for dimension in TRANSITION_VALUES
    }
    for row in completed.values():
        provenance_totals.update(row.get("anchor_provenance", {}))
        unresolved_provenance_families.update(
            row.get("unresolved_anchor_families", [])
        )
        excluded_out_of_scope_families.update(
            row.get("excluded_out_of_scope_families", [])
        )
        for name in row["false_positive_families"]:
            fp_families[name] += 1
        for scenario, values in row["scenarios"].items():
            totals[scenario].update({
                key: int(values[key]) for key in (
                    "TP", "TN", "FP", "FN", "eligible_positives",
                    "eligible_positive_builds", "unavailable_positives",
                    "off_scenario_only", "undetected_on_any_build",
                )
            })
            totals[scenario]["family_only_fn"] += int(values.get("family_only_fn", 0))
            tp = set(values["true_positive_families"])
            expected = tp | set(values["false_negative_families"])
            unavailable = set(values["unavailable_families"])
            off_scenario = set(values["off_scenario_only_families"])
            undetected = set(values["undetected_families"])
            family_only = set(values.get("family_only_fn_families", []))
            for family_name in expected:
                bucket = family_rows[scenario][family_name]
                bucket["occurrences"] += 1
                bucket["TP"] += family_name in tp
                bucket["FN"] += family_name not in tp
                bucket["unavailable"] += family_name in unavailable
                bucket["off_scenario_only"] += family_name in off_scenario
                bucket["undetected_on_any_build"] += family_name in undetected
                bucket["family_only_fn"] += family_name in family_only
        for dimension, rows in row.get("transitions", {}).items():
            for key, values in rows.items():
                totals_for_transition = transition_totals[dimension][key]
                totals_for_transition["elfs"] += 1
                totals_for_transition.update({
                    name: int(values[name]) for name in (
                        "TP", "TN", "FP", "FN", "source_positive_families",
                        "eligible_positives", "unavailable_positives",
                        "eligible_positive_builds", "off_target_only",
                        "undetected_on_any_build",
                    )
                })
                tp = set(values["true_positive_families"])
                unavailable = set(values["unavailable_families"])
                off_target = set(values["off_target_only_families"])
                undetected = set(values["undetected_families"])
                for family_name in values["false_negative_families"] + values["true_positive_families"]:
                    bucket = transition_families[dimension][key][family_name]
                    bucket["occurrences"] += 1
                    bucket["TP"] += family_name in tp
                    bucket["FN"] += family_name not in tp
                    bucket["unavailable"] += family_name in unavailable
                    bucket["off_target_only"] += family_name in off_target
                    bucket["undetected_on_any_build"] += family_name in undetected
    summaries = {
        scenario: {
            "strict": metric_row(values, conditioned=False),
            "coverage_conditioned": metric_row(values, conditioned=True),
        }
        for scenario, values in totals.items()
    }
    diagnostics = {
        "anchor_provenance": dict(provenance_totals),
        "unresolved_provenance_families": [
            {"family": name, "elf_count": count}
            for name, count in unresolved_provenance_families.most_common()
        ],
        "excluded_out_of_scope_families": [
            {"family": name, "elf_count": count}
            for name, count in excluded_out_of_scope_families.most_common()
        ],
        "false_positive_families": [
            {"family": name, "elf_count": count}
            for name, count in fp_families.most_common()
        ],
        "positive_family_outcomes": {
            scenario: [
                {"family": name, **dict(counts)}
                for name, counts in sorted(
                    rows.items(), key=lambda item: (-item[1]["FN"], item[0])
                )
            ]
            for scenario, rows in family_rows.items()
        },
    }
    transitions: dict[str, Any] = {}
    for dimension, rows in transition_totals.items():
        transitions[dimension] = {}
        for key, counts in sorted(rows.items()):
            strict = metric_row(counts, conditioned=False)
            conditioned = metric_row(counts, conditioned=True)
            transitions[dimension][key] = {
                "source": key.split("__to__", 1)[0],
                "target": key.split("__to__", 1)[1],
                "evaluated_elfs": counts["elfs"],
                "source_positive_families": counts["source_positive_families"],
                "strict": strict,
                "coverage_conditioned": conditioned,
                "family_outcomes": [
                    {"family": name, **dict(family_counts)}
                    for name, family_counts in sorted(
                        transition_families[dimension][key].items(),
                        key=lambda item: (-item[1]["FN"], item[0]),
                    )
                ],
            }
    return summaries, diagnostics, transitions


def render_markdown(report: dict[str, Any]) -> str:
    """Create a self-contained, thesis-oriented summary of the completed run."""
    family_fallback = bool(report.get("protocol", {}).get("family_only_fn_exception"))
    labels = {
        "same_exact_artifact": "Same exact artifact",
        "same_build_configuration": "Same build configuration",
        "cross_optimization_any": "Cross optimization",
        "cross_compiler_any": "Cross compiler",
        "cross_version_any": "Cross version",
    }
    provenance = report["diagnostics"].get("anchor_provenance", {})
    unresolved = report["diagnostics"].get("unresolved_provenance_families", [])
    lines = [
        "# Full-corpus constrained-positive evaluation",
        "",
        "## Obiettivo e protocollo",
        "",
        "Questa valutazione misura se LIVA riconosce una libreria presente usando "
        "una build compatibile con lo scenario, mantenendo contemporaneamente aperto "
        "l'intero corpus di build per tutte le famiglie assenti. È quindi uno stress "
        "test più severo del protocollo a corpus controllato.",
        "",
        "L'unità statistica è la coppia `(ELF, famiglia logica)`. Per una famiglia "
        "presente, si assegna un TP soltanto se almeno una build ammessa dallo scenario "
        "viene accettata; un'accettazione limitata a build non ammesse resta un FN ed è "
        "registrata come `off-scenario only`. Per una famiglia assente, qualunque build "
        "accettata produce un solo FP dopo il collasso a livello di famiglia; in assenza "
        "di accettazioni si ottiene un TN.",
        "",
        f"Sono stati valutati **{report['evaluated_elfs']} ELF** su "
        f"**{report['requested_elfs']} disponibili**, contro **{report['candidate_builds']} "
        f"build** appartenenti a **{report['candidate_families']} famiglie**.",
        "",
        "Gli scenari sono:",
        "",
        "- **Same exact artifact:** soltanto l'identico archivio effettivamente linkato può produrre un TP.",
        "- **Same build configuration:** stessa famiglia, versione, compilatore e ottimizzazione; il path fisico dell'archivio può differire.",
        "- **Cross optimization:** stessa famiglia, pacchetto, versione e compilatore, ma ottimizzazione diversa.",
        "- **Cross compiler:** stessa famiglia, pacchetto, versione e ottimizzazione, ma compilatore diverso.",
        "- **Cross version:** stessa famiglia, pacchetto, compilatore e ottimizzazione, ma versione diversa.",
        "- Per le famiglie non presenti nell'ELF, tutti i 4.472 candidati disponibili restano in competizione in ogni scenario.",
        "",
        "## Riconciliazione della provenance",
        "",
        "La ground truth può riferirsi sia all'archivio installato presente nella "
        "candidate matrix, sia a un archivio intermedio generato nel build tree del "
        "programma. Nel secondo caso versione sorgente, compilatore e ottimizzazione "
        "sono ricostruiti dal path del build e associati alla corrispondente voce della "
        "matrice. Questa associazione abilita il confronto `same build configuration` "
        "senza sostenere che i due file siano byte-identici.",
        "",
        f"Archivi risolti per path esatto: **{provenance.get('exact_matrix_path', 0)}**; "
        f"archivi intermedi risolti dal build path: "
        f"**{provenance.get('program_build_path', 0)}**; provenance non risolta: "
        f"**{provenance.get('unresolved', 0)}**.",
        "",
    ]
    if unresolved:
        unresolved_text = ", ".join(
            f"`{row['family']}` ({row['elf_count']})" for row in unresolved
        )
        if family_fallback:
            lines.extend([
                "Le provenance non risolte sono: " + unresolved_text + ". "
                "Nel solo scenario `same build configuration`, queste occorrenze "
                "sono incluse come FN a livello di famiglia, senza candidato di "
                "build compatibile. Non sono `off-scenario only`: nessuna build "
                "della famiglia viene accettata. Restano escluse dagli altri scenari.",
                "",
            ])
        else:
            lines.extend([
                "Le provenance non risolte sono: " + unresolved_text + ". Questi casi "
                "sono esterni alla matrice controllata e vengono esclusi completamente "
                "dall'universo della valutazione, senza classificarli come FP, FN o TN.",
                "",
            ])
    lines.extend([
        "## Risultati",
        "",
        "### Dominio valutato",
        "",
        "Questa vista esclude dal denominatore le combinazioni non previste dal disegno "
        "sperimentale o non prodotte dalla versione target. Queste combinazioni sono "
        "fuori supporto, non errori del matcher. Il corpus negativo rimane invece "
        "completamente aperto.",
        "",
        "| Scenario | TP | FP | FN | TN | Precision | Recall | F1 | Fuori supporto | Off-scenario only | Family-only FN |" if family_fallback else
        "| Scenario | TP | FP | FN | TN | Precision | Recall | F1 | Fuori supporto | Off-scenario only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" if family_fallback else
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if family_fallback:
        lines.insert(
            lines.index("## Risultati") + 5,
            "\nEccezione esplicita: i 15 casi `libm` senza build compatibile sono "
            "comunque conteggiati come FN nel solo scenario same-configuration.",
        )
    for scenario in SCENARIOS:
        row = report["scenarios"][scenario]["coverage_conditioned"]
        lines.append(
            f"| {labels[scenario]} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{row['TN']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['unavailable_positives']} | "
            f"{row['off_scenario_only']} |"
            + (f" {row['family_only_fn']} |" if family_fallback else "")
        )

    fps = report["diagnostics"]["false_positive_families"][:15]
    lines.extend([
        "",
        "## Famiglie maggiormente associate ai falsi positivi",
        "",
        "| Famiglia | ELF con FP |",
        "|---|---:|",
    ])
    if fps:
        lines.extend(f"| {row['family']} | {row['elf_count']} |" for row in fps)
    else:
        lines.append("| Nessuna | 0 |")

    lines.extend([
        "",
        "## Principali falsi negativi per scenario",
        "",
    ])
    family_outcomes = report["diagnostics"]["positive_family_outcomes"]
    for scenario in SCENARIOS:
        family_column = family_fallback and scenario == "same_build_configuration"
        lines.extend([
            f"### {labels[scenario]}",
            "",
            "| Famiglia | Occorrenze | TP | FN | Fuori supporto | Off-scenario | Mai rilevata | Family-only FN |" if family_column else
            "| Famiglia | Occorrenze | TP | FN | Fuori supporto | Off-scenario | Mai rilevata |",
            "|---|---:|---:|---:|---:|---:|---:|---:|" if family_column else
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in family_outcomes[scenario][:15]:
            lines.append(
                f"| {row['family']} | {row.get('occurrences', 0)} | "
                f"{row.get('TP', 0)} | {row.get('FN', 0)} | "
                f"{row.get('unavailable', 0)} | {row.get('off_scenario_only', 0)} | "
                f"{row.get('undetected_on_any_build', 0)} |"
                + (f" {row.get('family_only_fn', 0)} |" if family_column else "")
            )
        lines.append("")

    lines.extend([
        "## Matrici direzionali di trasferimento",
        "",
        "Le celle riportano `rilevati / eleggibili (recall condizionata)`. La riga "
        "identifica la configurazione realmente linkata nell'ELF, la colonna la "
        "configurazione della build candidate. La diagonale è esclusa perché non è "
        "cross-build.",
        "",
    ])
    dimension_titles = {
        "optimization": "Ottimizzazione",
        "compiler": "Compilatore",
        "version": "Versione",
    }
    for dimension, values in TRANSITION_VALUES.items():
        lines.extend([
            f"### {dimension_titles[dimension]}",
            "",
            "| Sorgente → target | " + " | ".join(values) + " |",
            "|---|" + "---:|" * len(values),
        ])
        transition_rows = report["transitions"][dimension]
        for source in values:
            cells: list[str] = []
            for target in values:
                if source == target:
                    cells.append("—")
                    continue
                row = transition_rows.get(f"{source}__to__{target}")
                if row is None:
                    cells.append("n.d.")
                    continue
                conditioned = row["coverage_conditioned"]
                eligible = conditioned["eligible_positives"]
                cells.append(
                    f"{conditioned['TP']}/{eligible} "
                    f"({conditioned['recall']:.4f})"
                )
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        lines.extend([
            "",
            "Dettaglio numerico:",
            "",
            "| Transizione | ELF | Positivi sorgente | TP | FN cond. | Recall cond. | Copertura | Off-target |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for key, row in sorted(transition_rows.items()):
            conditioned = row["coverage_conditioned"]
            lines.append(
                f"| {row['source']} → {row['target']} | {row['evaluated_elfs']} | "
                f"{row['source_positive_families']} | {conditioned['TP']} | "
                f"{conditioned['FN']} | {conditioned['recall']:.4f} | "
                f"{conditioned['coverage']:.4f} | "
                f"{conditioned['off_scenario_only']} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretazione metodologica",
        "",
        "FP e TN sono identici nei cinque scenari perché il lato negativo è "
        "deliberatamente invariato: tutte le build delle famiglie assenti sono sempre "
        "ammesse. Cambiano TP e FN, cioè la capacità di trasferire il match alla build "
        "richiesta; di conseguenza cambiano anche precisione, recall e F1.",
        "",
        "Un valore elevato di `off-scenario only` indica che la famiglia è riconoscibile, "
        "ma non attraverso la trasformazione controllata in esame. Un valore elevato di "
        "`mai rilevata` indica invece un errore di detection più generale. I casi `fuori "
        "supporto` descrivono combinazioni escluse per costruzione o archivi che la "
        "versione target non produce; non sono interpretati come fallimenti del matcher.",
        "",
        "Questo protocollo è truth-conditioned: la restrizione applicata alle famiglie "
        "positive dipende dalla ground truth. È appropriato come analisi diagnostica e "
        "stress test, ma non rappresenta da solo un corpus di deployment costruibile "
        "senza conoscere a priori le librerie presenti. Per il confronto causale puro "
        "tra trasformazioni di build va affiancato al protocollo a corpus controllato.",
        "",
        "## Riproducibilità",
        "",
        "Il calcolo è un replay offline dei feature log `replay_complete` con gli "
        "iperparametri finali. Non riesegue disassemblaggio, generazione di embedding o "
        "matching binario. Il file `cases.jsonl` è un checkpoint incrementale: quando "
        "saranno aggiunti nuovi report, una nuova esecuzione elaborerà soltanto gli ELF "
        "non ancora presenti con la stessa firma di configurazione.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir", type=Path,
        default=UNIFIED / "full_corpus_constrained_positive",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cases.jsonl"
    errors = args.output_dir / "errors.jsonl"

    meta = candidate_metadata()
    families = {value["family"] for value in meta.values()}
    signature = canonical_hash({
        "schema_version": 4,
        "protocol": "full_corpus_constrained_positive",
        "params": PARAMS,
        "candidate_metadata": meta,
        "scenarios": SCENARIOS,
        "positive_rule": "only scenario-compatible labels may produce TP",
        "negative_rule": "all build labels remain eligible; collapse FP by family",
        "provenance_rule": "exact path first; otherwise infer source version/compiler/optimization from library or program-build path",
        "scope_rule": "exclude positive families whose linked archive has no resolvable controlled-build provenance",
    })
    reports = report_index_all()
    tasks: list[dict[str, str]] = []
    for compiler in COMPILERS:
        for (program, optimization), report in sorted(reports[compiler].items()):
            tasks.append({
                "run_signature": signature,
                "case_id": f"{program}__{compiler}__{optimization}",
                "compiler": compiler,
                "program": program,
                "optimization": optimization,
                "feature": str(report.with_name(report.name.replace(
                    ".report.txt", ".features.jsonl.gz"
                ))),
                "truth": str(family.truth_path(program, compiler, optimization)),
            })
    if args.limit > 0:
        tasks = tasks[:args.limit]
    completed = load_checkpoint(checkpoint, signature)
    pending = [task for task in tasks if task["case_id"] not in completed]
    print(json.dumps({
        "run_signature": signature,
        "total_elfs": len(tasks),
        "resumed_elfs": len(completed),
        "pending_elfs": len(pending),
        "candidate_builds": len(meta),
        "candidate_families": len(families),
        "workers": args.workers,
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)

    started = time.monotonic()
    if pending:
        with checkpoint.open("a", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err:
            with multiprocessing.Pool(
                args.workers, initializer=init_worker, initargs=(meta, families)
            ) as pool:
                for index, result in enumerate(pool.imap_unordered(safe_worker, pending), 1):
                    if result["ok"]:
                        row = result["row"]
                        completed[row["case_id"]] = row
                        out.write(json.dumps(row, sort_keys=True) + "\n")
                        out.flush()
                    else:
                        err.write(json.dumps(result["error"], sort_keys=True) + "\n")
                        err.flush()
                        print(
                            f"ERROR {result['error']['case_id']}: {result['error']['error']}",
                            flush=True,
                        )
                    if index % 10 == 0 or index == len(pending):
                        print(
                            f"progress {len(completed)}/{len(tasks)} "
                            f"elapsed={time.monotonic()-started:.1f}s",
                            flush=True,
                        )

    summaries, diagnostics, transitions = aggregate(completed)
    report = {
        "schema_version": 4,
        "run_signature": signature,
        "complete": len(completed) == len(tasks),
        "evaluated_elfs": len(completed),
        "requested_elfs": len(tasks),
        "candidate_builds": len(meta),
        "candidate_families": len(families),
        "configuration": PARAMS,
        "protocol": {
            "positive_families": "restricted to scenario-compatible build labels",
            "absent_families": "all available build labels remain candidates",
            "aggregation": "one confusion-matrix outcome per ELF and logical family",
            "off_scenario_positive_acceptance": "diagnostic only; it remains an FN",
            "provenance_reconciliation": "exact candidate path when available; otherwise source version/compiler/optimization inferred from build path and mapped to candidate package/version role",
            "out_of_scope": "linked archives without controlled-build provenance are ignored by the confusion matrix",
        },
        "scenarios": summaries,
        "transitions": transitions,
        "diagnostics": diagnostics,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = args.output_dir / "thesis_report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    summary_path = args.output_dir / "summary.csv"
    fields = [
        "scenario", "view", "TP", "TN", "FP", "FN", "precision", "recall",
        "f1", "coverage", "eligible_positives", "unavailable_positives",
        "off_scenario_only", "undetected_on_any_build", "eligible_positive_builds",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario, views in summaries.items():
            for view, values in views.items():
                writer.writerow({"scenario": scenario, "view": view, **values})
    transition_path = args.output_dir / "transition_summary.csv"
    transition_fields = [
        "dimension", "source", "target", "view", "evaluated_elfs",
        "source_positive_families", "TP", "TN", "FP", "FN", "precision",
        "recall", "f1", "coverage", "eligible_positives",
        "unavailable_positives", "off_scenario_only",
        "undetected_on_any_build", "eligible_positive_builds",
    ]
    with transition_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=transition_fields)
        writer.writeheader()
        for dimension, rows in transitions.items():
            for row in rows.values():
                for view in ("strict", "coverage_conditioned"):
                    writer.writerow({
                        "dimension": dimension,
                        "source": row["source"],
                        "target": row["target"],
                        "view": view,
                        "evaluated_elfs": row["evaluated_elfs"],
                        "source_positive_families": row["source_positive_families"],
                        **row[view],
                    })
    transition_family_path = args.output_dir / "transition_family_outcomes.csv"
    family_fields = [
        "dimension", "source", "target", "family", "occurrences", "TP",
        "FN", "unavailable", "off_target_only", "undetected_on_any_build",
    ]
    with transition_family_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=family_fields)
        writer.writeheader()
        for dimension, rows in transitions.items():
            for row in rows.values():
                for outcome in row["family_outcomes"]:
                    writer.writerow({
                        "dimension": dimension,
                        "source": row["source"],
                        "target": row["target"],
                        **outcome,
                    })
    print(json.dumps({
        "complete": report["complete"],
        "evaluated_elfs": len(completed),
        "report": str(report_path),
        "summary": str(summary_path),
        "transition_summary": str(transition_path),
        "transition_family_outcomes": str(transition_family_path),
        "markdown": str(markdown_path),
        "scenarios": summaries,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
