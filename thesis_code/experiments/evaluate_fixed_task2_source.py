#!/usr/bin/env python3
"""Evaluate Task 2 with source-content-verified CU equivalence.

Primary identity: (logical family, normalized CU name, normalized-code SHA-256).
When a CU/source-version cannot be mapped unambiguously, it is retained with
an opaque source-version identity.  Such a CU can match directly within the
same recorded version but is never granted unverified cross-version credit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNIFIED = REPO_ROOT / "libseeker-unified"
sys.path.insert(0, str(SCRIPT_DIR))

import tune_family_f1_optuna as family  # noqa: E402
import evaluate_fixed_family_f1 as fixed  # noqa: E402
from audit_cu_source_equivalence import enrich_label_metadata  # noqa: E402
from evaluate_fixed_two_tasks import report_index_all  # noqa: E402


ENGINE = family.engine
COMPILERS = family.COMPILERS

Identity = tuple[str, str, str]
SourceKey = tuple[str, str, str]

_LABEL_META: dict[str, dict[str, str]] = {}
_CANDIDATE_FAMILIES: set[str] = set()
_SOURCE_HASHES: dict[SourceKey, str] = {}
_CU_UNIVERSE: set[Identity] = set()

TASK_IDENTITY = (
    "logical family|normalized CU|normalized-code SHA-256 "
    "(C/C++ comments and insignificant formatting ignored)"
)
EQUIVALENCE_RULE = (
    "another version is equivalent only for the same normalized CU "
    "with identical verified source SHA-256"
)
UNVERIFIED_RULE = (
    "ambiguous/unmapped CU is retained but can match only the exact "
    "recorded source version; no cross-version credit"
)
PREDICTION_RULE = (
    "all CU records accepted by CU-level gates; the family aggregate "
    "threshold is not a second gate for Task 2"
)


def canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def render_identity(identity: Identity) -> str:
    return "|".join(identity)


def source_identity(
    family_name: str,
    cu_name: str,
    source_version: str,
    source_hashes: dict[SourceKey, str],
) -> tuple[Identity, bool]:
    normalized = ENGINE.normalize_cu_name(cu_name)
    digest = source_hashes.get((family_name, normalized, source_version))
    if digest is not None:
        return (family_name, normalized, f"sha256:{digest}"), True
    # Conservative fallback: direct-version matches remain measurable, while
    # no equivalence across versions is asserted without source verification.
    return (
        family_name,
        normalized,
        f"unverified-source-version:{source_version}",
    ), False


def load_source_audit(
    path: Path,
) -> tuple[dict[SourceKey, str], set[Identity], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes: dict[SourceKey, str] = {}
    universe: set[Identity] = set()
    for row in payload.get("mappings", []):
        key = (
            str(row["family"]),
            str(row["cu"]),
            str(row["source_version"]),
        )
        digest = None
        if row.get("status") == "mapped":
            digest = row.get("source_code_sha256") or row.get("source_sha256")
        if digest:
            hashes[key] = str(digest)
        identity, _verified = source_identity(*key, hashes)
        universe.add(identity)
    return hashes, universe, payload


def expected_source_keys(
    path: Path, candidate_families: set[str]
) -> tuple[set[SourceKey], Counter[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: set[SourceKey] = set()
    audit: Counter[str] = Counter()
    for archive in payload.get("archives", []):
        confirmed = archive.get("confirmed_compilation_units")
        if confirmed is None:
            confirmed = [
                unit["compilation_unit"]
                for unit in archive.get("compilation_units", []) or []
                if unit.get("included")
                and unit.get("ground_truth_method") == "linker_map"
            ]
        elif archive.get("confirmed_compilation_units_method") != "linker_map":
            confirmed = []
        archive_name = str(
            archive.get("archive_basename")
            or Path(str(archive.get("archive", ""))).name
        )
        family_name = family.normalize_archive_family(
            archive_name, candidate_families
        )
        if family_name is None:
            audit["confirmed_outside_candidate_families"] += len(confirmed)
            continue
        source_version = str(archive.get("source") or "unknown")
        for member in confirmed:
            result.add(
                (
                    family_name,
                    ENGINE.normalize_cu_name(str(member)),
                    source_version,
                )
            )
    audit["source_keys"] = len(result)
    return result, audit


def init_worker(
    label_meta: dict[str, dict[str, str]],
    candidate_families: set[str],
    source_hashes: dict[SourceKey, str],
    universe: set[Identity],
) -> None:
    global _LABEL_META, _CANDIDATE_FAMILIES, _SOURCE_HASHES, _CU_UNIVERSE
    _LABEL_META = label_meta
    _CANDIDATE_FAMILIES = candidate_families
    _SOURCE_HASHES = source_hashes
    _CU_UNIVERSE = universe


def evaluate_case(task: dict[str, str]) -> dict[str, Any]:
    header, libraries, calls = fixed.parse_relevant_features(Path(task["feature"]))
    params = {
        **ENGINE.DECISION_DEFAULTS,
        **header.get("matching_configuration", {}),
        **fixed.PARAMS,
    }
    predicted: set[Identity] = set()
    predicted_verified: set[Identity] = set()
    accepted_build_labels = 0
    for label, records in libraries.items():
        metadata = _LABEL_META.get(label)
        if metadata is None:
            continue
        accepted_records, library_score = ENGINE.library_match_evidence(
            records, params, calls
        )
        if accepted_records and library_score >= float(params["library_min_score"]):
            accepted_build_labels += 1
        for record in accepted_records:
            name = str(record.get("name") or "")
            if not name:
                continue
            identity, verified = source_identity(
                metadata["family"], name, metadata["source"], _SOURCE_HASHES
            )
            predicted.add(identity)
            if verified:
                predicted_verified.add(identity)

    expected_keys, truth_audit = expected_source_keys(
        Path(task["truth"]), _CANDIDATE_FAMILIES
    )
    expected: set[Identity] = set()
    expected_verified: set[Identity] = set()
    for key in expected_keys:
        identity, verified = source_identity(*key, _SOURCE_HASHES)
        expected.add(identity)
        if verified:
            expected_verified.add(identity)

    tp = predicted & expected
    fp = predicted - expected
    fn = expected - predicted
    evaluated_universe = _CU_UNIVERSE | expected | predicted
    tn = evaluated_universe - predicted - expected
    verified_universe = {
        identity
        for identity in evaluated_universe
        if identity[2].startswith("sha256:")
    }
    verified_tp = predicted_verified & expected_verified
    verified_fp = predicted_verified - expected_verified
    verified_fn = expected_verified - predicted_verified
    verified_tn = verified_universe - predicted_verified - expected_verified
    return {
        "run_signature": task["run_signature"],
        "case_id": task["case_id"],
        "compiler": task["compiler"],
        "program": task["program"],
        "optimization": task["optimization"],
        "TP": len(tp),
        "TN": len(tn),
        "FP": len(fp),
        "FN": len(fn),
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "expected_verified_by_source_hash": len(expected_verified),
        "expected_unverified_fallback": len(expected - expected_verified),
        "predicted_verified_by_source_hash": len(predicted_verified),
        "predicted_unverified_fallback": len(predicted - predicted_verified),
        "accepted_build_labels_at_family_threshold": accepted_build_labels,
        "tp": sorted(render_identity(value) for value in tp),
        "fp": sorted(render_identity(value) for value in fp),
        "fn": sorted(render_identity(value) for value in fn),
        "verified_source_only": {
            "TP": len(verified_tp),
            "TN": len(verified_tn),
            "FP": len(verified_fp),
            "FN": len(verified_fn),
            "tp": sorted(render_identity(value) for value in verified_tp),
            "fp": sorted(render_identity(value) for value in verified_fp),
            "fn": sorted(render_identity(value) for value in verified_fn),
        },
        "truth_audit": dict(truth_audit),
    }


def evaluate_case_safe(task: dict[str, str]) -> dict[str, Any]:
    try:
        return {"ok": True, "row": evaluate_case(task)}
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "run_signature": task["run_signature"],
                "case_id": task["case_id"],
                "compiler": task["compiler"],
                "program": task["program"],
                "optimization": task["optimization"],
                "feature": task["feature"],
                "truth": task["truth"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }


def metrics(counts: Counter[str]) -> dict[str, int | float]:
    tp, tn, fp, fn = (counts[k] for k in ("TP", "TN", "FP", "FN"))
    total = tp + tn + fp + fn
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "evaluated_decisions": total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def summarize(
    rows: Iterable[dict[str, Any]], nested: str | None = None
) -> dict[str, Any]:
    rows = list(rows)
    per_compiler: list[dict[str, Any]] = []
    pooled: Counter[str] = Counter()
    coverage = Counter()
    for compiler in COMPILERS:
        selected = [r for r in rows if r["compiler"] == compiler]
        counts: Counter[str] = Counter()
        for row in selected:
            values = row[nested] if nested else row
            counts.update({k: int(values[k]) for k in ("TP", "TN", "FP", "FN")})
        pooled.update(counts)
        per_compiler.append(
            {"compiler": compiler, "evaluated_elfs": len(selected), **metrics(counts)}
        )
    for row in rows:
        for key in (
            "expected_verified_by_source_hash",
            "expected_unverified_fallback",
            "predicted_verified_by_source_hash",
            "predicted_unverified_fallback",
        ):
            coverage[key] += int(row[key])
    expected_total = (
        coverage["expected_verified_by_source_hash"]
        + coverage["expected_unverified_fallback"]
    )
    predicted_total = (
        coverage["predicted_verified_by_source_hash"]
        + coverage["predicted_unverified_fallback"]
    )
    return {
        "per_compiler": per_compiler,
        "pooled": {
            **metrics(pooled),
            "macro_compiler_f1": sum(float(r["f1"]) for r in per_compiler)
            / len(per_compiler),
        },
        "source_verification_coverage": {
            **coverage,
            "expected_verified_ratio": (
                coverage["expected_verified_by_source_hash"] / expected_total
                if expected_total else 0.0
            ),
            "predicted_verified_ratio": (
                coverage["predicted_verified_by_source_hash"] / predicted_total
                if predicted_total else 0.0
            ),
        },
    }


def load_checkpoint(
    path: Path,
    signature: str,
    compatible_signatures: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load per-ELF rows and identify compatible legacy rows to migrate."""
    compatible = set(compatible_signatures or ())
    stable: dict[str, dict[str, Any]] = {}
    legacy: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return stable, set()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row_signature = str(row.get("run_signature", ""))
        case_id = str(row["case_id"])
        if row_signature == signature:
            stable[case_id] = row
        elif row_signature in compatible:
            legacy[case_id] = row
    migrated: set[str] = set()
    for case_id, row in legacy.items():
        if case_id in stable:
            continue
        stable[case_id] = {**row, "run_signature": signature}
        migrated.add(case_id)
    return stable, migrated


def compatible_report_signatures(
    path: Path,
    *,
    candidate_family_count: int,
    source_audit_sha256: str,
    source_audit_summary: dict[str, Any],
) -> tuple[set[str], bool]:
    """Accept a legacy corpus-bound run only when its semantics match."""
    if not path.is_file():
        return set(), False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set(), False
    signature = payload.get("run_signature")
    recorded_audit_hash = payload.get("source_audit_sha256")
    recorded_identity = payload.get("identity")
    legacy_raw_hash_identity = (
        recorded_identity == "logical family|normalized CU|source SHA-256"
    )
    recorded_summary = payload.get("source_audit_summary") or {}
    if recorded_audit_hash is not None:
        audit_matches = recorded_audit_hash == source_audit_sha256
    elif legacy_raw_hash_identity:
        # The normalized-code audit changes cross-version grouping counts but
        # preserves the audited rows and their mapping statuses.  Raw source
        # hashes in the checkpoint are converted below using that audit.
        audit_matches = (
            recorded_summary.get("cu_source_version_rows")
            == source_audit_summary.get("cu_source_version_rows")
            and recorded_summary.get("status_counts")
            == source_audit_summary.get("status_counts")
        )
    else:
        audit_matches = recorded_summary == source_audit_summary
    if (
        not isinstance(signature, str)
        or payload.get("configuration") != fixed.PARAMS
        or int(payload.get("candidate_families", -1)) != candidate_family_count
        or recorded_identity
        not in {TASK_IDENTITY, "logical family|normalized CU|source SHA-256"}
        or payload.get("equivalence_rule") != EQUIVALENCE_RULE
        or payload.get("unverified_rule") != UNVERIFIED_RULE
        or payload.get("prediction_rule") != PREDICTION_RULE
        or not audit_matches
    ):
        return set(), False
    return {signature}, legacy_raw_hash_identity


def raw_to_normalized_hashes(
    audit_payload: dict[str, Any],
) -> dict[tuple[str, str, str], str]:
    """Map legacy raw source-file hashes to normalized-code hashes."""
    result: dict[tuple[str, str, str], str] = {}
    for mapping in audit_payload.get("mappings", []):
        if mapping.get("status") != "mapped":
            continue
        normalized = mapping.get("source_code_sha256")
        if not normalized:
            continue
        family_name = str(mapping["family"])
        cu_name = str(mapping["cu"])
        for raw_digest in (mapping.get("candidates_by_file_sha256") or {}):
            result[(family_name, cu_name, str(raw_digest))] = str(normalized)
    return result


def convert_cached_identity(
    value: str,
    raw_to_normalized: dict[tuple[str, str, str], str],
) -> str:
    family_name, cu_name, source_identity_value = value.split("|", 2)
    if source_identity_value.startswith("sha256:"):
        raw_digest = source_identity_value.removeprefix("sha256:")
        normalized = raw_to_normalized.get((family_name, cu_name, raw_digest))
        if normalized is not None:
            source_identity_value = f"sha256:{normalized}"
    return "|".join((family_name, cu_name, source_identity_value))


def reclassify_cached_row(
    original: dict[str, Any],
    universe: set[Identity],
    raw_to_normalized: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, Any]:
    """Convert legacy identities and recompute all set-derived counters."""
    conversion = raw_to_normalized or {}
    expected = {
        convert_cached_identity(value, conversion)
        for value in (*original.get("tp", ()), *original.get("fn", ()))
    }
    predicted = {
        convert_cached_identity(value, conversion)
        for value in (*original.get("tp", ()), *original.get("fp", ()))
    }
    rendered_universe = {render_identity(value) for value in universe}
    tp = predicted & expected
    fp = predicted - expected
    fn = expected - predicted
    tn = rendered_universe - predicted - expected
    verified_universe = {
        value for value in rendered_universe if "|sha256:" in value
    }
    expected_verified = {value for value in expected if "|sha256:" in value}
    predicted_verified = {value for value in predicted if "|sha256:" in value}
    verified_tp = predicted_verified & expected_verified
    verified_fp = predicted_verified - expected_verified
    verified_fn = expected_verified - predicted_verified
    verified_tn = verified_universe - predicted_verified - expected_verified
    return {
        **original,
        "TP": len(tp),
        "TN": len(tn),
        "FP": len(fp),
        "FN": len(fn),
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "expected_verified_by_source_hash": len(expected_verified),
        "expected_unverified_fallback": len(expected - expected_verified),
        "predicted_verified_by_source_hash": len(predicted_verified),
        "predicted_unverified_fallback": len(predicted - predicted_verified),
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
        "verified_source_only": {
            "TP": len(verified_tp),
            "TN": len(verified_tn),
            "FP": len(verified_fp),
            "FN": len(verified_fn),
            "tp": sorted(verified_tp),
            "fp": sorted(verified_fp),
            "fn": sorted(verified_fn),
        },
    }


def refresh_cached_tn(
    rows: dict[str, dict[str, Any]], universe: set[Identity]
) -> set[str]:
    """Refresh cheap TN counts if newly imported truth expands the universe."""
    changed: set[str] = set()
    for case_id, original in list(rows.items()):
        row = reclassify_cached_row(original, universe)
        if (
            int(original.get("TN", -1)) == int(row["TN"])
            and int(original.get("verified_source_only", {}).get("TN", -1))
            == int(row["verified_source_only"]["TN"])
        ):
            continue
        rows[case_id] = row
        changed.add(case_id)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-per-compiler", type=int, default=0)
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=UNIFIED / "cu_source_equivalence_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNIFIED / "task2_source_verified",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "task2_cases.jsonl"
    error_log = args.output_dir / "task2_errors.jsonl"
    report_path = args.output_dir / "task2_report.json"

    label_meta = enrich_label_metadata()
    candidate_families = {m["family"] for m in label_meta.values()}
    source_hashes, universe, audit_payload = load_source_audit(args.source_audit)
    reports = report_index_all()
    tasks: list[dict[str, str]] = []
    coordinates: dict[str, list[str]] = {}
    for compiler in COMPILERS:
        selected = sorted(reports[compiler].items())
        if args.limit_per_compiler:
            selected = selected[: args.limit_per_compiler]
        coordinates[compiler] = [f"{p}__{o}" for (p, o), _ in selected]
        for (program, optimization), report in selected:
            tasks.append(
                {
                    "case_id": f"{program}__{compiler}__{optimization}",
                    "compiler": compiler,
                    "program": program,
                    "optimization": optimization,
                    "feature": str(report.with_name(report.name.replace(
                        ".report.txt", ".features.jsonl.gz"
                    ))),
                    "truth": str(family.truth_path(program, compiler, optimization)),
                }
            )

    # Ground-truth-only identities must also belong to the fixed TN universe.
    for task in tasks:
        keys, _audit = expected_source_keys(Path(task["truth"]), candidate_families)
        for key in keys:
            identity, _verified = source_identity(*key, source_hashes)
            universe.add(identity)

    source_audit_sha256 = hashlib.sha256(
        args.source_audit.read_bytes()
    ).hexdigest()
    label_metadata_hash = canonical_hash(label_meta)
    # The signature excludes coordinates and the aggregate TN universe.
    # TP/FP/FN are independent per ELF; if later truth expands the universe,
    # cached TN counts are refreshed without replaying large feature logs.
    signature = canonical_hash(
        {
            "schema_version": 3,
            "protocol": "task2_source_verified_per_case",
            "configuration": fixed.PARAMS,
            "source_audit_sha256": source_audit_sha256,
            "label_metadata_hash": label_metadata_hash,
            "candidate_families": sorted(candidate_families),
            "identity": TASK_IDENTITY,
            "equivalence_rule": EQUIVALENCE_RULE,
            "prediction_rule": PREDICTION_RULE,
            "unverified_rule": UNVERIFIED_RULE,
        }
    )
    for task in tasks:
        task["run_signature"] = signature
    legacy_signatures, normalize_legacy_hashes = compatible_report_signatures(
        report_path,
        candidate_family_count=len(candidate_families),
        source_audit_sha256=source_audit_sha256,
        source_audit_summary=audit_payload.get("summary", {}),
    )
    completed, migrated = load_checkpoint(
        checkpoint, signature, legacy_signatures
    )
    if normalize_legacy_hashes:
        hash_mapping = raw_to_normalized_hashes(audit_payload)
        for case_id in migrated:
            completed[case_id] = reclassify_cached_row(
                completed[case_id], universe, hash_mapping
            )
    refreshed = refresh_cached_tn(completed, universe)
    persist = migrated | refreshed
    if persist:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as out:
            for case_id in sorted(persist):
                out.write(json.dumps(completed[case_id], sort_keys=True) + "\n")
        print(
            f"migrated {len(migrated)} compatible case(s); "
            f"refreshed TN counts for {len(refreshed)} cached case(s)",
            flush=True,
        )
    pending = [t for t in tasks if t["case_id"] not in completed]
    print(json.dumps({
        "run_signature": signature,
        "total_elfs": len(tasks),
        "resumed_elfs": len(completed),
        "pending_elfs": len(pending),
        "candidate_families": len(candidate_families),
        "verified_source_keys": len(source_hashes),
        "cu_source_variant_universe": len(universe),
        "workers": args.workers,
        "checkpoint": str(checkpoint),
        "error_log": str(error_log),
        "report": str(report_path),
    }, indent=2), flush=True)

    started = time.monotonic()
    run_errors: dict[str, dict[str, Any]] = {}
    if pending:
        with (
            checkpoint.open("a", encoding="utf-8") as out,
            error_log.open("a", encoding="utf-8") as errors,
            multiprocessing.Pool(
                args.workers,
                initializer=init_worker,
                initargs=(label_meta, candidate_families, source_hashes, universe),
            ) as pool,
        ):
            for number, result in enumerate(
                pool.imap_unordered(evaluate_case_safe, pending, chunksize=1), 1
            ):
                if result["ok"]:
                    row = result["row"]
                    completed[row["case_id"]] = row
                    out.write(json.dumps(row, sort_keys=True) + "\n")
                    out.flush()
                else:
                    error = result["error"]
                    run_errors[error["case_id"]] = error
                    errors.write(json.dumps(error, sort_keys=True) + "\n")
                    errors.flush()
                    print(
                        f"ERROR {error['case_id']}: "
                        f"{error['error_type']}: {error['error']}", flush=True
                    )
                if number % 10 == 0 or number == len(pending):
                    print(
                        f"progress {len(completed)}/{len(tasks)} "
                        f"elapsed={time.monotonic() - started:.1f}s", flush=True
                    )

    rows = [completed[t["case_id"]] for t in tasks if t["case_id"] in completed]
    missing = [
        {
            **{k: t[k] for k in (
                "case_id", "compiler", "program", "optimization", "feature"
            )},
            "error": run_errors.get(t["case_id"]),
        }
        for t in tasks if t["case_id"] not in completed
    ]
    report = {
        "schema_version": 2,
        "run_signature": signature,
        "task": "CU discrimination across source-code variants",
        "identity": TASK_IDENTITY,
        "equivalence_rule": EQUIVALENCE_RULE,
        "unverified_rule": UNVERIFIED_RULE,
        "prediction_rule": PREDICTION_RULE,
        "configuration": fixed.PARAMS,
        "checkpoint_protocol": "selection-independent-per-case-v1",
        "source_audit_sha256": source_audit_sha256,
        "label_metadata_hash": label_metadata_hash,
        "complete": not missing,
        "evaluated_elfs": len(rows),
        "total_elfs": len(tasks),
        "missing_cases": missing,
        "candidate_families": len(candidate_families),
        "cu_source_variant_universe": len(universe),
        "source_audit_summary": audit_payload.get("summary", {}),
        **summarize(rows),
        "verified_source_only_metrics": summarize(
            rows, nested="verified_source_only"
        ),
        "artifacts": {
            "source_audit": str(args.source_audit),
            "checkpoint": str(checkpoint),
            "errors": str(error_log),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "complete": report["complete"],
        "evaluated_elfs": report["evaluated_elfs"],
        "missing": len(missing),
        "pooled": report["pooled"],
        "source_verification_coverage": report["source_verification_coverage"],
        "report": str(report_path),
    }, indent=2, sort_keys=True), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
