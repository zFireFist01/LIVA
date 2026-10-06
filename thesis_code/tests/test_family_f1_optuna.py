from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "tune_family_f1_optuna.py"
)
SPEC = importlib.util.spec_from_file_location("tune_family_f1_optuna", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
family = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(family)


def passing_record() -> dict:
    return {
        "target_cu_index": 0,
        "windows": [
            {
                "coverage_mean": 1.0,
                "coverage_ratio": 1.0,
                "assignment_quality": 1.0,
                "assignment_ratio": 1.0,
                "call_edge_ratio": 1.0,
                "function_concentration": 1.0,
                "function_spread": 1.0,
                "function_coverage": 1.0,
                "function_matches": [],
            }
        ],
        "inter_cu_calls": [],
        "rodata": 0.0,
        "rodata_has_rodata": False,
        "rodata_strings": 0,
        "rodata_ngrams": 0,
        "rodata_bytes": 0,
    }


def test_family_is_archive_not_source_project() -> None:
    assert family.family_from_label("libssl.a.0123456789abcdef") == "libssl"
    assert family.family_from_label("libcrypto.a.fedcba9876543210") == "libcrypto"


def test_versioned_linker_archive_maps_to_searchable_family() -> None:
    candidates = {"libm", "libssl", "libcrypto"}
    assert family.normalize_archive_family("/tmp/libm-2.39.a", candidates) == "libm"
    assert family.normalize_archive_family("/tmp/libssl.a", candidates) == "libssl"
    assert family.normalize_archive_family("/tmp/libunknown.a", candidates) is None


def test_multiple_builds_collapse_to_one_family_prediction() -> None:
    params = dict(family.engine.DECISION_DEFAULTS)
    params["library_score_aggregator"] = "max"
    params["library_min_score"] = 0.975
    record = passing_record()
    case = {
        "libraries": {
            "libssl.a.0123456789abcdef": [record],
            "libssl.a.fedcba9876543210": [record],
            "libcrypto.a.1111111111111111": [record],
        },
        "family_by_label": {
            "libssl.a.0123456789abcdef": "libssl",
            "libssl.a.fedcba9876543210": "libssl",
            "libcrypto.a.1111111111111111": "libcrypto",
        },
        "source_call_targets": {},
        "expected_families": {"libssl", "libz"},
    }
    counts = family.family_counts([case], params)
    assert counts == Counter(tp=1, fp=1, fn=1, evaluated_elfs=1)
