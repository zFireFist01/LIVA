"""Protocol checks that do not require the optional binary-analysis stack."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


def load_search_module():
    root = Path(__file__).resolve().parents[1]
    stubs = {
        "analysis_cache": types.SimpleNamespace(
            CachedCodeUnitLoader=object,
            packaged_common_identity=lambda *args: None,
        ),
        "asm": types.SimpleNamespace(CodeUnit=object),
        "match": types.SimpleNamespace(
            LIBRARY_SCORE_AGGREGATORS=("max", "top3_mean", "top3_noisy_or"),
            aggregate_library_score=lambda *args: 0.0,
        ),
        "run_libseeker_batch": types.SimpleNamespace(
            select_libs=lambda *args: [],
            stable_library_labels=lambda *args: {},
        ),
    }
    spec = importlib.util.spec_from_file_location(
        "thesis_optuna_protocol_under_test", root / "optuna_threshold_search.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class ThesisOptunaProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = load_search_module()
        cls.search.library_match_evidence = lambda records, *_: (
            [records[0]] if records and records[0].get("pass") else [],
            1.0 if records and records[0].get("pass") else 0.0,
        )
        cls.search.cu_counts_for_prediction = lambda *args: (
            {"cu_tp": 0, "cu_fp": 0, "cu_fn": 0}, False
        )

    def test_archive_family_is_not_the_source_project(self):
        family = self.search.archive_family
        self.assertEqual(family("libc.a.0123456789abcdef"), "libc")
        self.assertEqual(family("libm.a.0123456789abcdef"), "libm")
        self.assertEqual(family("libpcre2-8.a"), "libpcre2-8")

    def test_thesis_mode_requires_separate_development_ground_truth(self):
        with patch.object(sys, "argv", ["search", "--thesis-protocol"]):
            with self.assertRaisesRegex(ValueError, "separate 1200-ELF development"):
                self.search.main()

    def test_reported_thesis_configuration_is_in_search_grids(self):
        structural = {
            "block_threshold": 0.85,
            "block_locality_window_multiplier": 2.5,
            "block_locality_window_padding": 6,
            "block_min_instructions": 5,
        }
        decisions = {
            "library_score_aggregator": "top3_mean",
            "library_min_score": 0.905,
            "block_coverage_mean_threshold": 0.975,
            "block_min_coverage_ratio": 0.75,
            "block_assignment_quality_threshold": 0.999,
            "block_min_assignment_ratio": 0.90,
            "block_min_call_edge_ratio": 0.30,
            "block_min_function_concentration": 0.75,
            "block_min_function_spread": 0.30,
            "cu_min_function_coverage": 0.60,
            "rodata_min_bytes": 512,
            "rodata_min_strings": 16,
            "rodata_min_ngrams": 32,
            "rodata_penalty_threshold": 0.35,
            "rodata_confirm_threshold": 0.90,
            "rodata_bonus_weight": 0.30,
            "rodata_string_weight": 0.70,
            "rodata_byte_only_weight": 0.50,
            "cross_cu_call_bonus_weight": 0.20,
            "cross_cu_call_penalty_weight": 0.00,
            "cross_cu_call_saturation_edges": 8,
        }
        for name, value in structural.items():
            self.assertIn(value, self.search.STRUCTURAL_GRIDS[name], name)
        for name, value in decisions.items():
            self.assertIn(value, self.search.DECISION_GRIDS[name], name)

    def test_two_builds_of_one_family_produce_one_classification(self):
        case = {
            "binary": Path("/tmp/thesis-protocol-elf"),
            "variant": "elf-o2", "program": "elf", "compiler": "gcc",
            "elf_optimization": "O2", "expected": {"libz"},
        }
        libraries = {
            "libz.a.1111111111111111": [{"pass": True}],
            "libz.a.2222222222222222": [{"pass": True}],
            "libssl.a.3333333333333333": [{"pass": True}],
        }
        metrics, _ = self.search.evaluate(
            {"library_min_score": 0.5},
            {"cases": {str(case["binary"]): libraries}},
            [case], list(libraries),
        )
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0)
        )
        label_counts = self.search.split_library_label_counts(
            [case], list(libraries)
        )
        self.assertEqual(label_counts, {"positive": 1, "negative": 1, "total": 2})
        rows = self.search.metrics_by_library(
            {"library_min_score": 0.5},
            {"cases": {str(case["binary"]): libraries}},
            [case], list(libraries),
        )
        self.assertEqual({row["family"] for row in rows}, {"libz", "libssl"})

    def test_optuna_uses_mean_of_fold_f1(self):
        passing = [([{"pass": True}], {})]
        failing = [([{"pass": False}], {})]
        pairs = {
            "profile": [
                [(passing, True)],
                [(failing, True), (passing, False), (passing, False),
                 (passing, False)],
            ]
        }
        value = self.search.evaluate_optuna_trial(
            "profile", {"library_min_score": 0.5}, pairs
        )
        self.assertEqual(value, 0.5)

    def test_screen_has_two_elf_per_program_and_balanced_builds(self):
        cases = [
            {
                "program": f"program-{program:02d}",
                "compiler": compiler.split("-")[0],
                "compiler_config": compiler,
                "elf_optimization": optimization,
                "variant": f"{program}-{compiler}-{optimization}",
            }
            for program in range(75)
            for compiler in self.search.THESIS_COMPILERS
            for optimization in self.search.THESIS_OPTIMIZATIONS
        ]
        selected = self.search.select_thesis_screen_cases(cases, 20260615)
        self.assertEqual(len(selected), 150)
        self.assertEqual(
            set(Counter(case["program"] for case in selected).values()), {2}
        )
        counts = Counter(
            (case["compiler_config"], case["elf_optimization"])
            for case in selected
        )
        self.assertEqual(set(counts.values()), {9, 10})

    def test_screen_requires_version_compiler_matrix_and_720_builds(self):
        archives = [
            {
                "name": f"lib{family:02d}.a.{index:016x}",
                "role": role,
                "compiler": compiler,
                "optimization": optimization,
            }
            for family in range(60)
            for index, (role, compiler, optimization) in enumerate(
                (role, compiler, optimization)
                for role in self.search.THESIS_VERSION_ROLES
                for compiler in self.search.THESIS_COMPILERS
                for optimization in self.search.THESIS_OPTIMIZATIONS
            )
        ]
        selected = self.search.select_thesis_screen_archives(
            archives, [{"expected": {"lib00"}}]
        )
        self.assertEqual(len(selected), 720)
        counts = Counter(self.search.archive_family(row["name"]) for row in selected)
        self.assertEqual(set(counts.values()), {12})
        for family in counts:
            optimizations = Counter(
                row["optimization"] for row in selected
                if self.search.archive_family(row["name"]) == family
            )
            self.assertEqual(set(optimizations.values()), {3})
        incomplete = [
            row for row in archives
            if not (
                row["role"] == "major-alternative"
                and row["compiler"] == "gcc-11-11.5.0"
            )
        ]
        with self.assertRaisesRegex(ValueError, "only 0 eligible families"):
            self.search.select_thesis_screen_archives(
                incomplete, [{"expected": {"lib00"}}]
            )


if __name__ == "__main__":
    unittest.main()
