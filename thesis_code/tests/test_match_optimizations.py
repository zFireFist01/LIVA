from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


THESIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, THESIS_DIR.as_posix())

import match  # noqa: E402


class FakeBlock:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding


class MatchOptimizationTests(unittest.TestCase):
    @staticmethod
    def _window(*, coverage_mean: float, function_coverage: float):
        return match.BlockWindowResult(
            window_start=0,
            window_stop=1,
            block_start=0,
            block_stop=1,
            num_source_blocks=1,
            coverage_mean=coverage_mean,
            coverage_min=coverage_mean,
            coverage_ratio=1.0,
            assignment_quality=1.0,
            assignment_ratio=1.0,
            call_edge_ratio=1.0,
            call_edges_evaluated=0,
            call_edges_total=0,
            function_concentration=1.0,
            function_spread=1.0,
            function_coverage=function_coverage,
        )

    def test_pareto_frontier_retains_function_coverage_tradeoff(self) -> None:
        higher_score = self._window(
            coverage_mean=0.99, function_coverage=0.50
        )
        broader_cu = self._window(
            coverage_mean=0.98, function_coverage=1.00
        )

        frontier = match.pareto_block_windows((higher_score, broader_cu))

        self.assertEqual(frontier, (higher_score, broader_cu))

    def test_normalized_matrix_matches_previous_cosine_similarity(self) -> None:
        generator = np.random.default_rng(20260902)
        target_values = generator.normal(size=(7, 128)).astype(np.float32)
        source_values = generator.normal(size=(11, 128)).astype(np.float32)
        target_blocks = [FakeBlock(row) for row in target_values]
        source_blocks = [FakeBlock(row) for row in source_values]

        expected = cosine_similarity(target_values, source_values)
        actual = match.compute_blocks_similarity_matrix(
            target_blocks,
            source_blocks,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_prefilter_maps_only_the_best_rejected_window(self) -> None:
        target_functions = [object(), object()]
        source_functions = [object() for _ in range(5)]
        target_unit = SimpleNamespace(functions=target_functions)
        source_unit = SimpleNamespace(functions=source_functions)
        target_records = [
            {"function_index": 0, "block": object()},
            {"function_index": 1, "block": object()},
        ]
        source_records = [
            {"function_index": index, "block": object()}
            for index in range(5)
        ]
        source_spans = [(index, index + 1) for index in range(5)]
        target_prepared = match.PreparedBlockUnit(
            target_unit,
            5,
            target_records,
            [(0, 1), (1, 2)],
            np.empty((2, 4), dtype=np.float32),
        )
        source_prepared = match.PreparedBlockUnit(
            source_unit,
            5,
            source_records,
            source_spans,
            np.empty((5, 4), dtype=np.float32),
        )
        full_similarity = np.asarray(
            [
                [0.10, 0.20, 0.30, 0.40, 0.50],
                [0.20, 0.30, 0.40, 0.50, 0.60],
            ],
            dtype=np.float32,
        )
        expected_match = match.FunctionMatch(1, 4, 1.0, 0.60, 0.0)

        with (
            mock.patch.object(
                match,
                "prepare_block_unit",
                return_value=target_prepared,
            ),
            mock.patch.object(
                match,
                "normalized_similarity_matrix",
                return_value=full_similarity,
            ),
            mock.patch.object(
                match,
                "function_concentration_scores",
                return_value=(0.5, 0.5, {1: 4}, (expected_match,)),
            ) as mapping,
        ):
            result = match.evaluate_block_presence(
                source_unit=source_unit,
                target_unit=target_unit,
                block_threshold=0.90,
                min_assignment_quality=0.80,
                min_assignment_ratio=0.50,
                min_coverage_ratio=0.90,
                min_coverage_mean=0.90,
                locality_window_multiplier=1.0,
                locality_window_padding=0,
                min_block_instructions=5,
                prepared_source=source_prepared,
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.windows_total, 4)
        self.assertEqual(result.windows_evaluated, 0)
        self.assertEqual(result.windows_skipped, 4)
        self.assertEqual(result.selected_window_start, 3)
        self.assertEqual(result.selected_window_stop, 5)
        self.assertEqual(result.function_matches, (expected_match,))
        mapping.assert_called_once()
        np.testing.assert_array_equal(
            mapping.call_args.args[2],
            np.asarray([4, 4]),
        )


if __name__ == "__main__":
    unittest.main()
