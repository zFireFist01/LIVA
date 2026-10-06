from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


THESIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, THESIS_DIR.as_posix())

from experiments import offline_pipeline_ablation as ablation  # noqa: E402


class OfflinePipelineAblationTests(unittest.TestCase):
    def test_function_coverage_is_an_independent_replay_gate(self) -> None:
        window = {
            "coverage_mean": 1.0,
            "coverage_ratio": 1.0,
            "assignment_quality": 1.0,
            "assignment_ratio": 1.0,
            "call_edge_ratio": 1.0,
            "function_concentration": 1.0,
            "function_spread": 1.0,
            "function_coverage": 0.5,
        }
        params = {
            **ablation.replay.DECISION_DEFAULTS,
            "cu_min_function_coverage": 0.6,
        }
        self.assertFalse(ablation.replay.block_window_passes(window, params))
        params["cu_min_function_coverage"] = 0.0
        self.assertTrue(ablation.replay.block_window_passes(window, params))

    def test_replays_schema_three_batch_features(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            shard = root / "shard"
            batch = shard / "results/run"
            feature_dir = batch / "reports/current"
            dataset_dir = shard / "datasets/libseeker"
            ground_truth_dir = shard / "ground_truth/libseeker/demo"
            feature_dir.mkdir(parents=True)
            dataset_dir.mkdir(parents=True)
            ground_truth_dir.mkdir(parents=True)
            binary = dataset_dir / "demo"
            binary.write_bytes(b"\x7fELF")

            ground_truth = ground_truth_dir / "ground_truth.json"
            ground_truth.write_text(
                json.dumps({
                    "archives": [{
                        "archive": "libdemo.a",
                        "included_compilation_units": 1,
                    }]
                }),
                encoding="utf-8",
            )
            (dataset_dir / "manifest.json").write_text(
                json.dumps({
                    "records": [{
                        "binary": "datasets/libseeker/demo",
                        "ground_truth": "ground_truth/libseeker/demo/ground_truth.json",
                        "program": "demo",
                        "compiler": "gcc-11-11.5.0",
                        "program_optimization": "O2",
                    }]
                }),
                encoding="utf-8",
            )
            matrix = root / "library_matrix.tsv"
            matrix.write_text(
                "archive\tpackage\trole\ttoolchain\toptimization\t"
                "project_candidates\tstatus\treason\tpath\n"
                "libdemo.a\tdemo\tcurrent\tgcc-13-13.3.0\tO2\t"
                "demo-1.0\tselected\t\tgcc/demo/libdemo.a\n",
                encoding="utf-8",
            )
            configuration = {
                **ablation.replay.DECISION_DEFAULTS,
                "library_score_aggregator": "top3_noisy_or",
            }
            records = [
                {
                    "schema_version": 3,
                    "type": "source_call_targets",
                    "feature_mode": "replay_complete",
                    "binary_path": binary.as_posix(),
                    "matching_configuration": configuration,
                    "functions": [],
                    "calls": [],
                },
                {
                    "schema_version": 3,
                    "type": "block_cu",
                    "record_detail": "replay",
                    "binary_path": binary.as_posix(),
                    "library": "libdemo.a",
                    "name": "member.o",
                    "target_cu_index": 0,
                    "target_function_count": 2,
                    "rodata": 0.0,
                    "rodata_has_rodata": False,
                    "rodata_strings": 0,
                    "rodata_ngrams": 0,
                    "rodata_bytes": 0,
                    "inter_cu_calls": [],
                    "windows": [{
                        "coverage_mean": 1.0,
                        "coverage_ratio": 1.0,
                        "assignment_quality": 0.99,
                        "assignment_ratio": 1.0,
                        "call_edge_ratio": 1.0,
                        "function_concentration": 1.0,
                        "function_spread": 1.0,
                        "function_coverage": 1.0,
                        "function_matches": [
                            {
                                "target_function_index": 0,
                                "source_function_index": 0,
                                "dominant_ratio": 1.0,
                                "coverage_mean": 1.0,
                                "coverage_ratio": 1.0,
                            },
                            {
                                "target_function_index": 1,
                                "source_function_index": 1,
                                "dominant_ratio": 1.0,
                                "coverage_mean": 1.0,
                                "coverage_ratio": 1.0,
                            },
                        ],
                    }],
                },
            ]
            with gzip.open(
                feature_dir / "demo.features.jsonl.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

            output = root / "ablation"
            argv = [
                "offline_pipeline_ablation.py",
                "--batch-output-dir", batch.as_posix(),
                "--library-matrix", matrix.as_posix(),
                "--configuration", "FULL",
                "--output-dir", output.as_posix(),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(ablation.main(), 0)

            result = json.loads(
                (output / "ablation_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["elf_count"], 1)
            self.assertEqual(result["summary"][0]["tp"], 1)
            self.assertEqual(result["summary"][0]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
