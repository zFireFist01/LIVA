from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np


THESIS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = THESIS_DIR.parent
sys.path.insert(0, THESIS_DIR.as_posix())

from main import (  # noqa: E402
    block_cu_feature_record,
    release_library_units,
    should_evaluate_rodata,
)
from analysis_cache import (  # noqa: E402
    AnalysisEmbeddingCache,
    CacheOnlyMissError,
    CachedCodeUnitLoader,
)
from asm import Block, CodeUnit, Function  # noqa: E402
from numpy_cache import read_entry, write_entry  # noqa: E402
from run_libseeker_batch import (  # noqa: E402
    automatic_job_count,
    binary_label,
    parse_args as parse_batch_args,
    parse_jobs,
    write_result_bundle,
)


class FakeFunction:
    def __init__(
        self, name: str, address: int, *, calls: set[int] | None = None
    ) -> None:
        self.name = name
        self.address = address
        self.size = 16
        self.resolved_call_targets = calls or set()

    def get_num_blocks(self) -> int:
        return 2

    def get_num_instructions(self) -> int:
        return 8


class FakeUnit:
    def __init__(self, name: str, functions: list[FakeFunction]) -> None:
        self.name = name
        self.functions = functions
        self.rodata_strings: list[str] = []

    def get_num_functions(self) -> int:
        return len(self.functions)


class ResultRecordTests(unittest.TestCase):
    def test_auto_jobs_respect_cpu_and_available_memory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cache = Path(raw_directory)
            (cache / "shard_inventory.json").write_text(
                json.dumps(
                    {
                        "valid": True,
                        "errors": [],
                        "full_library_catalog": True,
                        "elf_records": 876,
                        "experiment_cache_records": 876,
                        "library_archives_cached": 4472,
                        "selected_library_archives_total": 4472,
                    }
                ),
                encoding="utf-8",
            )

            jobs, reason = automatic_job_count(
                876,
                cache,
                cpu_count=16,
                memory_bytes=24 * 1024**3,
            )

            self.assertEqual(jobs, 12)
            self.assertIn("validated cache", reason)

    def test_auto_jobs_fall_back_to_one_without_validated_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            jobs, reason = automatic_job_count(
                876,
                Path(raw_directory),
                cpu_count=32,
                memory_bytes=128 * 1024**3,
            )

            self.assertEqual(jobs, 1)
            self.assertIn("incomplete", reason)

    def test_jobs_argument_accepts_auto_or_positive_integer(self) -> None:
        self.assertEqual(parse_jobs("auto"), "auto")
        self.assertEqual(parse_jobs("7"), 7)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_jobs("0")

    def test_batch_accepts_cache_only_mode(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["run_libseeker_batch.py", "--analysis-cache-only"],
        ):
            args = parse_batch_args()
        self.assertTrue(args.analysis_cache_only)

    def test_cache_only_uses_packaged_identity_and_never_calls_radare2(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            model = root / "model.bin"
            model.write_bytes(b"model")
            common_identity = {
                "cache_format": 1,
                "asm_normalization": "v2",
                "palmtree_pooling": "masked_mean",
                "palmtree_model_sha256": "a" * 64,
                "palmtree_vocab_sha256": "a" * 64,
                "implementation_sha256": "implementation",
                "radare2_version": "radare2 creator version",
            }
            (root / "experiment_elf_provenance.jsonl").write_text(
                json.dumps(
                    {
                        "cache_identity": {
                            **common_identity,
                            "binary_sha256": "b" * 64,
                            "binary_size": 100,
                            "unit_type": "ELF",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            binary = root / "program"
            binary.write_bytes(b"binary")

            with (
                mock.patch("analysis_cache.sha256_file", return_value="a" * 64),
                mock.patch(
                    "analysis_cache.implementation_signature",
                    return_value="implementation",
                ),
                mock.patch(
                    "analysis_cache.radare2_version",
                    side_effect=AssertionError("radare2 must not be queried"),
                ),
                mock.patch(
                    "analysis_cache.parse_r2_file",
                    side_effect=AssertionError("analysis must not run"),
                ),
            ):
                loader = CachedCodeUnitLoader(
                    model,
                    device="auto",
                    pooling="masked_mean",
                    asm_normalization="v2",
                    cache_dir=root,
                    cache_only=True,
                )
                self.assertEqual(loader._common_identity, common_identity)
                self.assertFalse(loader.write_cache)
                with self.assertRaisesRegex(CacheOnlyMissError, "cache-only miss"):
                    loader.load(binary, unit_type="ELF")
                archive = root / "libdemo.a"
                archive.write_bytes(b"archive")
                with self.assertRaisesRegex(
                    CacheOnlyMissError, "cache-only archive index miss"
                ):
                    loader.load_archive(archive)
                self.assertFalse((root / "locks").exists())

    def test_archive_index_is_enriched_without_loading_cu_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            cache = AnalysisEmbeddingCache(root)
            common_identity = {"cache_format": 1}
            identity = {
                **common_identity,
                "binary_sha256": "a" * 64,
                "binary_size": 123,
                "unit_type": "CU",
            }
            entry = cache.entry_path(identity)
            entry.mkdir(parents=True)
            (entry / "metadata.json").write_text(
                json.dumps({"identity": identity, "counts": {"functions": 1}}),
                encoding="utf-8",
            )
            index_path = root / "archive.json"
            index = {
                "schema": 1,
                "members": [{
                    "name": "single.o",
                    "occurrence": 1,
                    "binary_sha256": identity["binary_sha256"],
                    "binary_size": identity["binary_size"],
                    "cache_key": cache.key(identity),
                }],
            }
            index_path.write_text(json.dumps(index), encoding="utf-8")
            loader = object.__new__(CachedCodeUnitLoader)
            loader.cache = cache
            loader.cache_only = False
            loader._common_identity = common_identity

            upgraded = loader.ensure_archive_function_counts(index_path, index)

            self.assertEqual(upgraded["schema"], 2)
            self.assertEqual(upgraded["members"][0]["function_count"], 1)
            persisted = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["members"][0]["function_count"], 1)

    def test_archive_loader_skips_single_function_cu_before_cache_load(self) -> None:
        class FakeCache:
            def __init__(self) -> None:
                self.loaded_names: list[str] = []

            def load(self, identity, synthetic_path):
                self.loaded_names.append(synthetic_path.name)
                return FakeUnit(
                    synthetic_path.name,
                    [FakeFunction("first", 0x10), FakeFunction("second", 0x20)],
                )

        index = {
            "members": [
                {
                    "name": "single.o",
                    "occurrence": 1,
                    "binary_sha256": "a" * 64,
                    "binary_size": 10,
                    "function_count": 1,
                },
                {
                    "name": "multi.o",
                    "occurrence": 1,
                    "binary_sha256": "b" * 64,
                    "binary_size": 20,
                    "function_count": 2,
                },
            ]
        }
        loader = object.__new__(CachedCodeUnitLoader)
        loader.cache = FakeCache()
        loader.cache_only = False
        loader._common_identity = {}
        loader._hits = 0
        with tempfile.TemporaryDirectory() as raw_directory:
            index_path = Path(raw_directory) / "archive.json"
            with (
                mock.patch.object(
                    loader,
                    "inspect_archive_index",
                    return_value=(index_path, index),
                ),
                mock.patch.object(
                    loader,
                    "ensure_archive_function_counts",
                    return_value=index,
                ),
            ):
                units = loader.load_archive(
                    Path(raw_directory) / "libdemo.a",
                    min_functions=2,
                )

        self.assertEqual([unit.name for unit in units], ["multi.o"])
        self.assertEqual(loader.cache.loaded_names, ["multi.o"])

    def test_numpy_cache_reader_keeps_only_embedding_mmap_open(self) -> None:
        fd_root = Path("/proc/self/fd")
        if not fd_root.is_dir():
            self.skipTest("file-descriptor accounting requires procfs")

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            identity = {
                "cache_format": 1,
                "binary_sha256": "a" * 64,
                "binary_size": 123,
                "unit_type": "CU",
            }
            block = Block(0x1000, ["ret"], b"\xc3")
            block.embedding = np.ones(128, dtype=np.float32)
            original = CodeUnit(
                "member.o",
                "/tmp/member.o",
                functions=[Function("function", 0x1000, blocks=[block])],
                unit_type=CodeUnit.TYPE_CU,
                rodata_bytes=b"cached string\0",
                rodata_strings=["cached string"],
                rodata_section_count=1,
            )
            entry = root / "entry.numpy"
            write_entry(entry, identity, original)

            before = len(list(fd_root.iterdir()))
            loaded = [read_entry(entry, expected_identity=identity)[1] for _ in range(64)]
            after = len(list(fd_root.iterdir()))

            self.assertTrue(all(unit.get_num_functions() == 1 for unit in loaded))
            self.assertLessEqual(after - before, 64 * 2)

    def test_library_units_are_released_before_loading_the_next_archive(self) -> None:
        units = [FakeUnit(f"member-{index}", []) for index in range(32)]
        target_units = list(units)
        block_results = [(unit, object()) for unit in units]
        references = [weakref.ref(unit) for unit in units]

        release_library_units(block_results, target_units, units)

        self.assertEqual(block_results, [])
        self.assertEqual(target_units, [])
        self.assertEqual(units, [])
        self.assertTrue(all(reference() is None for reference in references))

    def test_rodata_runs_only_after_block_match_passes(self) -> None:
        self.assertFalse(
            should_evaluate_rodata(SimpleNamespace(passed=False), False)
        )
        self.assertTrue(
            should_evaluate_rodata(SimpleNamespace(passed=True), False)
        )
        self.assertFalse(
            should_evaluate_rodata(SimpleNamespace(passed=True), True)
        )

    def test_cu_record_names_both_functions_and_keeps_dominant_ratio(self) -> None:
        source = FakeUnit(
            "binary",
            [
                FakeFunction("elf_caller", 0x1000, calls={0x2000}),
                FakeFunction("elf_callee", 0x2000),
            ],
        )
        reference = FakeUnit("archive_member.o", [FakeFunction("ref_fn", 0x10)])
        function_match = SimpleNamespace(
            target_function_index=0,
            source_function_index=0,
            dominant_ratio=0.75,
            coverage_mean=0.90,
            coverage_ratio=0.80,
        )
        block_result = SimpleNamespace(
            windows=(),
            passed=True,
            score=0.91,
            coverage_mean=0.90,
            coverage_min=0.70,
            coverage_ratio=0.80,
            assignment_quality=0.88,
            assignment_ratio=0.77,
            call_edge_ratio=1.0,
            call_edges_evaluated=1,
            call_edges_total=1,
            function_concentration=0.75,
            function_spread=0.50,
            selected_window_start=0,
            selected_window_stop=2,
            selected_block_start=0,
            selected_block_stop=3,
            function_matches=(function_match,),
        )
        rodata = SimpleNamespace(
            score=0.0,
            has_rodata=False,
            target_strings=0,
            target_ngrams=0,
            target_bytes=0,
        )
        record = block_cu_feature_record(
            "/tmp/program",
            "libdemo.a",
            source,
            0,
            reference,
            block_result,
            (),
            rodata,
            "PASS",
            "v2",
            "masked_mean",
        )
        mapping = record["selected_match"]["function_matches"][0]
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["cu_status"], "PASS")
        self.assertEqual(mapping["reference_function"]["name"], "ref_fn")
        self.assertEqual(mapping["elf_function"]["name"], "elf_caller")
        self.assertEqual(mapping["dominant_ratio"], 0.75)
        self.assertEqual(mapping["source_call_targets"], [1])

    def test_rejected_cu_record_is_compact_but_keeps_best_function_hint(self) -> None:
        source = FakeUnit(
            "binary",
            [
                FakeFunction("elf_first", 0x1000),
                FakeFunction("elf_best", 0x2000),
            ],
        )
        reference = FakeUnit(
            "archive_member.o",
            [
                FakeFunction("ref_first", 0x10),
                FakeFunction("ref_best", 0x20),
            ],
        )
        function_matches = (
            SimpleNamespace(
                target_function_index=0,
                source_function_index=0,
                dominant_ratio=0.50,
                coverage_mean=0.80,
                coverage_ratio=0.70,
            ),
            SimpleNamespace(
                target_function_index=1,
                source_function_index=1,
                dominant_ratio=0.90,
                coverage_mean=0.85,
                coverage_ratio=0.75,
            ),
        )
        block_result = SimpleNamespace(
            passed=False,
            score=0.85,
            coverage_mean=0.85,
            coverage_min=0.60,
            coverage_ratio=0.75,
            assignment_quality=0.0,
            assignment_ratio=0.0,
            call_edge_ratio=0.0,
            call_edges_evaluated=0,
            call_edges_total=1,
            function_concentration=0.0,
            function_spread=0.0,
            selected_window_start=3,
            selected_window_stop=8,
            selected_block_start=10,
            selected_block_stop=30,
            windows_total=20,
            windows_evaluated=2,
            windows_skipped=18,
            function_matches=function_matches,
        )
        rodata = SimpleNamespace(score=0.0)

        record = block_cu_feature_record(
            "/tmp/program",
            "libdemo.a",
            source,
            4,
            reference,
            block_result,
            (),
            rodata,
            "DROP",
            "v2",
            "masked_mean",
            rodata_status="skipped",
        )

        self.assertEqual(record["record_detail"], "compact")
        self.assertEqual(record["rejection_reason"], "block_gates")
        self.assertEqual(record["rodata_status"], "skipped")
        self.assertEqual(record["target_function_count"], 2)
        self.assertEqual(
            record["best_function_match"]["elf_function"]["name"],
            "elf_best",
        )
        self.assertEqual(
            record["best_function_match"]["dominant_ratio"], 0.90
        )
        for omitted in (
            "target_functions",
            "windows",
            "inter_cu_calls",
            "selected_match",
        ):
            self.assertNotIn(omitted, record)

    def test_rejected_cu_replay_record_keeps_offline_ablation_evidence(self) -> None:
        source = FakeUnit(
            "binary",
            [FakeFunction("elf_fn", 0x1000), FakeFunction("other", 0x2000)],
        )
        reference = FakeUnit(
            "member.o",
            [FakeFunction("ref_one", 0x10), FakeFunction("ref_two", 0x20)],
        )
        mapping = SimpleNamespace(
            target_function_index=0,
            source_function_index=0,
            dominant_ratio=1.0,
            coverage_mean=0.99,
            coverage_ratio=1.0,
        )
        window = SimpleNamespace(
            window_start=0,
            window_stop=2,
            block_start=0,
            block_stop=4,
            coverage_mean=0.99,
            coverage_min=0.98,
            coverage_ratio=1.0,
            assignment_quality=0.40,
            assignment_ratio=0.25,
            call_edge_ratio=1.0,
            call_edges_evaluated=0,
            call_edges_total=0,
            function_concentration=1.0,
            function_spread=1.0,
            function_coverage=0.5,
            function_matches=(mapping,),
        )
        block_result = SimpleNamespace(
            windows=(window,),
            passed=False,
            score=0.99,
            coverage_mean=0.99,
            coverage_min=0.98,
            coverage_ratio=1.0,
            assignment_quality=0.40,
            assignment_ratio=0.25,
            call_edge_ratio=1.0,
            call_edges_evaluated=0,
            call_edges_total=0,
            function_concentration=1.0,
            function_spread=1.0,
            function_coverage=0.5,
            windows_total=4,
            windows_evaluated=1,
            windows_skipped=3,
            function_matches=(mapping,),
        )
        rodata = SimpleNamespace(
            score=0.75,
            has_rodata=True,
            target_strings=4,
            target_ngrams=12,
            target_bytes=256,
        )

        record = block_cu_feature_record(
            "/tmp/program",
            "libdemo.a",
            source,
            0,
            reference,
            block_result,
            (),
            rodata,
            "DROP",
            "v2",
            "masked_mean",
            rodata_status="skipped",
            offline_ablation_features=True,
        )

        self.assertEqual(record["schema_version"], 3)
        self.assertEqual(record["record_detail"], "replay")
        self.assertEqual(record["windows"][0]["function_coverage"], 0.5)
        self.assertEqual(
            record["windows"][0]["function_matches"],
            [{
                "target_function_index": 0,
                "source_function_index": 0,
                "dominant_ratio": 1.0,
                "coverage_mean": 0.99,
                "coverage_ratio": 1.0,
            }],
        )
        self.assertEqual(record["target_functions"][0]["name"], "ref_one")
        self.assertEqual(record["rodata_bytes"], 256)
        self.assertTrue(record["rodata_measured"])
        self.assertIn("inter_cu_calls", record)

    @staticmethod
    def _descriptor(index: int, name: str) -> dict:
        return {
            "index": index,
            "name": name,
            "address": 0x1000 + index * 0x100,
            "size": 16,
            "blocks": 2,
            "instructions": 8,
        }

    def _write_shard(self, directory: Path, compiler: str, number: int) -> Path:
        result = directory / f"shard-{number}.jsonl.gz"
        elf_id = hashlib.sha256(f"elf-{number}".encode()).hexdigest()
        elf_function = self._descriptor(0, "elf_fn")
        reference_function = self._descriptor(0, "ref_fn")
        payload = [
            {
                "type": "run",
                "schema_version": 1,
                "pipelines": ["current"],
                "elf_count": 1,
                "library_count": 1,
                "expected_library_matches": 1,
                "dataset_manifest": f"/pc{number}/manifest.json",
                "dataset_manifest_sha256": str(number) * 64,
                "library_matrix": "/shared/library_matrix.tsv",
                "library_matrix_sha256": "a" * 64,
                "matching_configuration": {"block_threshold": 0.9},
                "accepted_cu_details": True,
                "rejected_cu_details": False,
            },
            {
                "type": "library",
                "library_id": "libdemo.a",
                "archive": "libdemo.a",
                "archive_sha256": "b" * 64,
                "archive_size": 100,
                "dataset_path": "Dataset/builds/libraries/libdemo.a",
            },
            {
                "type": "elf",
                "elf_id": elf_id,
                "binary_sha256": "c" * 64,
                "binary_size": 1000,
                "program": f"program-{number}",
                "compiler": compiler,
                "compiler_command": compiler,
                "program_optimization": "O0",
            },
            {
                "type": "library_match",
                "elf_id": elf_id,
                "library_id": "libdemo.a",
                "pipeline": "current",
                "status": "YES",
                "score": "99.0",
                "matched_cu": "1",
            },
            {
                "type": "elf_function_catalog",
                "schema_version": 2,
                "elf_id": elf_id,
                "functions": [elf_function],
                "calls": [],
            },
            {
                "type": "cu_match",
                "schema_version": 2,
                "elf_id": elf_id,
                "library_id": "libdemo.a",
                "name": "member.o",
                "target_cu_index": 0,
                "cu_status": "PASS",
                "target_functions": [reference_function],
                "selected_match": {
                    "passed": True,
                    "function_matches": [
                        {
                            "target_function_index": 0,
                            "source_function_index": 0,
                            "reference_function": reference_function,
                            "elf_function": elf_function,
                            "dominant_ratio": 0.75,
                            "coverage_mean": 0.90,
                            "coverage_ratio": 0.80,
                            "source_call_targets": [],
                        }
                    ],
                },
            },
        ]
        with gzip.open(result, "wt", encoding="utf-8") as stream:
            for record in payload:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        digest = hashlib.sha256(result.read_bytes()).hexdigest()
        sidecar = {
            "schema_version": 1,
            "valid": True,
            "result_log_sha256": digest,
            "result_log_size": result.stat().st_size,
            "elf_records": 1,
            "library_records": 1,
            "library_matches": 1,
            "cu_matches": 1,
            "accepted_cu_matches": 1,
            "function_matches": 1,
            "failures": 0,
        }
        result.with_name(f"shard-{number}.summary.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        return result

    def test_result_only_merge_streams_four_valid_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            inputs = [
                self._write_shard(directory, compiler, index)
                for index, compiler in enumerate(
                    ("gcc-11", "gcc-13", "clang-14", "clang-18")
                )
            ]
            output = directory / "merged.jsonl.gz"
            command = [
                sys.executable,
                (REPO_DIR / "Dataset/scripts/merge_libseeker_shards.py").as_posix(),
                "--allow-incomplete",
                "--output",
                output.as_posix(),
            ]
            for path in inputs:
                command.extend(("--shard-result", path.as_posix()))
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads(
                output.with_name("merged.summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["aggregation"], "results_only")
            self.assertEqual(summary["elf_records"], 4)
            self.assertEqual(summary["library_matches"], 4)
            self.assertEqual(summary["function_matches"], 4)

    def test_batch_bundle_keeps_accepted_cu_details(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            binary = root / "datasets/libseeker/binaries/program/gcc-11/O0/program"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFsynthetic")
            binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = root / "datasets/libseeker/manifest.json"
            manifest.write_text(
                json.dumps({
                    "records": [{
                        "program": "program",
                        "source_project": "demo",
                        "compiler": "gcc-11-11.5.0",
                        "compiler_command": "gcc-11",
                        "program_optimization": "O0",
                        "binary_sha256": binary_sha,
                        "binary": binary.relative_to(root).as_posix(),
                        "ground_truth": "",
                        "ground_truth_complete": True,
                    }]
                }),
                encoding="utf-8",
            )
            library = root / "libraries/libdemo.a"
            library.parent.mkdir()
            library.write_bytes(b"!<arch>\n")
            matrix = root / "library_matrix.tsv"
            matrix.write_text("fixture\n", encoding="utf-8")
            output_dir = root / "results"
            report_dir = output_dir / "reports/current"
            raw_dir = output_dir / "raw/current"
            report_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            stem = binary_label(binary, binary.parents[3])
            report = report_dir / f"{stem}.report.txt"
            report.write_text(
                "Assembly normalization: v2\n"
                "PalmTree pooling: masked_mean\n"
                "Library score aggregator: top3_noisy_or\n"
                "Library minimum score: 0.975\n"
                "YES     | library=libdemo.a | score=99.0% | "
                "base_score=98.0% | block_best_any=97.0% | "
                "block_best_matched=97.0% | matched_cu=1/1 | "
                "matched_functions=1 | time=00:00:01\n",
                encoding="utf-8",
            )
            (raw_dir / f"{stem}.raw.log").write_text(
                "Done processing in 00:00:02\n", encoding="utf-8"
            )
            elf_function = self._descriptor(0, "elf_fn")
            reference_function = self._descriptor(0, "ref_fn")
            feature_records = [
                {
                    "schema_version": 3,
                    "type": "source_call_targets",
                    "feature_mode": "replay_complete",
                    "binary_path": binary.as_posix(),
                    "functions": [elf_function],
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
                    "cu_status": "PASS",
                    "target_functions": [reference_function],
                    "selected_match": {
                        "passed": True,
                        "function_matches": [{
                            "target_function_index": 0,
                            "source_function_index": 0,
                            "dominant_ratio": 0.75,
                            "coverage_mean": 0.9,
                            "coverage_ratio": 0.8,
                        }],
                    },
                    "windows": [{"not": "portable"}],
                },
            ]
            with gzip.open(
                report_dir / f"{stem}.features.jsonl.gz", "wt", encoding="utf-8"
            ) as stream:
                for record in feature_records:
                    stream.write(json.dumps(record) + "\n")

            with mock.patch.object(sys, "argv", ["run_libseeker_batch.py"]):
                args = parse_batch_args()
            args.dataset_dir = binary.parents[3]
            args.dataset_manifest = manifest
            args.output_dir = output_dir
            args.result_log = output_dir / "results.jsonl.gz"
            args.library_matrix = matrix
            args.library_root = library.parent
            args.library_metadata = {
                library.as_posix(): {
                    "package": "demo",
                    "dataset_path": "Dataset/builds/libraries/libdemo.a",
                }
            }
            args.include_rejected_cu = False
            with redirect_stdout(io.StringIO()):
                summary = write_result_bundle(
                    args=args,
                    elfs=[binary],
                    libraries=[library],
                    library_labels={library.as_posix(): "libdemo.a"},
                    pipelines=[("current", THESIS_DIR)],
                )
            self.assertEqual(summary["accepted_cu_matches"], 1)
            with gzip.open(args.result_log, "rt", encoding="utf-8") as stream:
                output_records = [json.loads(line) for line in stream]
            cu = next(record for record in output_records if record["type"] == "cu_match")
            self.assertNotIn("windows", cu)
            self.assertEqual(
                cu["selected_match"]["function_matches"][0]["dominant_ratio"],
                0.75,
            )
            self.assertEqual(
                cu["selected_match"]["function_matches"][0]
                ["reference_function"]["name"],
                "ref_fn",
            )
            self.assertEqual(
                cu["selected_match"]["function_matches"][0]
                ["elf_function"]["name"],
                "elf_fn",
            )


if __name__ == "__main__":
    unittest.main()
