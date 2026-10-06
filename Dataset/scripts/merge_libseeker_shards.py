#!/usr/bin/env python3
"""Aggregate four LibSeeker result logs without copying datasets or caches."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
COMPILERS = ("gcc-11", "gcc-13", "clang-14", "clang-18")
OPTIMIZATIONS = {"O0", "O2", "O3", "Os"}
EXPECTED_SHARDS = 4
EXPECTED_PER_SHARD = 876
EXPECTED_PROGRAMS = 219
EXPECTED_TOTAL = 3504


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        help=(
            "Shard artifact root; repeat four times. The result is read from "
            "<root>/results/libseeker/results.jsonl.gz."
        ),
    )
    parser.add_argument(
        "--shard-result",
        type=Path,
        action="append",
        help="Result log path; repeat four times instead of --shard-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "merged/libseeker-results/results.jsonl.gz",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing merged result and summary.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow fewer than 876 ELF per shard (intended only for tests).",
    )
    args = parser.parse_args()
    if args.shard_root and args.shard_result:
        parser.error("use either --shard-root or --shard-result, not both")
    if args.shard_result:
        inputs = args.shard_result
    else:
        roots = args.shard_root or [
            DATASET_DIR / f"shards/libseeker-shard-{index}"
            for index in range(EXPECTED_SHARDS)
        ]
        inputs = [root / "results/libseeker/results.jsonl.gz" for root in roots]
    if len(inputs) != EXPECTED_SHARDS:
        parser.error(f"exactly {EXPECTED_SHARDS} shard inputs are required")
    args.inputs = [path.expanduser().resolve() for path in inputs]
    args.output = args.output.expanduser().resolve()
    if args.output == Path("/") or args.output.is_dir():
        parser.error(f"unsafe output path: {args.output}")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summary_path(result_log: Path) -> Path:
    name = result_log.name
    for suffix in (".jsonl.gz", ".gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return result_log.with_name(f"{name}.summary.json")


def records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSON record is not an object")
            yield line_number, payload


def emit(stream, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def load_sidecar(path: Path) -> dict:
    sidecar_path = summary_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not sidecar.get("valid"):
        raise ValueError(f"{sidecar_path}: result is not marked valid")
    actual_hash = sha256_file(path)
    if sidecar.get("result_log_sha256") != actual_hash:
        raise ValueError(f"{path}: SHA-256 does not match {sidecar_path.name}")
    return sidecar


def compiler_command(record: dict) -> str:
    value = str(record.get("compiler_command") or record.get("compiler") or "")
    name = Path(value).name
    for command in COMPILERS:
        if name == command or name.startswith(f"{command}-"):
            return command
    return name


def compatible_run(record: dict) -> dict:
    """Drop only fields that legitimately differ between compiler shards."""
    return {
        key: value
        for key, value in record.items()
        if key not in {
            "elf_count",
            "expected_library_matches",
            "dataset_manifest",
            "dataset_manifest_sha256",
            "library_matrix",
        }
    }


def normalized_library(record: dict) -> dict:
    # No machine-local paths should normally be present. Ignore one if an older
    # producer included it so content identity remains portable.
    return {
        key: value
        for key, value in record.items()
        if key not in {"path", "absolute_path"}
    }


def finite_ratio(value, *, field: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid {field}={value!r}") from error
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{context}: {field}={number} is outside [0, 1]")
    return number


def validate_descriptor(descriptor: object, *, context: str) -> dict:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{context}: function descriptor is missing")
    for field in ("index", "address", "size", "blocks", "instructions"):
        value = descriptor.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{context}: invalid function {field}={value!r}")
    if not isinstance(descriptor.get("name"), str):
        raise ValueError(f"{context}: invalid function name")
    return descriptor


def validate_function_match(record: object, *, context: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{context}: function match is not an object")
    reference = validate_descriptor(
        record.get("reference_function"), context=f"{context} reference"
    )
    elf_function = validate_descriptor(
        record.get("elf_function"), context=f"{context} ELF"
    )
    if record.get("target_function_index") != reference["index"]:
        raise ValueError(f"{context}: reference function index mismatch")
    if record.get("source_function_index") != elf_function["index"]:
        raise ValueError(f"{context}: ELF function index mismatch")
    for field in ("dominant_ratio", "coverage_mean", "coverage_ratio"):
        finite_ratio(record.get(field), field=field, context=context)


class ShardValidator:
    def __init__(
        self,
        *,
        path: Path,
        catalog: dict[str, dict],
        pipelines: tuple[str, ...],
        global_coordinates: set[tuple[str, str, str]],
    ) -> None:
        self.path = path
        self.catalog = catalog
        self.pipelines = pipelines
        self.global_coordinates = global_coordinates
        self.current_elf: dict | None = None
        self.matches: dict[str, set[str]] = {}
        self.function_catalogs = 0
        self.elf_functions: dict[int, dict] = {}
        self.elf_calls: dict[int, list[int]] = {}
        self.coordinates: list[tuple[str, str, str]] = []
        self.counts = Counter()

    def finish_elf(self) -> None:
        if self.current_elf is None:
            return
        elf_id = self.current_elf["elf_id"]
        expected_libraries = set(self.catalog)
        for pipeline in self.pipelines:
            observed = self.matches.get(pipeline, set())
            if observed != expected_libraries:
                raise ValueError(
                    f"{self.path}: ELF {elf_id} pipeline={pipeline} has "
                    f"{len(observed)}/{len(expected_libraries)} library results"
                )
        expected_catalogs = 1 if "current" in self.pipelines else 0
        if self.function_catalogs != expected_catalogs:
            raise ValueError(
                f"{self.path}: ELF {elf_id} has {self.function_catalogs} "
                f"function catalogs, expected {expected_catalogs}"
            )
        self.current_elf = None

    def consume(self, record: dict, *, line_number: int) -> None:
        kind = record.get("type")
        context = f"{self.path}:{line_number}"
        if kind == "elf":
            self.finish_elf()
            elf_id = record.get("elf_id")
            if not isinstance(elf_id, str) or len(elf_id) != 64:
                raise ValueError(f"{context}: invalid elf_id")
            coordinate = (
                str(record.get("program", "")),
                compiler_command(record),
                str(record.get("program_optimization", "")),
            )
            if not all(coordinate):
                raise ValueError(f"{context}: incomplete ELF coordinate {coordinate}")
            if coordinate in self.global_coordinates:
                raise ValueError(f"{context}: duplicate ELF coordinate {coordinate}")
            self.global_coordinates.add(coordinate)
            self.coordinates.append(coordinate)
            self.current_elf = record
            self.matches = {pipeline: set() for pipeline in self.pipelines}
            self.function_catalogs = 0
            self.elf_functions = {}
            self.elf_calls = {}
            self.counts["elf_records"] += 1
            return

        if self.current_elf is None:
            raise ValueError(f"{context}: {kind!r} record precedes first ELF")
        if record.get("elf_id") != self.current_elf["elf_id"]:
            raise ValueError(f"{context}: record belongs to a different ELF")

        if kind == "library_match":
            pipeline = str(record.get("pipeline", ""))
            library_id = str(record.get("library_id", ""))
            if pipeline not in self.matches:
                raise ValueError(f"{context}: unknown pipeline {pipeline!r}")
            if library_id not in self.catalog:
                raise ValueError(f"{context}: unknown library_id {library_id!r}")
            if library_id in self.matches[pipeline]:
                raise ValueError(
                    f"{context}: duplicate result for {pipeline}/{library_id}"
                )
            self.matches[pipeline].add(library_id)
            self.counts["library_matches"] += 1
            return

        if kind == "elf_function_catalog":
            if "current" not in self.pipelines:
                raise ValueError(f"{context}: unexpected ELF function catalog")
            if self.function_catalogs:
                raise ValueError(f"{context}: duplicate ELF function catalog")
            functions = record.get("functions")
            if not isinstance(functions, list):
                raise ValueError(f"{context}: missing ELF functions")
            for index, descriptor in enumerate(functions):
                parsed = validate_descriptor(descriptor, context=context)
                if parsed["index"] != index:
                    raise ValueError(f"{context}: non-contiguous ELF function indices")
                self.elf_functions[index] = parsed
            calls = record.get("calls")
            if not isinstance(calls, list):
                raise ValueError(f"{context}: missing ELF call catalog")
            for call in calls:
                if not isinstance(call, list) or len(call) != 2:
                    raise ValueError(f"{context}: malformed ELF call entry")
                source_index, targets = call
                if source_index not in self.elf_functions or source_index in self.elf_calls:
                    raise ValueError(f"{context}: invalid/duplicate ELF caller index")
                if not isinstance(targets, list) or any(
                    target not in self.elf_functions for target in targets
                ):
                    raise ValueError(f"{context}: invalid ELF call target")
                self.elf_calls[source_index] = targets
            self.function_catalogs += 1
            return

        if kind == "cu_match":
            if record.get("library_id") not in self.catalog:
                raise ValueError(f"{context}: unknown CU library_id")
            if int(record.get("schema_version", 0)) < 2:
                raise ValueError(f"{context}: CU schema is not autonomous")
            selected = record.get("selected_match")
            if not isinstance(selected, dict):
                raise ValueError(f"{context}: selected_match is missing")
            status = record.get("cu_status")
            if status not in {"PASS", "DROP", "DROP_RODATA"}:
                raise ValueError(f"{context}: invalid CU status {status!r}")
            if status == "PASS" and selected.get("passed") is not True:
                raise ValueError(f"{context}: accepted CU has passed=false")
            if status == "DROP" and selected.get("passed") is not False:
                raise ValueError(f"{context}: block-rejected CU has passed=true")
            if not isinstance(record.get("name"), str):
                raise ValueError(f"{context}: CU name is missing")
            target_cu_index = record.get("target_cu_index")
            if not isinstance(target_cu_index, int) or target_cu_index < 0:
                raise ValueError(f"{context}: invalid target_cu_index")
            target_functions = record.get("target_functions")
            if not isinstance(target_functions, list):
                raise ValueError(f"{context}: reference CU functions are missing")
            reference_functions: dict[int, dict] = {}
            for index, descriptor in enumerate(target_functions):
                parsed = validate_descriptor(descriptor, context=context)
                if parsed["index"] != index:
                    raise ValueError(
                        f"{context}: non-contiguous reference function indices"
                    )
                reference_functions[index] = parsed
            function_matches = selected.get("function_matches")
            if not isinstance(function_matches, list):
                raise ValueError(f"{context}: selected function matches are missing")
            for index, function_match in enumerate(function_matches):
                validate_function_match(
                    function_match, context=f"{context} function_match[{index}]"
                )
                reference_index = function_match["target_function_index"]
                source_index = function_match["source_function_index"]
                if function_match["reference_function"] != reference_functions.get(
                    reference_index
                ):
                    raise ValueError(
                        f"{context}: mapped reference descriptor is inconsistent"
                    )
                if function_match["elf_function"] != self.elf_functions.get(
                    source_index
                ):
                    raise ValueError(
                        f"{context}: mapped ELF descriptor is inconsistent"
                    )
                call_targets = function_match.get("source_call_targets")
                if not isinstance(call_targets, list) or call_targets != self.elf_calls.get(
                    source_index, []
                ):
                    raise ValueError(
                        f"{context}: mapped ELF call targets are inconsistent"
                    )
            self.counts["cu_matches"] += 1
            if status == "PASS":
                self.counts["accepted_cu_matches"] += 1
            self.counts["function_matches"] += len(function_matches)
            return

        if kind == "failure":
            self.counts["failures"] += 1
            return
        raise ValueError(f"{context}: unsupported record type {kind!r}")


def read_header(path: Path):
    """Read the small run/catalog prefix and stream the remaining records."""
    iterator = iter(records(path))
    try:
        line_number, run = next(iterator)
    except StopIteration as error:
        raise ValueError(f"{path}: empty result log") from error
    if line_number != 1 or run.get("type") != "run":
        raise ValueError(f"{path}: first record is not a run header")
    try:
        library_count = int(run["library_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid library_count") from error
    catalog: dict[str, dict] = {}
    for _index in range(library_count):
        try:
            line_number, record = next(iterator)
        except StopIteration as error:
            raise ValueError(f"{path}: truncated library catalog") from error
        if record.get("type") != "library":
            raise ValueError(f"{path}:{line_number}: expected library record")
        library_id = str(record.get("library_id", ""))
        if not library_id or library_id in catalog:
            raise ValueError(f"{path}:{line_number}: invalid/duplicate library_id")
        catalog[library_id] = normalized_library(record)
    return run, catalog, iterator


def process_shard(
    path: Path,
    payload_path: Path,
    *,
    common_run: dict | None,
    common_catalog: dict[str, dict] | None,
    global_coordinates: set[tuple[str, str, str]],
) -> tuple[dict, dict[str, dict], Counter, list[tuple[str, str, str]], dict]:
    sidecar = load_sidecar(path)
    run, catalog, payload_records = read_header(path)
    if common_run is not None and compatible_run(run) != compatible_run(common_run):
        raise ValueError(f"{path}: matching configuration/catalog provenance differs")
    if common_catalog is not None and catalog != common_catalog:
        raise ValueError(f"{path}: library catalog differs from the first shard")
    pipelines = tuple(str(value) for value in run.get("pipelines", []))
    if not pipelines or len(pipelines) != len(set(pipelines)):
        raise ValueError(f"{path}: invalid pipeline list")
    validator = ShardValidator(
        path=path,
        catalog=catalog,
        pipelines=pipelines,
        global_coordinates=global_coordinates,
    )
    with gzip.open(payload_path, "wt", encoding="utf-8", compresslevel=6) as output:
        for line_number, record in payload_records:
            validator.consume(record, line_number=line_number)
            emit(output, record)
    validator.finish_elf()

    for field in (
        "elf_records", "library_matches", "cu_matches",
        "accepted_cu_matches", "function_matches", "failures",
    ):
        if int(sidecar.get(field, 0)) != validator.counts[field]:
            raise ValueError(
                f"{path}: sidecar {field}={sidecar.get(field, 0)} but "
                f"log contains {validator.counts[field]}"
            )
    expected_matches = (
        validator.counts["elf_records"] * len(catalog) * len(pipelines)
    )
    if validator.counts["library_matches"] != expected_matches:
        raise ValueError(f"{path}: incomplete library-match matrix")
    if int(run.get("elf_count", -1)) != validator.counts["elf_records"]:
        raise ValueError(f"{path}: run header ELF count is inconsistent")
    if int(run.get("expected_library_matches", -1)) != expected_matches:
        raise ValueError(f"{path}: run header library-match count is inconsistent")
    if int(sidecar.get("library_records", -1)) != len(catalog):
        raise ValueError(f"{path}: sidecar library catalog count is inconsistent")
    if int(sidecar.get("result_log_size", -1)) != path.stat().st_size:
        raise ValueError(f"{path}: sidecar result size is inconsistent")
    return run, catalog, validator.counts, validator.coordinates, sidecar


def validate_matrix(
    shard_coordinates: list[list[tuple[str, str, str]]], *, allow_incomplete: bool
) -> dict:
    compilers: list[str] = []
    all_coordinates = [coordinate for shard in shard_coordinates for coordinate in shard]
    for index, coordinates in enumerate(shard_coordinates):
        commands = {compiler for _program, compiler, _optimization in coordinates}
        if len(commands) != 1:
            raise ValueError(f"shard {index}: expected one compiler, found {sorted(commands)}")
        compilers.append(next(iter(commands)))
        if not allow_incomplete and len(coordinates) != EXPECTED_PER_SHARD:
            raise ValueError(
                f"shard {index}: ELF={len(coordinates)}, expected={EXPECTED_PER_SHARD}"
            )
    if not allow_incomplete and set(compilers) != set(COMPILERS):
        raise ValueError(f"compiler shards={compilers}, expected={list(COMPILERS)}")
    if not allow_incomplete:
        cells: dict[str, set[tuple[str, str]]] = {}
        for program, compiler, optimization in all_coordinates:
            cells.setdefault(program, set()).add((compiler, optimization))
        expected_cells = {(compiler, opt) for compiler in COMPILERS for opt in OPTIMIZATIONS}
        incomplete = {
            program: len(observed)
            for program, observed in cells.items()
            if observed != expected_cells
        }
        if len(cells) != EXPECTED_PROGRAMS or incomplete:
            raise ValueError(
                f"merged program matrix is incomplete: programs={len(cells)}, "
                f"incomplete={len(incomplete)}"
            )
        if len(all_coordinates) != EXPECTED_TOTAL:
            raise ValueError(
                f"merged ELF={len(all_coordinates)}, expected={EXPECTED_TOTAL}"
            )
    return {
        "compiler_shards": compilers,
        "programs": len({coordinate[0] for coordinate in all_coordinates}),
        "coordinates": len(all_coordinates),
    }


def concatenate_gzip_payloads(paths: list[Path], output) -> None:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                output.write(line)


def main() -> int:
    args = parse_args()
    output_summary = summary_path(args.output)
    if (args.output.exists() or output_summary.exists()) and not args.replace:
        raise FileExistsError(
            f"output exists; use --replace: {args.output} / {output_summary}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload_paths: list[Path] = []
    output_temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    summary_temporary = output_summary.with_name(
        f".{output_summary.name}.{os.getpid()}.tmp"
    )
    common_run: dict | None = None
    common_catalog: dict[str, dict] | None = None
    aggregate_counts = Counter()
    shard_coordinates: list[list[tuple[str, str, str]]] = []
    shard_summaries: list[dict] = []
    global_coordinates: set[tuple[str, str, str]] = set()
    try:
        for index, path in enumerate(args.inputs):
            handle, temporary_name = tempfile.mkstemp(
                prefix=f".libseeker-shard-{index}.", suffix=".jsonl.gz",
                dir=args.output.parent,
            )
            os.close(handle)
            payload = Path(temporary_name)
            payload_paths.append(payload)
            run, catalog, counts, coordinates, sidecar = process_shard(
                path,
                payload,
                common_run=common_run,
                common_catalog=common_catalog,
                global_coordinates=global_coordinates,
            )
            common_run = common_run or run
            common_catalog = common_catalog or catalog
            aggregate_counts.update(counts)
            shard_coordinates.append(coordinates)
            shard_summaries.append({
                "result_log_sha256": sidecar["result_log_sha256"],
                "dataset_manifest_sha256": run.get(
                    "dataset_manifest_sha256"
                ),
                "compiler": (
                    coordinates[0][1] if coordinates else None
                ),
                "elf_records": counts["elf_records"],
                "library_matches": counts["library_matches"],
            })

        assert common_run is not None and common_catalog is not None
        matrix = validate_matrix(
            shard_coordinates, allow_incomplete=args.allow_incomplete
        )
        merged_run = dict(common_run)
        merged_run.update({
            "type": "run",
            "schema_version": max(1, int(common_run.get("schema_version", 1))),
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "aggregation": "results_only",
            "shard_count": EXPECTED_SHARDS,
            "elf_count": aggregate_counts["elf_records"],
            "library_count": len(common_catalog),
            "expected_library_matches": aggregate_counts["library_matches"],
            "dataset_manifest": None,
            "dataset_manifest_sha256": None,
            "source_shards": shard_summaries,
        })
        with gzip.open(
            output_temporary, "wt", encoding="utf-8", compresslevel=6
        ) as output:
            emit(output, merged_run)
            for library_id in sorted(common_catalog):
                emit(output, common_catalog[library_id])
            concatenate_gzip_payloads(payload_paths, output)
        with gzip.open(output_temporary, "rb") as stream:
            while stream.read(1024 * 1024):
                pass

        summary = {
            "schema_version": 1,
            "valid": True,
            "aggregation": "results_only",
            "result_log": args.output.as_posix(),
            "result_log_sha256": sha256_file(output_temporary),
            "result_log_size": output_temporary.stat().st_size,
            "shard_count": EXPECTED_SHARDS,
            "library_records": len(common_catalog),
            **aggregate_counts,
            **matrix,
            "shards": shard_summaries,
        }
        summary_temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(output_temporary, args.output)
        os.replace(summary_temporary, output_summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        output_temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)
        for path in payload_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
