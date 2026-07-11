import time
import os
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import re
import site
import glob
import ctypes
import warnings
import sys
import tempfile
import subprocess


def _prepare_tf_gpu_runtime() -> None:
    """Expose and preload pip-installed NVIDIA libs before importing TensorFlow."""
    nvidia_lib_dirs: list[str] = []
    for base in site.getsitepackages():
        for lib_dir in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
            if os.path.isdir(lib_dir):
                nvidia_lib_dirs.append(lib_dir)

    if not nvidia_lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    missing = [p for p in nvidia_lib_dirs if p not in current_parts]
    if missing:
        prefix = ":".join(nvidia_lib_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix

    # Preload all NVIDIA shared libs so TensorFlow can resolve CUDA symbols
    # even when the process started without a complete linker path.
    for lib_dir in nvidia_lib_dirs:
        for so_file in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
            if os.path.basename(so_file).startswith("libnvblas.so"):
                continue
            try:
                ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def _configure_runtime_from_cli() -> None:
    """Apply runtime options that must be set before importing TensorFlow."""
    hide_warnings = ("--hide-warnings" in sys.argv)
    if not hide_warnings:
        return

    # Hide informational TensorFlow logs that are noisy in CLI runs.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Optional: disable oneDNN custom ops to avoid related startup notice.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # Suppress a known Keras warning caused by optimizer state mismatch.
    warnings.filterwarnings(
        "ignore",
        message=r"Skipping variable loading for optimizer 'adam'.*",
        category=UserWarning,
    )

_prepare_tf_gpu_runtime()
_configure_runtime_from_cli()

import tensorflow as tf
from model import PalmTree
from asm import CodeUnit, parse_r2_file
from match import (
    RodataEvidence,
    RodataMatchResult,
    build_rodata_index,
    classify_rodata_evidence,
    evaluate_block_presence,
    evaluate_rodata_match,
)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("TensorFlow GPUs:", gpus)


PATH_TO_CALCULATOR = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..", "..", "..",
        "Thesis_Binary_Analysis",
        "Exploration", "easy", "gcc11", "calculator_static_opt"
    )
)

PATH_TO_LIBRARIES = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..", "..", "..",
        "Thesis_Binary_Analysis",
        "Exploration", "easy", "libraries"
    )
)

OUTPUT = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "output.txt"
    )
)

def get_args():
    parser = argparse.ArgumentParser(description="Compilation-unit block matching")

    parser.add_argument(
        "--path_to_binary",
        type=str,
        default=PATH_TO_CALCULATOR,
        help="Path to the binary to analyze"
    )
    parser.add_argument(
        "--libraries_dir",
        type=str,
        default=PATH_TO_LIBRARIES,
        help="Path to the directory containing the libraries (.a or .a.*)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT,
        help="Optional output file path"
    )
    parser.add_argument(
        "--features_output",
        type=str,
        default=None,
        help="Optional JSONL file with full CU-level matching features"
    )
    parser.add_argument(
        "--asm_model",
        type=str,
        default="palmtree/model/transformer.ep19",
        help="Path to the PalmTree model"
    )
    parser.add_argument(
        "--block_threshold",
        type=float,
        default=0.70,
        help="Per-block similarity threshold used when computing coverage_ratio"
    )
    parser.add_argument(
        "--block_coverage_mean_threshold",
        type=float,
        default=0.80,
        help="Minimum mean best-block similarity"
    )
    parser.add_argument(
        "--block_assignment_threshold",
        type=float,
        default=0.875,
        help="Minimum Hungarian-assignment mean similarity for the second-stage block match"
    )
    parser.add_argument(
        "--block_min_coverage_ratio",
        type=float,
        default=0.50,
        help="Minimum ratio of target CU blocks with similarity >= block_threshold"
    )
    parser.add_argument(
        "--block_locality_window_multiplier",
        type=float,
        default=5.0,
        help="Source-function window multiplier for all-CU block matching"
    )
    parser.add_argument(
        "--block_locality_window_padding",
        type=int,
        default=2,
        help="Extra source functions added to the all-CU block-locality window"
    )
    parser.add_argument(
        "--block_min_edge_locality_ratio",
        type=float,
        default=0.7,
        help="Minimum ratio of internal target call edges whose matched source functions stay local"
    )
    parser.add_argument(
        "--block_min_instructions",
        type=int,
        default=3,
        help="Minimum number of instructions required for a basic block to be used in block matching"
    )
    parser.add_argument(
        "--block_min_function_concentration",
        type=float,
        default=0.45,
        help="Minimum ratio of target blocks that must stay in each target function's dominant source function"
    )
    parser.add_argument(
        "--block_min_function_spread",
        type=float,
        default=0.30,
        help="Minimum ratio of distinct dominant source functions to mapped target functions"
    )
    parser.add_argument(
        "--disable_block_window_prefilter",
        action="store_true",
        help="Evaluate assignment/locality for every source-function window"
    )
    parser.add_argument(
        "--disable_rodata_filter",
        action="store_true",
        help="Disable .rodata-based penalty on block-level CU matches"
    )
    parser.add_argument(
        "--rodata_min_bytes",
        type=int,
        default=512,
        help="Minimum target .rodata bytes for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_min_strings",
        type=int,
        default=0,
        help="Minimum target .rodata strings for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_min_ngrams",
        type=int,
        default=32,
        help="Minimum target .rodata byte n-grams for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_penalty_threshold",
        type=float,
        default=0.20,
        help="Drop a block-passed CU when informative .rodata score is at or below this value"
    )
    parser.add_argument(
        "--rodata_confirm_threshold",
        type=float,
        default=0.70,
        help="Mark a CU as .rodata-confirmed when informative .rodata score is at or above this value"
    )
    parser.add_argument(
        "--hide-warnings",
        action="store_true",
        help="Hide non-critical TensorFlow/Keras startup warnings"
    )
    return parser.parse_args()


def log_line(line: str, output_file: str | None = None) -> None:
    print(line)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            print(line, file=f)


def initialize_output_file(output_file: str | None) -> None:
    """Prepare output file before writing logs."""
    if not output_file:
        return

    output_path = Path(output_file)
    if output_path.parent and not output_path.parent.exists():
        raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    with open(output_file, "w", encoding="utf-8"):
        pass


def log_parse_debug(code_unit: CodeUnit) -> None:
    print(
        f"[DEBUG] parsed {code_unit.name}: "
        f"type={code_unit.unit_type}, "
        f"functions={code_unit.get_num_functions()}, "
        f"symbol_fallback_count={code_unit.symbol_fallback_count}, "
        f"pseudo_block_fallback_count={code_unit.pseudo_block_fallback_count}, "
        f"rodata_sections={code_unit.rodata_section_count}, "
        f"rodata_bytes={code_unit.get_rodata_size()}, "
        f"rodata_strings={len(code_unit.rodata_strings)}"
    )


def empty_rodata_result(target_unit: CodeUnit) -> RodataMatchResult:
    """Return a neutral .rodata result for CUs where .rodata was not checked."""
    return RodataMatchResult(
        score=0.0,
        string_score=0.0,
        byte_score=0.0,
        target_bytes=target_unit.get_rodata_size(),
        source_bytes=0,
        matched_strings=0,
        target_strings=len(target_unit.rodata_strings),
        matched_ngrams=0,
        target_ngrams=0,
        has_rodata=bool(target_unit.get_rodata_size() or target_unit.rodata_strings),
    )


def cu_status(block_result, rodata_evidence: RodataEvidence, rodata_disabled: bool) -> str:
    """Return the report status for a candidate CU."""
    if not block_result.passed:
        return "DROP"
    if not rodata_disabled and rodata_evidence.status == "penalty":
        return "DROP_RODATA"
    return "PASS"


def format_block_cu_line(
    index: int,
    target_unit: CodeUnit,
    block_status: str,
    rodata_evidence: RodataEvidence,
) -> str:
    """Format one compact CU-level report line."""
    return (
        f"    BLOCK_CU[{index:03}] {block_status} "
        f"name={target_unit.name} "
        f"rodata_status={rodata_evidence.status} "
        f"target_functions={target_unit.get_num_functions()}"
    )


def block_cu_feature_record(
    binary_path: str,
    library_name: str,
    target_unit: CodeUnit,
    block_result,
    rodata_result: RodataMatchResult,
    rodata_evidence: RodataEvidence,
    block_status: str,
) -> dict:
    """Build the full CU-level feature row used by threshold tuning."""
    return {
        "type": "block_cu",
        "binary_path": binary_path,
        "library": Path(library_name).name,
        "name": target_unit.name,
        "status": block_status,
        "block": float(block_result.score),
        "coverage_mean": float(block_result.coverage_mean),
        "coverage_min": float(block_result.coverage_min),
        "coverage_ratio": float(block_result.coverage_ratio),
        "assignment_mean": float(block_result.assignment_mean),
        "assignment_min": float(block_result.assignment_min),
        "rodata": float(rodata_result.score),
        "rodata_strings": int(rodata_result.target_strings),
        "rodata_matched_strings": int(rodata_result.matched_strings),
        "rodata_ngrams": int(rodata_result.target_ngrams),
        "rodata_matched_ngrams": int(rodata_result.matched_ngrams),
        "rodata_bytes": int(rodata_result.target_bytes),
        "rodata_status": rodata_evidence.status,
        "blocks_source": int(block_result.num_source_blocks),
        "blocks_target": int(block_result.num_target_blocks),
        "locality_span": int(block_result.locality_span),
        "windows_evaluated": int(block_result.windows_evaluated),
        "windows_total": int(block_result.windows_total),
        "windows_skipped": int(block_result.windows_skipped),
        "edge_locality_ratio": float(block_result.edge_locality_ratio),
        "function_concentration": float(block_result.function_concentration),
        "function_spread": float(block_result.function_spread),
        "target_functions": int(target_unit.get_num_functions()),
    }


def process_bin(binary: CodeUnit, lib: dict[str, list[CodeUnit]], args):
    """Process a parsed binary against all libraries."""
    output_file = args.output if args.output else None
    features_context = (
        open(args.features_output, "a", encoding="utf-8")
        if args.features_output
        else nullcontext(None)
    )
    binary_path = str(Path(args.path_to_binary).resolve())
    binary_rodata_index = None

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    with features_context as features_file:
        for library_name, comp_units in lib.items():
            start_library = time.time()
            target_comp_units = [
                comp_unit
                for comp_unit in comp_units
                if comp_unit.get_num_functions() > 1
            ]

            if not target_comp_units:
                stop_library = time.time()
                elapsed_time = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(stop_library - start_library),
                )
                line = (
                    f"{'NO':7} | "
                    f"library={Path(library_name).name} | "
                    f"score={0.0:6.2f}% | "
                    f"block_best_any={0.0:6.2f}% | "
                    f"block_best_matched={0.0:6.2f}% | "
                    f"matched_cu=0/0 | "
                    f"time={elapsed_time} | "
                    f"no multi-function compilation units"
                )
                log_line(line, output_file)
                continue

            block_results = []
            rodata_evidence_by_cu = {}

            for target_unit in target_comp_units:
                block_result = evaluate_block_presence(
                    source_unit=binary,
                    target_unit=target_unit,
                    block_threshold=args.block_threshold,
                    block_assignment_threshold=args.block_assignment_threshold,
                    min_coverage_ratio=args.block_min_coverage_ratio,
                    min_coverage_mean=args.block_coverage_mean_threshold,
                    locality_window_multiplier=args.block_locality_window_multiplier,
                    locality_window_padding=args.block_locality_window_padding,
                    min_edge_locality_ratio=args.block_min_edge_locality_ratio,
                    min_block_instructions=args.block_min_instructions,
                    min_function_concentration=args.block_min_function_concentration,
                    min_function_spread=args.block_min_function_spread,
                    enable_window_prefilter=not args.disable_block_window_prefilter,
                )
                block_results.append((target_unit, block_result))

                if block_result.passed and not args.disable_rodata_filter:
                    if binary_rodata_index is None:
                        binary_rodata_index = build_rodata_index(binary)
                    rodata_result = evaluate_rodata_match(
                        binary_rodata_index,
                        target_unit,
                    )
                    rodata_evidence = classify_rodata_evidence(
                        rodata_result,
                        args.rodata_min_bytes,
                        args.rodata_min_strings,
                        args.rodata_min_ngrams,
                        args.rodata_penalty_threshold,
                        args.rodata_confirm_threshold,
                    )
                else:
                    rodata_result = empty_rodata_result(target_unit)
                    rodata_evidence = RodataEvidence(
                        status="disabled" if args.disable_rodata_filter else "skipped",
                        informative=False,
                    )

                rodata_evidence_by_cu[id(target_unit)] = rodata_evidence
                block_status = cu_status(
                    block_result,
                    rodata_evidence,
                    args.disable_rodata_filter,
                )
                if features_file:
                    print(
                        json.dumps(
                            block_cu_feature_record(
                                binary_path,
                                library_name,
                                target_unit,
                                block_result,
                                rodata_result,
                                rodata_evidence,
                                block_status,
                            ),
                            sort_keys=True,
                        ),
                        file=features_file,
                    )

            successful_block_results = [
                (target_unit, block_result)
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and (
                        args.disable_rodata_filter
                        or rodata_evidence_by_cu[id(target_unit)].status != "penalty"
                    )
                )
            ]
            block_best_score = max(
                (block_result.score for _, block_result in block_results),
                default=0.0,
            )
            successful_block_best_score = max(
                (block_result.score for _, block_result in successful_block_results),
                default=0.0,
            )
            score_tot = (
                sum(block_result.score for _, block_result in successful_block_results)
                / len(successful_block_results)
                if successful_block_results
                else 0.0
            )

            stop_library = time.time()
            elapsed_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(stop_library - start_library),
            )

            matched_cu = len(successful_block_results)
            total_cu = len(target_comp_units)
            percentage = score_tot * 100.0

            status = "YES" if matched_cu >= 1 else "NO"

            rodata_confirmed_count = sum(
                1
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and rodata_evidence_by_cu[id(target_unit)].status == "confirm"
                )
            )
            rodata_penalty_count = sum(
                1
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and rodata_evidence_by_cu[id(target_unit)].status == "penalty"
                )
            )
            line = (
                f"{status:7} | "
                f"library={Path(library_name).name} | "
                f"score={percentage:6.2f}% | "
                f"block_best_any={block_best_score * 100.0:6.2f}% | "
                f"block_best_matched={successful_block_best_score * 100.0:6.2f}% | "
                f"rodata_confirmed_cu={rodata_confirmed_count} | "
                f"rodata_penalty_cu={rodata_penalty_count} | "
                f"matched_cu={matched_cu}/{total_cu} | "
                f"time={elapsed_time}"
            )

            for i, (target_unit, block_result) in enumerate(block_results):
                rodata_evidence = rodata_evidence_by_cu[id(target_unit)]
                log_line(
                    format_block_cu_line(
                        i,
                        target_unit,
                        cu_status(
                            block_result,
                            rodata_evidence,
                            args.disable_rodata_filter,
                        ),
                        rodata_evidence,
                    ),
                    output_file
                )

            log_line(line, output_file)

if __name__ == "__main__":
    start_main = time.time()

    args = get_args()
    initialize_output_file(args.output)
    initialize_output_file(args.features_output)

    binary_path = Path(args.path_to_binary)
    if not binary_path.exists():
        raise ValueError(f"Can't find binary '{binary_path}'")
    if not binary_path.is_file():
        raise ValueError(f"Not a file '{binary_path}'")

    libraries_dir = Path(args.libraries_dir)
    if not libraries_dir.exists():
        raise ValueError(f"Can't find '{libraries_dir}'")
    if not libraries_dir.is_dir():
        raise ValueError(f"Not a directory '{libraries_dir}'")

    pattern = re.compile(r".*\.a(\..+)?$")

    library_files = sorted(
        [
            f for f in libraries_dir.iterdir()
            if f.is_file() and pattern.match(f.name)
        ]
    )

    if not library_files:
        raise ValueError(f"No .a or .a.* files found in directory '{libraries_dir}'")
    else:
        print(f"Found {len(library_files)} library files in '{libraries_dir}'")
        
    asm_model = PalmTree("Palm Tree")
    asm_model.load(args.asm_model)

    lib: dict[str, list[CodeUnit]] = {}
    for library_file in library_files:
        comp_units = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["ar", "x", library_file.as_posix()],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"[ERROR] ar failed on {library_file.name}")
                print(result.stderr)
                lib[library_file.as_posix()] = comp_units
                continue

            extracted_objects = sorted(
                [
                    Path(tmpdir) / f
                    for f in os.listdir(tmpdir)
                    if (Path(tmpdir) / f).is_file()
                ]
            )

            for obj_file in extracted_objects:
                try:
                    b = parse_r2_file(
                        obj_file.as_posix(),
                        asm_model=asm_model,
                        unit_type=CodeUnit.TYPE_CU,
                    )
                    log_parse_debug(b)
                    if b.get_num_functions() > 0:
                        comp_units.append(b)
                except Exception as e:
                    print(f"[WARN] Failed to parse {obj_file}: {e}")

        lib[library_file.as_posix()] = comp_units
        print(f"[DEBUG] {library_file.name}: loaded {len(comp_units)} object files")

    log_line(f"Found {len(library_files)} libraries to process", args.output)
    log_line(f"Target binary: {binary_path}", args.output)
    log_line(f"Start processing: {time.strftime('%H:%M:%S', time.gmtime())}\n", args.output)

    binary = parse_r2_file(
        binary_path.as_posix(),
        asm_model=asm_model,
        unit_type=CodeUnit.TYPE_ELF,
    )
    log_parse_debug(binary)

    process_bin(binary, lib, args)

    elapsed_main = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_main))
    log_line(f"\nDone processing in {elapsed_main}", args.output)
