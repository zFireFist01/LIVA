import time
import os
import argparse
from pathlib import Path
import signal
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

class TimeoutException(Exception):
    """Exception raised when a timeout occurs"""
    pass


def handler(signum, frame):
    raise TimeoutException("Timeout while processing the binary")


signal.signal(signal.SIGALRM, handler)


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
        help="Minimum ratio of source blocks with similarity >= block_threshold"
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
        "--min_cu",
        type=int,
        default=1,
        help="Minimum number of matched compilation units"
    )
    parser.add_argument(
        "--min_functions_cu",
        type=int,
        default=1,
        help="Minimum matched functions per compilation unit"
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


def filter_compilation_units(comp_units: list[CodeUnit]) -> tuple[list[CodeUnit], int]:
    """Drop single-function compilation units before matching."""
    eligible_comp_units = [cu for cu in comp_units if cu.get_num_functions() > 1]
    excluded_single_function_cu = len(comp_units) - len(eligible_comp_units)
    return eligible_comp_units, excluded_single_function_cu


def process_bin(binary: CodeUnit, lib: dict[str, list[CodeUnit]], args):
    """Process a parsed binary against all libraries."""
    output_file = args.output if args.output else None
    binary_rodata_index = build_rodata_index(binary)

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    for library_name, comp_units in lib.items():
        start_library = time.time()
        eligible_comp_units, excluded_single_function_cu = filter_compilation_units(comp_units)

        if not eligible_comp_units:
            stop_library = time.time()
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))
            line = (
                f"{'NO':7} | "
                f"library={Path(library_name).name} | "
                f"score={0.0:6.2f}% | "
                f"matched_cu=0/0 | "
                f"matched_functions=0 | "
                f"time={elapsed_time} | "
                f"no eligible compilation units | "
                f"excluded_single_function_cu={excluded_single_function_cu}"
            )
            log_line(line, output_file)
            continue

        block_results = []
        rodata_results_by_cu = {}
        rodata_evidence_by_cu = {}

        for target_unit in eligible_comp_units:
            rodata_result = evaluate_rodata_match(binary_rodata_index, target_unit)
            rodata_evidence = classify_rodata_evidence(
                rodata_result,
                args.rodata_min_bytes,
                args.rodata_min_strings,
                args.rodata_min_ngrams,
                args.rodata_penalty_threshold,
                args.rodata_confirm_threshold,
            )
            rodata_results_by_cu[id(target_unit)] = rodata_result
            rodata_evidence_by_cu[id(target_unit)] = rodata_evidence

            block_result = evaluate_block_presence(
                binary,
                target_unit,
                args.block_threshold,
                args.block_assignment_threshold,
                args.block_min_coverage_ratio,
                args.block_coverage_mean_threshold,
                args.block_locality_window_multiplier,
                args.block_locality_window_padding,
                args.block_min_edge_locality_ratio,
                args.block_min_instructions,
                args.block_min_function_concentration,
                args.block_min_function_spread,
                not args.disable_block_window_prefilter,
            )
            block_results.append((target_unit, block_result))

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
        successful_units = [
            target_unit
            for target_unit, _ in successful_block_results
        ]
        block_best_score = max(
            (block_result.score for _, block_result in block_results),
            default=0.0,
        )
        score_tot = (
            sum(block_result.score for _, block_result in successful_block_results)
            / len(successful_block_results)
            if successful_block_results
            else 0.0
        )

        stop_library = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))

        matched_cu = len(successful_units)
        total_cu = len(eligible_comp_units)
        matched_functions = sum(unit.get_num_functions() for unit in successful_units)
        percentage = score_tot * 100.0

        enough_cu = matched_cu >= min(args.min_cu, total_cu)
        enough_functions = (
            len([
                unit
                for unit in successful_units
                if unit.get_num_functions() >= args.min_functions_cu
            ])
            >= min(args.min_cu, total_cu)
        )

        if enough_cu and enough_functions:
            status = "YES"
        elif enough_cu:
            status = "YES [W]"
        else:
            status = "NO"

        rodata_best_score = max(
            (rodata_result.score for rodata_result in rodata_results_by_cu.values()),
            default=0.0,
        )
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
            f"block_best={block_best_score * 100.0:6.2f}% | "
            f"rodata_best={rodata_best_score * 100.0:6.2f}% | "
            f"rodata_confirmed_cu={rodata_confirmed_count} | "
            f"rodata_penalty_cu={rodata_penalty_count} | "
            f"candidate_cu={len(eligible_comp_units)}/{total_cu} | "
            f"matched_cu={matched_cu}/{total_cu} | "
            f"matched_functions={matched_functions} | "
            f"time={elapsed_time}"
        )

        for i, (target_unit, block_result) in enumerate(block_results):
            rodata_evidence = rodata_evidence_by_cu[id(target_unit)]
            if (
                block_result.passed
                and not args.disable_rodata_filter
                and rodata_evidence.status == "penalty"
            ):
                block_status = "DROP_RODATA"
            else:
                block_status = "PASS" if block_result.passed else "DROP"

            rodata_result = rodata_results_by_cu[id(target_unit)]
            log_line(
                f"    BLOCK_CU[{i:03}] {block_status} "
                f"name={target_unit.name} "
                f"block={block_result.score:.4f} "
                f"coverage_mean={block_result.coverage_mean:.4f} "
                f"coverage_min={block_result.coverage_min:.4f} "
                f"coverage_ratio={block_result.coverage_ratio:.4f} "
                f"assignment_mean={block_result.assignment_mean:.4f} "
                f"assignment_min={block_result.assignment_min:.4f} "
                f"rodata={rodata_result.score:.4f} "
                f"rodata_strings={rodata_result.matched_strings}/"
                f"{rodata_result.target_strings} "
                f"rodata_ngrams={rodata_result.matched_ngrams}/"
                f"{rodata_result.target_ngrams} "
                f"rodata_bytes={rodata_result.target_bytes} "
                f"rodata_status={rodata_evidence.status} "
                f"blocks={block_result.num_source_blocks}/{block_result.num_target_blocks} "
                f"locality_span={block_result.locality_span} "
                f"windows={block_result.windows_evaluated}/"
                f"{block_result.windows_total} "
                f"windows_skipped={block_result.windows_skipped} "
                f"edge_locality_ratio={block_result.edge_locality_ratio:.4f} "
                f"function_concentration={block_result.function_concentration:.4f} "
                f"function_spread={block_result.function_spread:.4f} "
                f"matched_functions={target_unit.get_num_functions()}",
                output_file
            )

        log_line(line, output_file)


if __name__ == "__main__":
    start_main = time.time()

    args = get_args()
    initialize_output_file(args.output)

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

            #print(f"[DEBUG] {library_file.name}: extracted {len(extracted_objects)} files")
            for obj_file in extracted_objects:
                print(f"         -> {obj_file.name}")

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

    signal.alarm(3600 * 4)
    try:
        process_bin(binary, lib, args)
    except TimeoutException as e:
        log_line(f"Timeout while processing binary '{binary_path.name}': {e}", args.output)
    finally:
        signal.alarm(0)

    elapsed_main = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_main))
    log_line(f"\nDone processing in {elapsed_main}", args.output)
