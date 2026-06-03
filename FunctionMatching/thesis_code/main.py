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
import numpy as np
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
from struct2vec import GraphNetwork
from asm import CodeUnit, parse_r2_file
from match import Match, evaluate_block_match, evaluate_block_presence, match_functions, select_block_candidates

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

SPACES = "     "


class TimeoutException(Exception):
    """Exception raised when a timeout occurs"""
    pass


def handler(signum, frame):
    raise TimeoutException("Timeout while processing the binary")


signal.signal(signal.SIGALRM, handler)


def get_args():
    parser = argparse.ArgumentParser(description="Function Matching")

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
        "--graph_model",
        type=str,
        default="struct2vec/struct2vec.weights.h5",
        help="Path to the graph model weights"
    )

    parser.add_argument(
        "--function_threshold",
        type=float,
        default=0.8,
        help="Function-matching CU score threshold"
    )
    parser.add_argument(
        "--candidate_low_threshold",
        type=float,
        default=0.45,
        help="Lower first-stage CU score threshold for recovery candidates that still receive block matching"
    )
    parser.add_argument(
        "--candidate_top_k",
        type=int,
        default=0,
        help="Maximum CU candidates per library to pass to block matching; 0 means no cap"
    )
    parser.add_argument(
        "--block_scope",
        choices=("all_cu", "function_candidates"),
        default="all_cu",
        help="Use all eligible CUs for block matching, or only CUs selected by function matching"
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
        default=0.78,
        help="Minimum mean best-block similarity"
    )
    parser.add_argument(
        "--block_assignment_threshold",
        type=float,
        default=0.75,
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
        default=3.0,
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
        default=0.5,
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
        default=0.50,
        help="Minimum ratio of distinct dominant source functions to mapped target functions"
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
    parser.add_argument(
        "--disable_block_matching",
        action="store_true",
        help="Disable second-stage block matching and use the original function-matching decision"
    )

    return parser.parse_args()


def instantiate_gnn(weights_path: str) -> GraphNetwork:
    """Instantiate the GNN model."""
    gnn_model = GraphNetwork(512)
    gnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    )

    dummy_adj_matrix = tf.constant(np.zeros((1, 1), dtype=np.float32))
    dummy_node_features = tf.constant(np.zeros((1, 128), dtype=np.float32))
    gnn_model(dummy_adj_matrix, dummy_node_features)

    gnn_model.load_weights(weights_path)
    return gnn_model


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
        f"readelf_fallback_count={code_unit.readelf_fallback_count}, "
        f"pseudo_block_fallback_count={code_unit.pseudo_block_fallback_count}"
    )


def filter_compilation_units(comp_units: list[CodeUnit]) -> tuple[list[CodeUnit], int]:
    """Drop single-function compilation units before matching."""
    eligible_comp_units = [cu for cu in comp_units if cu.get_num_functions() > 1]
    excluded_single_function_cu = len(comp_units) - len(eligible_comp_units)
    return eligible_comp_units, excluded_single_function_cu


def process_bin(binary: CodeUnit, lib: dict[str, list[CodeUnit]], args):
    """Process a parsed binary against all libraries."""
    output_file = args.output if args.output else None

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    # Perform function matching
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

        matches = match_functions(binary, eligible_comp_units)
        function_scores_by_cu = {
            id(match.target_unit): match.get_score()
            for match in matches
        }
        matches_by_cu = {
            id(match.target_unit): match
            for match in matches
        }
        function_best_score = max((m.get_score() for m in matches), default=0.0)
        block_best_score = 0.0
        block_results = []

        if args.disable_block_matching:
            candidate_matches = [
                m for m in matches if m.get_score() >= args.function_threshold
            ]
            successful_matches = candidate_matches
            if successful_matches:
                score_tot = sum(m.get_score() for m in successful_matches) / len(successful_matches)
            else:
                score_tot = 0.0
        else:
            if args.block_scope == "all_cu":
                candidate_matches = [
                    Match(source_unit=binary, target_unit=target_unit)
                    for target_unit in eligible_comp_units
                ]
                primary_candidate_count = sum(
                    1 for match in candidate_matches
                    if function_scores_by_cu.get(id(match.target_unit), 0.0) >= args.function_threshold
                )
                recovery_candidate_count = sum(
                    1 for match in candidate_matches
                    if (
                        args.candidate_low_threshold
                        <= function_scores_by_cu.get(id(match.target_unit), 0.0)
                        < args.function_threshold
                    )
                )
                for candidate_match in candidate_matches:
                    function_match = matches_by_cu.get(id(candidate_match.target_unit))
                    if function_match is not None:
                        block_result = evaluate_block_match(
                            function_match,
                            args.block_threshold,
                            args.block_assignment_threshold,
                            args.block_min_coverage_ratio,
                            args.block_coverage_mean_threshold,
                            args.block_min_instructions,
                            args.block_min_function_concentration,
                            args.block_min_function_spread,
                        )
                        if block_result.passed:
                            block_results.append((function_match, block_result))
                            continue

                    block_result = evaluate_block_presence(
                        binary,
                        candidate_match.target_unit,
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
                    )
                    block_results.append((candidate_match, block_result))
            else:
                candidate_matches = select_block_candidates(
                    matches,
                    args.function_threshold,
                    args.candidate_top_k,
                    args.candidate_low_threshold,
                )
                primary_candidate_count = sum(
                    1 for m in candidate_matches
                    if m.get_score() >= args.function_threshold
                )
                recovery_candidate_count = len(candidate_matches) - primary_candidate_count
                for candidate_match in candidate_matches:
                    block_result = evaluate_block_match(
                        candidate_match,
                        args.block_threshold,
                        args.block_assignment_threshold,
                        args.block_min_coverage_ratio,
                        args.block_coverage_mean_threshold,
                        args.block_min_instructions,
                        args.block_min_function_concentration,
                        args.block_min_function_spread,
                    )
                    block_results.append((candidate_match, block_result))

            successful_block_results = [
                (candidate_match, block_result)
                for candidate_match, block_result in block_results
                if block_result.passed
            ]
            successful_matches = [
                candidate_match
                for candidate_match, _ in successful_block_results
            ]
            block_best_score = max(
                (block_result.score for _, block_result in block_results),
                default=0.0,
            )
            if successful_block_results:
                score_tot = (
                    sum(block_result.score for _, block_result in successful_block_results)
                    / len(successful_block_results)
                )
            else:
                score_tot = 0.0

        stop_library = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))

        matched_cu = len(successful_matches)
        total_cu = len(eligible_comp_units)
        if args.disable_block_matching or args.block_scope == "function_candidates":
            matched_functions = sum(m.get_num_matched_functions() for m in successful_matches)
        else:
            matched_functions = sum(m.target_unit.get_num_functions() for m in successful_matches)
        percentage = score_tot * 100.0

        enough_cu = matched_cu >= min(args.min_cu, total_cu)
        if args.disable_block_matching or args.block_scope == "function_candidates":
            enough_functions = (
                len([m for m in successful_matches if m.get_num_matched_functions() >= args.min_functions_cu])
                >= min(args.min_cu, total_cu)
            )
        else:
            enough_functions = (
                len([m for m in successful_matches if m.target_unit.get_num_functions() >= args.min_functions_cu])
                >= min(args.min_cu, total_cu)
            )

        if enough_cu and enough_functions:
            status = "YES"
        elif enough_cu:
            status = "YES [W]"
        else:
            status = "NO"

        if args.disable_block_matching:
            line = (
                f"{status:7} | "
                f"library={Path(library_name).name} | "
                f"score={percentage:6.2f}% | "
                f"function_best={function_best_score * 100.0:6.2f}% | "
                f"matched_cu={matched_cu}/{total_cu} | "
                f"matched_functions={matched_functions} | "
                f"time={elapsed_time}"
            )

            for i, m in enumerate(successful_matches):
                avg_call_graph = (
                    float(np.mean([fp.features.call_graph_similarity for fp in m.matched_functions]))
                    if m.matched_functions else 0.0
                )
                avg_locality = (
                    float(np.mean([fp.features.function_locality_similarity for fp in m.matched_functions]))
                    if m.matched_functions else 0.0
                )
                log_line(
                    f"    CU[{i:03}] score={m.get_score():.4f} "
                    f"call_graph={avg_call_graph:.4f} "
                    f"locality={avg_locality:.4f} "
                    f"matched_functions={m.get_num_matched_functions()}",
                    output_file
                )
        else:
            line = (
                f"{status:7} | "
                f"library={Path(library_name).name} | "
                f"score={percentage:6.2f}% | "
                f"function_best={function_best_score * 100.0:6.2f}% | "
                f"block_best={block_best_score * 100.0:6.2f}% | "
                f"block_scope={args.block_scope} | "
                f"candidate_cu={len(candidate_matches)}/{total_cu} | "
                f"primary_candidate_cu={primary_candidate_count} | "
                f"recovery_candidate_cu={recovery_candidate_count} | "
                f"matched_cu={matched_cu}/{total_cu} | "
                f"matched_functions={matched_functions} | "
                f"time={elapsed_time}"
            )

            for i, (m, block_result) in enumerate(block_results):
                block_status = "PASS" if block_result.passed else "DROP"
                function_score = function_scores_by_cu.get(id(m.target_unit), m.get_score())
                if function_score >= args.function_threshold:
                    candidate_band = "primary"
                elif function_score >= args.candidate_low_threshold:
                    candidate_band = "recovery"
                else:
                    candidate_band = "all_cu"
                reported_functions = (
                    m.get_num_matched_functions()
                    if args.block_scope == "function_candidates"
                    else m.target_unit.get_num_functions()
                )
                log_line(
                    f"    BLOCK_CU[{i:03}] {block_status} "
                    f"band={candidate_band} "
                    f"function={function_score:.4f} "
                    f"block={block_result.score:.4f} "
                    f"coverage_mean={block_result.coverage_mean:.4f} "
                    f"coverage_min={block_result.coverage_min:.4f} "
                    f"coverage_ratio={block_result.coverage_ratio:.4f} "
                    f"assignment_mean={block_result.assignment_mean:.4f} "
                    f"assignment_min={block_result.assignment_min:.4f} "
                    f"blocks={block_result.num_source_blocks}/{block_result.num_target_blocks} "
                    f"locality_span={block_result.locality_span} "
                    f"edge_locality_ratio={block_result.edge_locality_ratio:.4f} "
                    f"function_concentration={block_result.function_concentration:.4f} "
                    f"function_spread={block_result.function_spread:.4f} "
                    f"matched_functions={reported_functions}",
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

    graph_model = instantiate_gnn(args.graph_model)

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
                        graph_model=graph_model,
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
        graph_model=graph_model,
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
