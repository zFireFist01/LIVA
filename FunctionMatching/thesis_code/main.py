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
from match import match_functions

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("TensorFlow GPUs:", gpus)


PATH_TO_CALCULATOR = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..", "..", "..",
        "Thesis_Binary_Analysis",
        "Exploration", "easy", "gcc11", "calculator_static"
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
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold"
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
        successful_matches = [m for m in matches if m.get_score() >= args.threshold]

        if successful_matches:
            score_tot = sum(m.get_score() for m in successful_matches) / len(successful_matches)
        else:
            score_tot = 0.0

        stop_library = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))

        matched_cu = len(successful_matches)
        total_cu = len(eligible_comp_units)
        matched_functions = sum(m.get_num_matched_functions() for m in successful_matches)
        percentage = score_tot * 100.0

        enough_cu = matched_cu >= min(args.min_cu, total_cu)
        enough_functions = (
            len([m for m in successful_matches if m.get_num_matched_functions() >= args.min_functions_cu])
            >= min(args.min_cu, total_cu)
        )

        if enough_cu and enough_functions:
            status = "YES"
        elif enough_cu:
            status = "YES [W]"
        else:
            status = "NO"

        line = (
            f"{status:7} | "
            f"library={Path(library_name).name} | "
            f"score={percentage:6.2f}% | "
            f"matched_cu={matched_cu}/{total_cu} | "
            f"matched_functions={matched_functions} | "
            f"time={elapsed_time}"
        )

        for i, m in enumerate(successful_matches):
            log_line(
                f"    CU[{i:03}] score={m.get_score():.4f} matched_functions={m.get_num_matched_functions()}",
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
