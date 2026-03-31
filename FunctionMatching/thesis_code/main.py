import time
import os
import argparse
from pathlib import Path
import signal
import re
import numpy as np
import tensorflow as tf
import tempfile
import subprocess

from model import PalmTree
from struct2vec import GraphNetwork
from asm import Binary, parse_r2_file
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


def process_bin(binary_path: Path, lib: dict[str, list[Binary]], args, asm_model, graph_model):
    """Process a single binary against all libraries."""
    output_file = args.output if args.output else None

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    binary = [parse_r2_file(binary_path.as_posix(), asm_model=asm_model, graph_model=graph_model)]

    # Perform function matching
    for library_name, comp_units in lib.items():
        start_library = time.time()

        matches = match_functions(binary, comp_units)
        successful_matches = [m for m in matches if m.get_score() >= args.threshold]

        if successful_matches:
            score_tot = sum(m.get_score() for m in successful_matches) / len(successful_matches)
        else:
            score_tot = 0.0

        stop_library = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))

        matched_cu = len(successful_matches)
        total_cu = len(comp_units)
        matched_functions = sum(m.get_num_matched_functions() for m in successful_matches)
        percentage = score_tot * 100.0

        if total_cu == 0:
            line = (
                f"{'NO':7} | "
                f"library={Path(library_name).name} | "
                f"score={percentage:6.2f}% | "
                f"matched_cu={matched_cu}/{total_cu} | "
                f"matched_functions={matched_functions} | "
                f"time={elapsed_time} | "
                f"empty library parsing"
            )
            log_line(line, output_file)
            continue

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

    lib: dict[str, list[Binary]] = {}
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
                    b = parse_r2_file(obj_file.as_posix(), asm_model, graph_model)
                    if b.get_num_functions() > 0:
                        comp_units.append(b)
                except Exception as e:
                    print(f"[WARN] Failed to parse {obj_file}: {e}")

        lib[library_file.as_posix()] = comp_units
        print(f"[DEBUG] {library_file.name}: loaded {len(comp_units)} object files")

    log_line(f"Found {len(library_files)} libraries to process", args.output)
    log_line(f"Target binary: {binary_path}", args.output)
    log_line(f"Start processing: {time.strftime('%H:%M:%S', time.gmtime())}\n", args.output)

    signal.alarm(3600 * 4)
    try:
        process_bin(binary_path, lib, args, asm_model, graph_model)
    except TimeoutException as e:
        log_line(f"Timeout while processing binary '{binary_path.name}': {e}", args.output)
    finally:
        signal.alarm(0)

    elapsed_main = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_main))
    log_line(f"\nDone processing in {elapsed_main}", args.output)