from model import DistanceMetric, PalmTree 
from struct2vec import GraphNetwork
import numpy as np
import tensorflow as tf
from enum import Enum
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction, Namespace
from pathlib import Path, PosixPath
import time
import signal
import sys
from typing import Optional, TextIO
from asm import Binary, parse_r2_file
from match import match_functions

class TimeoutException(Exception):
    """Exception raised when a timeout occurs"""
    pass

def handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Timeout occurred while processing the binary")

signal.signal(signal.SIGALRM, handler)


# Default values for command line arguments
ARCH = "intel"
LIBRARIES_DIR = Path(__file__).parent / "libraries"
LIBRARIES = "all"
EXCLUDE = None
KEEP_VERSIONS = "newest"
ASM_MODEL = Path(__file__).parent / "palmtree" / "model" / "transformer.ep19"
GRAPH_MODEL = Path(__file__).parent / "struct2vec" / "struct2vec.weights.h5"

THRESHOLD = 0.8 # match.SIMILARITY_THRESHOLD
MIN_CU = 1
MIN_FUNCTIONS_CU = 10
OUTPUT = None
VERBOSITY = 1
SORT_CU = "alpha"
COLORS = True

SPACES = "     "


class Color(Enum):
    """Terminal color codes"""

    BLACK: str = "\33[1;30m"
    RED: str = "\33[1;31m"
    GREEN: str = "\33[1;32m"
    YELLOW: str = "\33[1;33m"
    BLUE: str = "\33[1;34m"
    VIOLET: str = "\33[1;35m"
    BROWN: str = "\33[1;36m"
    WHITE: str = "\33[1;37m"
    DEFAULT: str = ""


def print_and_write(text: str = "", end: str = "\n", use_colors: bool = COLORS, color: Color = Color.DEFAULT, output_file: TextIO = OUTPUT):
    """Print text to stdout and write text to file"""

    if not use_colors:
        color = Color.DEFAULT

    # print(color.value + text + Color.DEFAULT.value, end=end, flush=True)

    if output_file:
        output_file.write(text + end)
        output_file.flush()


def resolve_output_path(output_arg: str, binary_path: Path, multiple_binaries: bool) -> Path:
    """Resolve the output path for a binary match report."""

    if "{binary}" in output_arg:
        resolved_path = Path(output_arg.format(binary=binary_path.name, binary_stem=binary_path.stem))
    else:
        candidate = Path(output_arg)

        if output_arg.endswith("/") or (candidate.exists() and candidate.is_dir()):
            resolved_path = candidate / f"{binary_path.name}.matches.txt"
        elif "file" in candidate.name:
            resolved_path = candidate.with_name(candidate.name.replace("file", binary_path.name))
        elif multiple_binaries:
            suffix = "".join(candidate.suffixes)
            stem = candidate.name[:-len(suffix)] if suffix else candidate.name
            resolved_path = candidate.with_name(f"{stem}.{binary_path.name}{suffix or '.txt'}")
        else:
            resolved_path = candidate

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def get_args() -> Namespace:
    """Get command line arguments"""

    # Initialize parser
    parser = ArgumentParser(
        prog="main",
        usage="\t%(prog)s [OPTIONS] FILE\n\t%(prog)s [--help]",
        description="Identify which libraries have been statically linked into a binary file.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False,
        allow_abbrev=False,
        exit_on_error=True
    )

    # Add mandatory arguments
    mandatory_arguments = parser.add_argument_group("mandatory arguments")
    mandatory_arguments.add_argument(
        "binaries",
        help="path of binary's ASM file",
        metavar="FILE",
        type=str
    )

    # Add binary options
    binary_options = parser.add_argument_group("binary options")
    binary_options.add_argument(
        "-a",
        "--asm",
        help="assembly representation [%(choices)s]",
        metavar="ASM",
        type=str,
        default=ARCH,
        choices=["intel", "att"]
    )

    # Add libraries options
    libraries_options = parser.add_argument_group("libraries options")
    libraries_options.add_argument(
        "-L",
        "--libraries-dir",
        help="path of directory containing libraries' ASM files",
        metavar="DIR",
        type=str,
        default=LIBRARIES_DIR
    )
    libraries_options.add_argument(
        "-l",
        "--libraries",
        help="comma-separated list of library names to be matched; a version number can be appended after each name, preceded by an underscore",
        metavar="LIST",
        type=str,
        default=LIBRARIES
    )
    libraries_options.add_argument(
        "-e",
        "--exclude",
        help="comma-separated list of library names not to be matched; a version number can be appended after each name, preceded by an underscore",
        metavar="LIST",
        type=str,
        default=EXCLUDE
    )
    libraries_options.add_argument(
        "-k",
        "--keep-versions",
        help="specify which versions of each library should be used; this is overridden by explicitly appended versions in '--libraries' option [%(choices)s]",
        metavar="STR",
        type=str,
        default=KEEP_VERSIONS,
        choices=["all", "newest", "oldest"]
    )

    # Add model options
    asm_model_options = parser.add_argument_group("model options")
    asm_model_options.add_argument(
        "-am",
        "--asm_model",
        help="path of word embedding model file",
        metavar="FILE",
        type=str,
        default=ASM_MODEL
    )

    # Add model options
    model_options = parser.add_argument_group("model options")
    model_options.add_argument(
        "-gm",
        "--graph_model",
        help="path of graph embedding model file",
        metavar="FILE",
        type=str,
        default=GRAPH_MODEL
    )

    # Add match options
    match_options = parser.add_argument_group("match options")
    match_options.add_argument(
        "-t",
        "--threshold",
        help="threshold value to consider a match successful",
        metavar="THRESHOLD",
        type=float,
        default=THRESHOLD
    )
    match_options.add_argument(
        "-C",
        "--min-cu",
        help="minimum number of successfully matched compilation units to consider a library match successful",
        metavar="N",
        type=int,
        default=MIN_CU
    )
    match_options.add_argument(
        "-c",
        "--min-functions-cu",
        help="minimum number of functions inside a compilation unit to consider its match reliable",
        metavar="N",
        type=int,
        default=MIN_FUNCTIONS_CU
    )

    # Add output options
    output_options = parser.add_argument_group("output options")
    output_options.add_argument(
        "-o",
        "--output",
        help="path of output file",
        metavar="FILE",
        type=str,
        default=OUTPUT
    )
    output_options.add_argument(
        "-v",
        "--verbosity",
        help="output's verbosity level [%(choices)s]",
        metavar="LEVEL",
        type=int,
        default=VERBOSITY,
        choices=[0, 1, 2]
    )
    output_options.add_argument(
        "-s",
        "--sort-cu",
        help="sort compilation units [%(choices)s]",
        metavar="KEY",
        type=str,
        default=SORT_CU,
        choices=["alpha", "score"]
    )
    output_options.add_argument(
        "--colors",
        help="activate/deactivate colored output",
        default=COLORS,
        action=BooleanOptionalAction
    )

    # Add other options
    other_options = parser.add_argument_group("other options")
    other_options.add_argument(
        "-h",
        "--help",
        help="show this help message and exit",
        action="help"
    )

    # Parse command line arguments
    args = parser.parse_args()

    return args


def instantiateGNN(weights_path) -> GraphNetwork:
    """Instantiate the GNN model"""

    # Create the GNN model
    gnn_model = GraphNetwork(512)
    gnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    # Build the model by passing a dummy sample input
    dummy_adj_matrix = tf.constant(np.zeros((1, 1), dtype=np.float32))
    dummy_node_features = tf.constant(np.zeros((1, 128), dtype=np.float32))
    gnn_model(dummy_adj_matrix, dummy_node_features)

    # Load the model weights
    gnn_model.load_weights(weights_path)

    return gnn_model


def process_bin(ins = tuple[PosixPath, dict[str, list[Binary]], Namespace, Optional[Path], bool]):
    bin, lib, args, requested_output, multiple_binaries = ins

    # Check output
    output_file = OUTPUT
    if requested_output:
        out = resolve_output_path(requested_output.as_posix(), bin, multiple_binaries)

        if out.exists():
            with open(out, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                if any((l for l in lines if "DONE" in l)):
                    print_and_write(f"Output file '{out}' already exists, using cached results", color=Color.YELLOW, use_colors=args.colors)
                    return
    
        output_file = open(out, "w", encoding='utf-8')

    # Load binary's file
    print_and_write(f"Binary:\n\t{bin.name}\n".expandtabs(4), use_colors=args.colors, output_file=output_file)
    binary = [parse_r2_file(bin.as_posix(), asm_model=asm_model, graph_model=graph_model)]

    if not binary[0]:
        print_and_write(f"Skipping binary '{bin.name}' due to parsing error", color=Color.RED, use_colors=args.colors, output_file=output_file)
        if output_file:
            output_file.close()
        return

    # perform function matching
    print_and_write("Libraries:", use_colors=args.colors, output_file=output_file)
    for library_dir, comp_units in lib.items():
        num_comp_units = len(comp_units)
        if num_comp_units == 0:
            continue

        # Start timer for current library
        start_library = time.time()

        if args.verbosity > 1:
            print_and_write(f"\t{library_dir:<110}", end=" ... ", use_colors=args.colors, output_file=output_file)
        else:
            print_and_write(f"\t{library_dir:<50}", end=" ... ", use_colors=args.colors, output_file=output_file)

        # Match functions
        matches = match_functions(binary, comp_units)
        successful_matches = list(filter(lambda x: x.get_score() >= args.threshold, matches))
        if len(successful_matches) == 0:
            score_tot = 0
        else:
            score_tot = sum(m.get_score() for m in successful_matches) / len(successful_matches)

        # Stop timer for current library
        stop_library = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(stop_library - start_library))

        # Print output with verbosity level 0
        stats = f"[{score_tot:.2f}] [{len(successful_matches):>3}/{len(comp_units):>3}] [{sum(m.get_num_matched_functions() for m in successful_matches):>4}] [{elapsed_time}]"
        if len(successful_matches) >= min(args.min_cu, num_comp_units):
            if len(list(filter(lambda x: x.get_num_matched_functions() >= args.min_functions_cu, successful_matches))) >= min(args.min_cu, num_comp_units):
                print_and_write(f"{'YES':>3}"+SPACES+stats, color=Color.GREEN, use_colors=args.colors, output_file=output_file)
            else:
                print_and_write(F"{'YES':>3}  [W] "+stats, color=Color.YELLOW, use_colors=args.colors, output_file=output_file)
        else:
            print_and_write(f"{'NO':>3}"+SPACES+stats, color=Color.RED, use_colors=args.colors, output_file=output_file)

        # Print output with verbosity level 1
        if args.verbosity > 0:
            if args.sort_cu == "score":
                matches = sorted(matches, key=lambda x: x.get_score(), reverse=True)

            for m in matches:
                if args.verbosity > 1:
                    print_and_write(f"\t\t{m.target_bin.name:<90}", end=" ... ", use_colors=args.colors, output_file=output_file)
                else:
                    print_and_write(f"\t\t{m.target_bin.name:<50}", end=" ... ", use_colors=args.colors, output_file=output_file)

                score = f"[{m.get_score():.2f}]"
                matched_f = f"[{m.get_num_matched_functions():>4}]"
                if m.get_score() >= args.threshold:
                    if m.get_num_matched_functions() >= args.min_functions_cu:
                        print_and_write(SPACES+score+f"[{1:>4}]"+matched_f, color=Color.GREEN, use_colors=args.colors, output_file=output_file)
                    else:
                        print_and_write("[W] "+score+f"[{1:>4}]"+matched_f, color=Color.YELLOW, use_colors=args.colors, output_file=output_file)
                else:
                    print_and_write(SPACES+score+f"[{0:>4}]"+matched_f, color=Color.RED, use_colors=args.colors, output_file=output_file)

                # Print output with verbosity level 2
                if args.verbosity > 1:
                    for function_pair in m.matched_functions:
                        print_and_write(f"\t\t\t{str(function_pair):<90}", end=" ... ", use_colors=args.colors, output_file=output_file)
                        if function_pair.features.similarity >= args.threshold:
                            print_and_write(f"    [{function_pair.features.similarity:.2f}]", color=Color.GREEN, use_colors=args.colors, output_file=output_file)
                        else:
                            print_and_write(f"    [{function_pair.features.similarity:.2f}]", color=Color.RED, use_colors=args.colors, output_file=output_file)

            print_and_write(use_colors=args.colors, output_file=output_file)

        del successful_matches
        del matches
        del comp_units

    if output_file:
        print_and_write("\nDONE PROCESSING\n",use_colors=args.colors, output_file=output_file)
        output_file.close()
    


if __name__ == "__main__":
    """Main function"""

    # Start timer for whole execution
    start_main = time.time()

    # Do not show the traceback when an exception is raised
    # sys.tracebacklimit = 0

    # Get command line arguments
    args = get_args()

    # Check libraries dir
    libraries_dir: Path = Path(args.libraries_dir)
    if not libraries_dir.exists():
        raise ValueError(f"Can't find '{libraries_dir}'")
    if not libraries_dir.is_dir():
        raise ValueError(f"Not a directory '{libraries_dir}'")

    # Check libraries
    libraries_subdir: list[str] = sorted([l.as_posix() for l in libraries_dir.iterdir() if l.is_dir()])
    if not libraries_subdir:
        raise ValueError(f"Directory '{libraries_dir}' contains no subdirectories")
    
    # filtering libraries
    if args.keep_versions == "newest":
        # keeping only the newest version
        for i in range(len(libraries_subdir) - 1, 0, -1):
            curr: str = libraries_subdir[i]
            curr_idx: int = curr.find("_")
            
            prev: str = libraries_subdir[i - 1]
            prev_idx: int = prev.find("_")

            if curr[:curr_idx] == prev[:prev_idx]:
                libraries_subdir.pop(i - 1)
    elif args.keep_versions == "oldest":
        # keeping only the oldest version
        for i in range(len(libraries_subdir) - 1, 0, -1):
            curr: str = libraries_subdir[i]
            curr_idx: int = curr.find("_")
            
            prev: str = libraries_subdir[i - 1]
            prev_idx: int = prev.find("_")

            if curr[:curr_idx] == prev[:prev_idx]:
                libraries_subdir.pop(i)


    # filtering libraries maintaining only the ones specified
    if args.libraries != "all":
        libraries_subdir = []
        for l in args.libraries.split(','):
            l = l.strip()
            libraries_subdir += [p.as_posix() for p in libraries_dir.iterdir() if l in p.name]
    
    if args.exclude:
        for library_to_exclude in args.exclude.split(","):
            library_to_exclude = library_to_exclude.strip()
            to_exclude = [p for p in libraries_subdir if library_to_exclude in p]
            for p in to_exclude:
                libraries_subdir.remove(p)

    if not libraries_subdir:
        raise ValueError("The specified options produced an empty list of libraries")


    # Load word embedding model
    asm_model = PalmTree("Palm Tree")
    asm_model.load(args.asm_model)

    # Load graph model
    graph_model = instantiateGNN(args.graph_model)

    # Load each library's compilation unit
    lib: dict[str, list[Binary]] = {}
    libraries_subdir.sort()
    
    for library_dir in libraries_subdir:

        library_path = Path(library_dir)
        if not library_path.is_dir() or not any(library_path.iterdir()):
            continue

        # Get library files
        library_files = sorted([library_file for library_file in library_path.iterdir()])

        # Load library files
        comp_units = []
        for library_file in library_files:
            try:
                b = parse_r2_file(library_file.as_posix(), asm_model, graph_model)
            except Exception as e:
                print_and_write(f"Skipping library '{library_file.name}' due to parsing error: {e}", color=Color.RED, use_colors=args.colors)
                continue

            comp_units.append(b)
        lib[library_dir] = comp_units

    binaries_path = Path(args.binaries)
    if binaries_path.is_file():
        binaries: list[PosixPath] = [binaries_path]
    elif binaries_path.is_dir():
        binaries = sorted([bin for bin in binaries_path.iterdir() if bin.is_file()])
    else:
        raise ValueError(f"Can't find binaries path '{binaries_path}'")

    print(f"Found {len(binaries)} binaries to process")
    print(f"Start processing binaries: {time.strftime('%H:%M:%S', time.gmtime())}\n")

    ## Check binaries with multiple processes in parallel
    for b in binaries:
        signal.alarm(3600*4)  # Set a timeout of 3 hours for each binary processing
        try:
            print(f"\t{b.name} ... ", end="", flush=True)
            process_bin((b, lib, args, Path(args.output) if args.output else None, len(binaries) > 1))
            print("DONE")
        except TimeoutException as e:
            print(f"timeout while processing binary '{b.name}': {e}")
        finally:
            signal.alarm(0)  

    print("Done processing binaries")
