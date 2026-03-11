from model import DistanceMetric, PalmTree 
from struct2vec import GraphNetwork


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction, Namespace
from pathlib import Path, PosixPath
import time
import signal
import sys

def get_args():
    return 


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
        raise ValueError(f"Directory '{libraries_subdir}' is empty")
    
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
                b = parse_angr_file(library_file.as_posix(), asm_model, graph_model)
            except angr.errors.AngrCFGError as e:
                print_and_write(f"Skipping library '{library_file.name}' due to CFG error: {e}", color=Color.RED, use_colors=args.colors)
                continue
            except angr.errors.SimEngineError as e1:
                print_and_write(f"Skipping library '{library_file.name}' due to SimProcedure error: {e1}", color=Color.RED, use_colors=args.colors)
                continue

            comp_units.append(b)
        lib[library_dir] = comp_units

    binaries: list[PosixPath] = sorted([bin for bin in Path(args.binaries).iterdir() if bin.is_file()])

    print(f"Found {len(binaries)} binaries to process")
    print(f"Start processing binaries: {time.strftime('%H:%M:%S', time.gmtime())}\n")

    ## Check binaries with multiple processes in parallel
    for b in binaries:
        signal.alarm(3600*4)  # Set a timeout of 3 hours for each binary processing
        try:
            print(f"\t{b.name} ... ", end="", flush=True)
            process_bin((b, lib, args))
            print("DONE")
        except TimeoutException as e:
            print(f"timeout while processing binary '{b.name}': {e}")
        finally:
            signal.alarm(0)  

    print("Done processing binaries")
