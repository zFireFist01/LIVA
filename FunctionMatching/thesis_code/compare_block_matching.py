"""Compare an optimized function against a set of candidate functions by blocks.

This is the "method 1" experiment: instead of building a synthetic inlined
function, it matches basic-block embeddings from A' against the union of blocks
from A and the candidate callees.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def _configure_runtime_from_cli() -> None:
    if "--hide-warnings" not in sys.argv:
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    warnings.filterwarnings(
        "ignore",
        message=r"Skipping variable loading for optimizer 'adam'.*",
        category=UserWarning,
    )


_configure_runtime_from_cli()

from asm import Block, CodeUnit, Function, parse_r2_file
from model import PalmTree


SCRIPT_DIR = Path(__file__).resolve().parent


def format_address(address: int) -> str:
    return f"0x{address:x}"


def canonical_function_name(name: str) -> str:
    """Normalize common radare2/symbol prefixes for function-name matching."""
    normalized = name.strip()
    normalized = normalized.split("@@", maxsplit=1)[0]
    normalized = normalized.split("@", maxsplit=1)[0]

    for prefix in ("sym.imp.", "sym.", "fcn.", "imp."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    return normalized


def parse_function_names(raw_names: list[str]) -> list[str]:
    """Parse repeated or comma-separated function names from CLI-style input."""
    names: list[str] = []
    for raw_name in raw_names:
        names.extend(
            name.strip()
            for name in raw_name.split(",")
            if name.strip()
        )

    if not names:
        raise ValueError("At least one function name is required")

    return names


def function_stats(function: Function) -> dict[str, int | str]:
    return {
        "name": function.name,
        "blocks": function.get_num_blocks(),
        "instructions": function.get_num_instructions(),
    }


def find_function_by_name(code_unit: CodeUnit, requested_name: str) -> Function:
    """Find one function in a code unit by exact or canonicalized name."""
    exact_matches = [
        function
        for function in code_unit.functions
        if function.name == requested_name
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Multiple exact matches for '{requested_name}'")

    requested_canonical = canonical_function_name(requested_name)
    canonical_matches = [
        function
        for function in code_unit.functions
        if canonical_function_name(function.name) == requested_canonical
    ]
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        names = ", ".join(function.name for function in canonical_matches)
        raise ValueError(f"Multiple canonical matches for '{requested_name}': {names}")

    available = ", ".join(function.name for function in code_unit.functions)
    raise ValueError(
        f"Function '{requested_name}' not found in {code_unit.file_path}. "
        f"Available functions: {available}"
    )


def instantiate_graph_model(weights_path: str):
    """Instantiate and load the struct2vec graph model used by this experiment."""
    import tensorflow as tf
    from struct2vec import GraphNetwork

    gnn_model = GraphNetwork(512)
    gnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    dummy_adj_matrix = tf.constant(np.zeros((1, 1), dtype=np.float32))
    dummy_node_features = tf.constant(np.zeros((1, 128), dtype=np.float32))
    gnn_model(dummy_adj_matrix, dummy_node_features)

    gnn_model.load_weights(str(weights_path))
    return gnn_model


def embedding_similarity(left_embedding: np.ndarray, right_embedding: np.ndarray) -> float:
    """Return cosine similarity between two stored embedding vectors."""
    left = np.squeeze(left_embedding).reshape(1, -1)
    right = np.squeeze(right_embedding).reshape(1, -1)
    return float(cosine_similarity(left, right).item(0))


def compute_blocks_similarity_matrix(
    source_blocks: list[Block],
    target_blocks: list[Block],
) -> np.ndarray:
    """Compute cosine similarity between two lists of embedded basic blocks."""
    source_embeddings = np.stack([np.squeeze(block.embedding) for block in source_blocks])
    target_embeddings = np.stack([np.squeeze(block.embedding) for block in target_blocks])
    return cosine_similarity(source_embeddings, target_embeddings)


def best_match_per_source(sim_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each source row, return the best target index and similarity."""
    target_indices = np.argmax(sim_matrix, axis=1)
    values = sim_matrix[np.arange(sim_matrix.shape[0]), target_indices]
    return target_indices, values


def hungarian_assignment(sim_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return maximum-similarity linear assignment rows, cols, and values."""
    source_indices, target_indices = linear_sum_assignment(sim_matrix, maximize=True)
    values = sim_matrix[source_indices, target_indices]
    return source_indices, target_indices, values


def threshold_counts(values: list[float] | np.ndarray, thresholds: list[float]) -> dict[str, int]:
    """Count how many similarity values are above each threshold."""
    return {
        f">={threshold:.2f}": int(sum(1 for value in values if value >= threshold))
        for threshold in thresholds
    }


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    script_relative = (SCRIPT_DIR / candidate).resolve()
    if script_relative.exists():
        return script_relative

    return candidate.resolve()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_report(report_path: Path, report: dict) -> None:
    if report_path.parent:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True, default=json_default)
        report_file.write("\n")


def block_label(function: Function, block_address: int) -> str:
    return f"{function.name}:{format_address(block_address)}"


def block_records(functions: list[Function]) -> list[dict]:
    records: list[dict] = []

    for function in functions:
        for block in function.blocks:
            if block.embedding is None:
                continue

            records.append(
                {
                    "function": function.name,
                    "function_address": function.address,
                    "block_address": block.address,
                    "label": block_label(function, block.address),
                    "num_instructions": block.get_num_instructions(),
                    "block": block,
                    "instructions": list(block.instructions),
                }
            )

    return records


def safe_mean(values: list[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def safe_min(values: list[float] | np.ndarray) -> float:
    return float(np.min(values)) if len(values) else 0.0


def parse_thresholds(raw_thresholds: str) -> list[float]:
    thresholds = [
        float(part.strip())
        for part in raw_thresholds.split(",")
        if part.strip()
    ]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def best_coverage_by_optimized_block(
    sim_matrix: np.ndarray,
    optimized_blocks: list[dict],
    candidate_blocks: list[dict],
) -> tuple[list[dict], np.ndarray]:
    best_candidate_indices, best_values = best_match_per_source(sim_matrix)
    best_matches = []

    for optimized_index, (candidate_index, value) in enumerate(
        zip(best_candidate_indices, best_values)
    ):
        optimized = optimized_blocks[optimized_index]
        candidate = candidate_blocks[int(candidate_index)]
        best_matches.append(
            {
                "optimized_block": optimized["label"],
                "optimized_function": optimized["function"],
                "optimized_block_address": format_address(optimized["block_address"]),
                "candidate_block": candidate["label"],
                "candidate_function": candidate["function"],
                "candidate_block_address": format_address(candidate["block_address"]),
                "similarity": float(value),
            }
        )

    return best_matches, best_values


def assignment_matches(
    sim_matrix: np.ndarray,
    optimized_blocks: list[dict],
    candidate_blocks: list[dict],
) -> tuple[list[dict], np.ndarray]:
    optimized_indices, candidate_indices, values = hungarian_assignment(sim_matrix)
    assignment = []

    for optimized_index, candidate_index, value in zip(
        optimized_indices,
        candidate_indices,
        values,
    ):
        optimized = optimized_blocks[int(optimized_index)]
        candidate = candidate_blocks[int(candidate_index)]
        assignment.append(
            {
                "optimized_block": optimized["label"],
                "optimized_function": optimized["function"],
                "optimized_block_address": format_address(optimized["block_address"]),
                "candidate_block": candidate["label"],
                "candidate_function": candidate["function"],
                "candidate_block_address": format_address(candidate["block_address"]),
                "similarity": float(value),
            }
        )

    assignment.sort(key=lambda item: item["similarity"], reverse=True)
    return assignment, values


def candidate_function_coverage(
    sim_matrix: np.ndarray,
    optimized_blocks: list[dict],
    candidate_blocks: list[dict],
) -> dict[str, dict]:
    coverage: dict[str, dict] = {}

    candidate_best_values = np.max(sim_matrix, axis=0)
    for candidate, value in zip(candidate_blocks, candidate_best_values):
        bucket = coverage.setdefault(
            candidate["function"],
            {
                "num_candidate_blocks": 0,
                "best_block_similarities": [],
            },
        )
        bucket["num_candidate_blocks"] += 1
        bucket["best_block_similarities"].append(float(value))

    for bucket in coverage.values():
        values = bucket.pop("best_block_similarities")
        bucket["mean_best_similarity"] = safe_mean(values)
        bucket["min_best_similarity"] = safe_min(values)

    return coverage


def top_k_matches(
    sim_matrix: np.ndarray,
    optimized_blocks: list[dict],
    candidate_blocks: list[dict],
    top_k: int,
) -> list[dict]:
    records = []
    for optimized_index, optimized in enumerate(optimized_blocks):
        candidate_indices = np.argsort(sim_matrix[optimized_index])[::-1][:top_k]
        records.append(
            {
                "optimized_block": optimized["label"],
                "matches": [
                    {
                        "candidate_block": candidate_blocks[int(candidate_index)]["label"],
                        "candidate_function": candidate_blocks[int(candidate_index)]["function"],
                        "similarity": float(sim_matrix[optimized_index, candidate_index]),
                    }
                    for candidate_index in candidate_indices
                ],
            }
        )
    return records


def block_matching_report(
    optimized_function: Function,
    candidate_functions: list[Function],
    thresholds: list[float],
    top_k: int,
) -> dict:
    optimized_blocks = block_records([optimized_function])
    candidate_blocks = block_records(candidate_functions)

    if not optimized_blocks:
        raise ValueError(f"Optimized function '{optimized_function.name}' has no block embeddings")
    if not candidate_blocks:
        raise ValueError("Candidate functions have no block embeddings")

    sim_matrix = compute_blocks_similarity_matrix(
        [record["block"] for record in optimized_blocks],
        [record["block"] for record in candidate_blocks],
    )
    best_matches, best_values = best_coverage_by_optimized_block(
        sim_matrix,
        optimized_blocks,
        candidate_blocks,
    )
    assignment, assignment_values = assignment_matches(
        sim_matrix,
        optimized_blocks,
        candidate_blocks,
    )

    return {
        "optimized_function": function_stats(optimized_function),
        "candidate_functions": [
            function_stats(function)
            for function in candidate_functions
        ],
        "num_optimized_blocks": len(optimized_blocks),
        "num_candidate_blocks": len(candidate_blocks),
        "coverage_optimized_to_candidates": {
            "mean_best_similarity": safe_mean(best_values),
            "min_best_similarity": safe_min(best_values),
            "threshold_counts": threshold_counts(best_values, thresholds),
            "matches": best_matches,
            "top_k_matches": top_k_matches(
                sim_matrix,
                optimized_blocks,
                candidate_blocks,
                top_k,
            ),
        },
        "hungarian_assignment": {
            "num_matches": len(assignment),
            "mean_similarity": safe_mean(assignment_values),
            "min_similarity": safe_min(assignment_values),
            "threshold_counts": threshold_counts(assignment_values, thresholds),
            "matches": assignment,
        },
        "candidate_function_coverage": candidate_function_coverage(
            sim_matrix,
            optimized_blocks,
            candidate_blocks,
        ),
    }


def function_level_baselines(
    optimized_function: Function,
    caller_function: Function,
    callee_functions: list[Function],
) -> dict[str, float]:
    baselines = {
        "optimized_vs_original_caller": embedding_similarity(
            optimized_function.embedding,
            caller_function.embedding,
        ),
    }

    for callee_function in callee_functions:
        baselines[
            f"optimized_vs_callee_{canonical_function_name(callee_function.name)}"
        ] = embedding_similarity(optimized_function.embedding, callee_function.embedding)

    return baselines


def print_block_matching_report(report: dict) -> None:
    optimized = report["optimized_function"]
    candidates = report["candidate_functions"]
    coverage = report["coverage_optimized_to_candidates"]
    assignment = report["hungarian_assignment"]

    print("\nBlock-level matching")
    print(
        "  functions:        "
        f"optimized={optimized['name']}, "
        f"candidates={', '.join(candidate['name'] for candidate in candidates)}"
    )
    print(
        "  block counts:     "
        f"optimized={report['num_optimized_blocks']}, "
        f"candidates={report['num_candidate_blocks']}"
    )
    print("  optimized-block coverage:")
    print(f"    mean_best_similarity: {coverage['mean_best_similarity']:.6f}")
    print(f"    min_best_similarity:  {coverage['min_best_similarity']:.6f}")
    print(f"    threshold_counts:     {coverage['threshold_counts']}")
    print("  hungarian assignment:")
    print(f"    num_matches:          {assignment['num_matches']}")
    print(f"    mean_similarity:      {assignment['mean_similarity']:.6f}")
    print(f"    min_similarity:       {assignment['min_similarity']:.6f}")
    print(f"    threshold_counts:     {assignment['threshold_counts']}")
    print("  candidate-function coverage:")
    for name, stats in report["candidate_function_coverage"].items():
        print(
            f"    {name}: blocks={stats['num_candidate_blocks']}, "
            f"mean_best={stats['mean_best_similarity']:.6f}, "
            f"min_best={stats['min_best_similarity']:.6f}"
        )

    print("  best optimized-block matches:")
    best_matches = sorted(
        coverage["matches"],
        key=lambda match_record: match_record["similarity"],
        reverse=True,
    )
    for match_record in best_matches[:20]:
        print(
            f"    {match_record['optimized_block']} -> "
            f"{match_record['candidate_block']} "
            f"({match_record['similarity']:.6f})"
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match optimized blocks against A+B/C/D candidate blocks.",
    )
    parser.add_argument("--optimized-object", required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--optimized-function", required=True)
    parser.add_argument("--caller-function", required=True)
    parser.add_argument(
        "--callee-function",
        action="append",
        required=True,
        help="Repeat or pass comma-separated callee names.",
    )
    parser.add_argument(
        "--asm-model",
        default="palmtree/model/transformer.ep19",
        help="Path to the PalmTree model.",
    )
    parser.add_argument(
        "--graph-model",
        default="struct2vec/struct2vec.weights.h5",
        help="Path to struct2vec weights, used for optional function-level baselines.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidate blocks to retain per optimized block.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.80",
        help="Comma-separated similarity thresholds for coverage counts.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--hide-warnings",
        action="store_true",
        help="Hide non-critical TensorFlow/Keras startup warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    optimized_object = resolve_path(args.optimized_object)
    source_object = resolve_path(args.source_object)
    asm_model_path = resolve_path(args.asm_model)
    graph_model_path = resolve_path(args.graph_model)
    callee_names = parse_function_names(args.callee_function)
    thresholds = parse_thresholds(args.thresholds)

    for path in (optimized_object, source_object, asm_model_path, graph_model_path):
        if not path.exists():
            raise FileNotFoundError(f"Can't find '{path}'")

    asm_model = PalmTree("Palm Tree")
    asm_model.load(str(asm_model_path))
    graph_model = instantiate_graph_model(graph_model_path.as_posix())

    optimized_unit = parse_r2_file(
        optimized_object.as_posix(),
        asm_model=asm_model,
        graph_model=graph_model,
        unit_type=CodeUnit.TYPE_CU,
    )
    source_unit = parse_r2_file(
        source_object.as_posix(),
        asm_model=asm_model,
        graph_model=graph_model,
        unit_type=CodeUnit.TYPE_CU,
    )

    optimized_function = find_function_by_name(optimized_unit, args.optimized_function)
    caller_function = find_function_by_name(source_unit, args.caller_function)
    callee_functions = [
        find_function_by_name(source_unit, callee_name)
        for callee_name in callee_names
    ]
    candidate_functions = [caller_function, *callee_functions]

    report = block_matching_report(
        optimized_function=optimized_function,
        candidate_functions=candidate_functions,
        thresholds=thresholds,
        top_k=args.top_k,
    )
    report["function_level_baselines"] = function_level_baselines(
        optimized_function,
        caller_function,
        callee_functions,
    )

    print_block_matching_report(report)
    print("\nFunction-level baselines:")
    for name, value in sorted(
        report["function_level_baselines"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {name}: {value:.6f}")

    if args.output_json:
        write_json_report(resolve_path(args.output_json), report)


if __name__ == "__main__":
    main()
