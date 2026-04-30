import typing 


import numpy as np
from scipy.optimize import linear_sum_assignment

import asm
import model

# Similarity score modifiers
MODIFIER_GLOBAL_STRINGS = 0.05
MODIFIER_CALLED_FUNCTIONS = 0.05
MODIFIER_NUM_BLOCKS = 0.05
MODIFIER_NUM_ARGUMENTS = 0.3
MODIFIER_RETURN_TYPE = 0.3

# Matched functions ratio threshold
FUNCTIONS_RATIO_THRESHOLD = 0.2

# Similarity score threshold
SIMILARITY_THRESHOLD = 0.4

# Call-graph score modifier. The value is intentionally small: embeddings still
# drive the match, while call edges help choose between plausible windows.
MODIFIER_CALL_GRAPH = 0.15


class FeatureVector(typing.NamedTuple):
    """Vector of features between two `asm.Function` objects"""

    similarity: float
    call_graph_similarity: float = 0.0


class FunctionPair(typing.NamedTuple):
    """Pair of `asm.Function` objects with the corresponding `FeatureVector` object"""

    source_function: asm.Function
    target_function: asm.Function
    features: FeatureVector

    def __str__(self) -> str:
        return f"{self.source_function.name:<50} ---> {self.target_function.name:>50}"


class Match:
    """Match between two `asm.CodeUnit` objects"""

    def __init__(self, source_unit: asm.CodeUnit = None, target_unit: asm.CodeUnit = None, matched_functions: list[FunctionPair] = None):
        self.source_unit = source_unit  # Source ELF code unit (the executable to analyze)
        self.target_unit = target_unit  # Target CU code unit (the library compilation unit)
        self.matched_functions = matched_functions or []

    def __repr__(self) -> str:
        return "<{}.{}; source_unit={!r}, target_unit={!r}>".format(
            __name__, type(self).__name__, self.source_unit, self.target_unit
        )

    def __str__(self) -> str:
        return "{}\n\n{}\n\n{}\n\n{}".format(
            "Source file:\n\t{}".format(self.source_unit.file_path),
            "Target file:\n\t{}".format(self.target_unit.file_path),
            "Matched functions:\n\t{}".format(
                "\n\t".join(str(function_pair) for function_pair in self.matched_functions)
            ),
            "Match score:\n\t{:.2f}".format(self.get_score()),
        )

    def get_num_matched_functions(self) -> int:
        return len(self.matched_functions)

    def get_similarity_tot(self) -> float:
        return round(
            float(sum(fp.features.similarity for fp in self.matched_functions)), 2
        )

    def get_score(self) -> float:
        successful_functions = [
            fp for fp in self.matched_functions
            if fp.features.similarity >= SIMILARITY_THRESHOLD
        ]

        if len(successful_functions) == 0:
            score = 0
        elif len(successful_functions) / self.get_num_matched_functions() >= FUNCTIONS_RATIO_THRESHOLD:
            score = sum(fp.features.similarity for fp in successful_functions) / len(successful_functions)
        else:
            score = 0

        return round(float(score), 2)


def _internal_call_edges(functions: list[asm.Function]) -> set[tuple[int, int]]:
    """Return call edges whose caller and callee are both in `functions`."""
    addresses = {function.address for function in functions}
    edges = set()

    for caller in functions:
        for callee_addr in caller.resolved_call_targets:
            if callee_addr in addresses:
                edges.add((caller.address, callee_addr))

    return edges


def _call_graph_assignment_score(
    source_functions: list[asm.Function],
    target_functions: list[asm.Function],
    target_to_source: dict[int, int],
) -> float:
    """Score whether matched functions preserve the target CU call edges.

    The score uses only numeric call targets resolved to local function addresses.
    Symbols and function names are not consulted.
    """
    target_edges = _internal_call_edges(target_functions)
    if not target_edges:
        return 0.0

    source_edges = _internal_call_edges(source_functions)

    matched_edges = 0
    for target_caller, target_callee in target_edges:
        source_caller = target_to_source.get(target_caller)
        source_callee = target_to_source.get(target_callee)

        if source_caller is None or source_callee is None:
            continue

        if (source_caller, source_callee) in source_edges:
            matched_edges += 1

    return matched_edges / len(target_edges)


def _combined_similarity(base_similarity: float, call_graph_similarity: float) -> float:
    return min(
        1.0,
        base_similarity + (MODIFIER_CALL_GRAPH * call_graph_similarity),
    )


def match_functions(source_bin: asm.CodeUnit, target_cu_list: list[asm.CodeUnit]) -> list[Match]:
    """Match assembly functions between a source binary and target compilation units.

    Args:
        source_bin: Source binary (executable to analyze).
        target_cu_list: List of target compilation units (library object files).

    Returns:
        List of Match objects representing successful matches.
    """

    matches = []
    source_functions = source_bin.functions
    num_source_functions = source_bin.get_num_functions()

    for target_cu in target_cu_list:
        num_target_cu_functions = target_cu.get_num_functions()
        if num_target_cu_functions == 0:
            continue

        matched_cu_windows = []
        for target_cu_functions in target_cu.blobs:
            num_target_window_functions = len(target_cu_functions)

            # Compute similarity matrix using assembly-level embeddings
            sim_matrix = model.compute_similarity_matrix(source_functions, target_cu_functions)
            print(
                f"[DEBUG] {target_cu.name}: "
                f"target_functions={num_target_window_functions}, max_sim={sim_matrix.max():.4f}"
            )
            source_peak_idx, target_peak_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)

            if sim_matrix[source_peak_idx][target_peak_idx] < SIMILARITY_THRESHOLD:
                # No match: max similarity is below the threshold for this target CU window.
                continue 

            # Extract all possible source-binary windows for this target CU window.
            source_candidate_windows = [
                slice(
                    source_peak_idx - target_offset,
                    source_peak_idx - target_offset + num_target_window_functions,
                )
                for target_offset in range(num_target_window_functions)
            ]
            source_candidate_windows = [
                window for window in source_candidate_windows
                if window.start >= 0 and window.stop <= num_source_functions
            ]
            if len(source_candidate_windows) == 0:
                continue

            # Extract one similarity sub-matrix for each source-binary window.
            source_window_matrices = [sim_matrix[window, :] for window in source_candidate_windows]

            # Perform linear sum assignment on each source-window matrix.
            source_to_target_assignments = [
                linear_sum_assignment(source_window_matrix, maximize=True)
                for source_window_matrix in source_window_matrices
            ]

            # Sum assigned similarities, including a call-graph topology bonus.
            window_similarity_sums = []
            window_call_graph_scores = []
            for (
                source_window,
                source_window_matrix,
                (source_rows, target_cols),
            ) in zip(source_candidate_windows, source_window_matrices, source_to_target_assignments):
                source_global_rows = np.arange(source_window.start, source_window.stop)[source_rows]
                target_to_source = {
                    target_cu_functions[target_idx].address: source_functions[source_idx].address
                    for source_idx, target_idx in zip(source_global_rows, target_cols)
                }
                call_graph_score = _call_graph_assignment_score(
                    source_functions,
                    target_cu_functions,
                    target_to_source,
                )
                assigned_similarities = [
                    _combined_similarity(
                        float(source_window_matrix[source_row][target_col]),
                        call_graph_score,
                    )
                    for source_row, target_col in zip(source_rows, target_cols)
                ]
                window_similarity_sums.append(sum(assigned_similarities))
                window_call_graph_scores.append(call_graph_score)

            # Pick the source-binary window with the best combined score.
            best_idx = np.argmax(window_similarity_sums)
            best_source_window = source_candidate_windows[best_idx]
            best_source_to_target_assignment = source_to_target_assignments[best_idx]
            best_sum_similarity = float(window_similarity_sums[best_idx])
            best_call_graph_score = float(window_call_graph_scores[best_idx])
            best_source_rows, best_target_cols = best_source_to_target_assignment
            best_num_assigned = len(best_source_rows)

            if best_num_assigned == 0:
                continue

            # Normalize by number of assignments so thresholding is CU-size independent.
            best_mean_similarity = best_sum_similarity / best_num_assigned

            if best_mean_similarity < (SIMILARITY_THRESHOLD - 0.1):
                # No match: best mean similarity is below the threshold
                continue

            # Reconstruct full indices from submatrix rows to global sim_matrix rows
            source_global_indices = np.arange(
                best_source_window.start,
                best_source_window.stop,
            )[best_source_rows]
            target_cu_indices = best_target_cols

            # Create function match pairs
            matched_functions = [
                FunctionPair(
                    source_functions[source_idx],
                    target_cu_functions[target_idx],
                    FeatureVector(
                        _combined_similarity(sim_matrix[source_idx][target_idx], best_call_graph_score),
                        best_call_graph_score,
                    ),
                )
                for source_idx, target_idx in zip(source_global_indices, target_cu_indices)
            ]
            good = sum(1 for fp in matched_functions if fp.features.similarity >= SIMILARITY_THRESHOLD)
            print(
                f"[DEBUG] {target_cu.name}: "
                f"best_source_window=({best_source_window.start},{best_source_window.stop}), "
                f"assigned={len(matched_functions)}, good={good}, "
                f"ratio={good/len(matched_functions):.3f}, "
                f"mean={best_mean_similarity:.4f}, "
                f"sum={best_sum_similarity:.4f}, "
                f"call_graph={best_call_graph_score:.4f}"
            )
            matched_cu_windows.append((best_mean_similarity, best_num_assigned, matched_functions))
            break  # only one target CU window can match; later windows are not evaluated

        if (
            len(matched_cu_windows) <= 0
            or sum(s * b for s, b, _ in matched_cu_windows) / sum(b for _, b, _ in matched_cu_windows)
            < SIMILARITY_THRESHOLD
        ):
            # No match: weighted average similarity is below the threshold
            continue

        matched = []
        for _, _, window_matched_functions in matched_cu_windows:
            matched.extend(window_matched_functions)
        matches.append(
            Match(source_unit=source_bin, target_unit=target_cu, matched_functions=matched)
        )

    return matches
