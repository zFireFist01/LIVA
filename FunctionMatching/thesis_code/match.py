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


class FeatureVector(typing.NamedTuple):
    """Vector of features between two `asm.Function` objects"""

    similarity: float


class FunctionPair(typing.NamedTuple):
    """Pair of `asm.Function` objects with the corresponding `FeatureVector` object"""

    source_function: asm.Function
    target_function: asm.Function
    features: FeatureVector

    def __str__(self) -> str:
        return f"{self.source_function.name:<50} ---> {self.target_function.name:>50}"


class Match:
    """Match between two `asm.Binary` objects"""

    def __init__(self, source_bin: asm.Binary = None, target_bin: asm.Binary = None, matched_functions: list[FunctionPair] = None):
        self.source_bin = source_bin    # Source binary (the executable to analyze)
        self.target_bin = target_bin    # Target binary (the library compilation unit)
        self.matched_functions = matched_functions or []

    def __repr__(self) -> str:
        return "<{}.{}; source_bin={!r}, target_bin={!r}>".format(
            __name__, type(self).__name__, self.source_bin, self.target_bin
        )

    def __str__(self) -> str:
        return "{}\n\n{}\n\n{}\n\n{}".format(
            "Source file:\n\t{}".format(self.source_bin.file_path),
            "Target file:\n\t{}".format(self.target_bin.file_path),
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


def match_functions(source_bin_list: list[asm.Binary], target_cu_list: list[asm.Binary]) -> list[Match]:
    """Match assembly functions between source binaries and target compilation units.

    Args:
        source_bin_list: List of source binaries (executables to analyze).
        target_cu_list: List of target compilation units (library object files).

    Returns:
        List of Match objects representing successful matches.
    """

    matches = []
    for source_bin in source_bin_list:
        n_bin_functions = source_bin.get_num_functions()

        for target_cu in target_cu_list:
            cu_fun_size = target_cu.get_num_functions()
            if cu_fun_size == 0:
                continue

            matched_blob = []
            for blob in target_cu.blobs:
                blob_size = len(blob)

                # Compute similarity matrix using assembly-level embeddings
                sim_matrix = model.compute_similarity_matrix(source_bin.functions, blob)
                print(f"[DEBUG] {target_cu.name}: blob_size={blob_size}, max_sim={sim_matrix.max():.4f}")
                bin_idx, cu_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)

                if sim_matrix[bin_idx][cu_idx] < SIMILARITY_THRESHOLD:
                    # No match: max similarity is below the threshold for this blob
                    break

                # Extract all possible matching windows per blob
                matching_windows = [
                    slice(bin_idx - i, bin_idx - i + blob_size)
                    for i in range(blob_size)
                ]
                matching_windows = [
                    w for w in matching_windows
                    if w.start >= 0 and w.stop <= n_bin_functions
                ]
                if len(matching_windows) == 0:
                    continue

                # Extract sub matrices for each matching window
                sub_matrices = [sim_matrix[w, :] for w in matching_windows]

                # Perform linear sum assignment on each sub matrix
                assignments = [
                    linear_sum_assignment(sub_matrix, maximize=True)
                    for sub_matrix in sub_matrices
                ]

                # Sum assigned similarities
                similarities = [
                    sub_matrix[rows, cols].sum()
                    for sub_matrix, (rows, cols) in zip(sub_matrices, assignments)
                ]

                # Get best submatrix index
                best_idx = np.argmax(similarities)
                best_window = matching_windows[best_idx]
                best_assignment = assignments[best_idx]

                if similarities[best_idx] < (SIMILARITY_THRESHOLD - 0.1):
                    # No match: best similarity sum is below the threshold
                    continue

                # Reconstruct full indices from submatrix rows to global sim_matrix rows
                global_row_indices = np.arange(best_window.start, best_window.stop)[best_assignment[0]]
                global_col_indices = best_assignment[1]

                # Create function match pairs
                matched_functions = [
                    FunctionPair(
                        source_bin.functions[i],
                        target_cu.functions[j],
                        FeatureVector(sim_matrix[i][j]),
                    )
                    for i, j in zip(global_row_indices, global_col_indices)
                ]
                good = sum(1 for fp in matched_functions if fp.features.similarity >= SIMILARITY_THRESHOLD)
                print(
                    f"[DEBUG] {target_cu.name}: best_window=({best_window.start},{best_window.stop}), "
                    f"assigned={len(matched_functions)}, good={good}, "
                    f"ratio={good/len(matched_functions):.3f}, "
                    f"sum={similarities[best_idx]:.4f}"
                )
                matched_blob.append((similarities[best_idx], blob_size, matched_functions))
                break  # only one blob can match

            if (
                len(matched_blob) <= 0
                or sum(s * b for s, b, _ in matched_blob) / sum(b for _, b, _ in matched_blob)
                < SIMILARITY_THRESHOLD
            ):
                # No match: weighted average similarity is below the threshold
                continue

            matched = []
            for _, _, functions in matched_blob:
                matched.extend(functions)
            matches.append(
                Match(source_bin=source_bin, target_bin=target_cu, matched_functions=matched)
            )

    return matches