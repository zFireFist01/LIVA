import typing 


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

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

# Locality modifier for the standard function matcher. This is weaker than the
# exact call-graph bonus: it rewards target internal call edges whose matched
# source functions remain close in the executable layout.
MODIFIER_FUNCTION_LOCALITY = 0.08
FUNCTION_LOCALITY_WINDOW_MULTIPLIER = 3.0
FUNCTION_LOCALITY_WINDOW_PADDING = 2
DEFAULT_MIN_BLOCK_INSTRUCTIONS = 3


class FeatureVector(typing.NamedTuple):
    """Vector of features between two `asm.Function` objects"""

    similarity: float
    call_graph_similarity: float = 0.0
    function_locality_similarity: float = 0.0


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


class BlockMatchResult(typing.NamedTuple):
    """Result of a block-level refinement for one CU match."""

    score: float
    coverage_mean: float
    coverage_min: float
    coverage_ratio: float
    assignment_mean: float
    assignment_min: float
    num_source_blocks: int
    num_target_blocks: int
    passed: bool
    locality_span: int = 0
    edge_locality_ratio: float = 1.0
    function_concentration: float = 1.0
    function_spread: float = 1.0


def safe_mean(values: list[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def safe_min(values: list[float] | np.ndarray) -> float:
    return float(np.min(values)) if len(values) else 0.0


def block_records(
    functions: list[asm.Function],
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
) -> list[dict]:
    records: list[dict] = []

    for function_index, function in enumerate(functions):
        for block in function.blocks:
            if block.embedding is None:
                continue
            if block.get_num_instructions() < min_block_instructions:
                continue

            records.append(
                {
                    "function": function.name,
                    "function_address": function.address,
                    "function_index": function_index,
                    "block_address": block.address,
                    "block": block,
                }
            )

    return records


def indexed_block_records(
    functions: list[asm.Function],
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
) -> tuple[list[dict], list[tuple[int, int]]]:
    records: list[dict] = []
    spans: list[tuple[int, int]] = []

    for function_index, function in enumerate(functions):
        start = len(records)
        for block in function.blocks:
            if block.embedding is None:
                continue
            if block.get_num_instructions() < min_block_instructions:
                continue

            records.append(
                {
                    "function": function.name,
                    "function_address": function.address,
                    "function_index": function_index,
                    "block_address": block.address,
                    "block": block,
                }
            )
        spans.append((start, len(records)))

    return records, spans


def compute_blocks_similarity_matrix(
    source_blocks: list[asm.Block],
    target_blocks: list[asm.Block],
) -> np.ndarray:
    """Compute cosine similarity between two lists of embedded basic blocks."""
    source_embeddings = np.stack([np.squeeze(block.embedding) for block in source_blocks])
    target_embeddings = np.stack([np.squeeze(block.embedding) for block in target_blocks])
    return cosine_similarity(source_embeddings, target_embeddings)


def best_coverage_by_source_block(sim_matrix: np.ndarray) -> np.ndarray:
    """For each source block, return the best target-block similarity."""
    target_indices = np.argmax(sim_matrix, axis=1)
    return sim_matrix[np.arange(sim_matrix.shape[0]), target_indices]


def assignment_values(sim_matrix: np.ndarray) -> np.ndarray:
    """Return maximum-similarity linear-assignment values."""
    source_indices, target_indices = linear_sum_assignment(sim_matrix, maximize=True)
    return sim_matrix[source_indices, target_indices]


def internal_call_edge_locality_ratio(
    target_functions: list[asm.Function],
    source_function_by_target: dict[int, int],
    locality_multiplier: float,
    locality_padding: int,
    empty_score: float = 1.0,
) -> float:
    """Return how many internal target call edges stay local after matching."""
    target_index_by_address = {
        function.address: index
        for index, function in enumerate(target_functions)
    }
    edge_results = []

    for caller_index, caller in enumerate(target_functions):
        source_caller_index = source_function_by_target.get(caller_index)
        if source_caller_index is None:
            continue

        for callee_address in caller.resolved_call_targets:
            callee_index = target_index_by_address.get(callee_address)
            if callee_index is None:
                continue

            source_callee_index = source_function_by_target.get(callee_index)
            if source_callee_index is None:
                continue

            target_distance = abs(caller_index - callee_index)
            max_source_distance = max(
                1,
                int(np.ceil((target_distance + 1) * locality_multiplier)) + locality_padding,
            )
            edge_results.append(
                abs(source_caller_index - source_callee_index) <= max_source_distance
            )

    if not edge_results:
        return empty_score

    return float(np.mean(edge_results))


def function_concentration_scores(
    target_block_records: list[dict],
    source_block_records: list[dict],
    matched_source_block_indices: np.ndarray,
) -> tuple[float, float, dict[int, int]]:
    """Measure whether blocks from each target function stay in one source function.

    Returns:
        concentration: weighted ratio of target blocks explained by each target
            function's dominant source function.
        spread: ratio of distinct dominant source functions to mapped target
            functions. Low values mean many target functions collapse together.
        source_function_by_target: dominant source function index per target
            function, useful for locality checks.
    """
    source_indices_by_target_function: dict[int, list[int]] = {}
    for target_block_index, source_block_index in enumerate(matched_source_block_indices):
        target_function_index = target_block_records[target_block_index]["function_index"]
        source_function_index = source_block_records[int(source_block_index)]["function_index"]
        source_indices_by_target_function.setdefault(target_function_index, []).append(
            source_function_index
        )

    if not source_indices_by_target_function:
        return 0.0, 0.0, {}

    dominant_total = 0
    block_total = 0
    source_function_by_target: dict[int, int] = {}

    for target_function_index, source_indices in source_indices_by_target_function.items():
        values, counts = np.unique(source_indices, return_counts=True)
        dominant_position = int(np.argmax(counts))
        dominant_source_index = int(values[dominant_position])
        dominant_count = int(counts[dominant_position])

        source_function_by_target[target_function_index] = dominant_source_index
        dominant_total += dominant_count
        block_total += len(source_indices)

    concentration = dominant_total / block_total if block_total else 0.0
    mapped_target_functions = len(source_function_by_target)
    spread = (
        len(set(source_function_by_target.values())) / mapped_target_functions
        if mapped_target_functions else 0.0
    )

    return float(concentration), float(spread), source_function_by_target


def unique_functions(functions: list[asm.Function]) -> list[asm.Function]:
    """Return functions once, preserving the matching order."""
    seen = set()
    unique = []

    for function in functions:
        key = (function.address, function.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(function)

    return unique


def evaluate_block_match(
    match: Match,
    block_threshold: float,
    block_assignment_threshold: float,
    block_min_coverage_ratio: float,
    block_coverage_mean_threshold: float | None = None,
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
    min_function_concentration: float = 0.45,
    min_function_spread: float = 0.50,
) -> BlockMatchResult:
    """Evaluate one function-selected CU match with block-level matching."""
    source_functions = unique_functions(
        [function_pair.source_function for function_pair in match.matched_functions]
    )
    target_functions = unique_functions(
        [function_pair.target_function for function_pair in match.matched_functions]
    )

    source_block_records = block_records(source_functions, min_block_instructions)
    target_block_records = block_records(target_functions, min_block_instructions)

    if not source_block_records or not target_block_records:
        return BlockMatchResult(
            score=0.0,
            coverage_mean=0.0,
            coverage_min=0.0,
            coverage_ratio=0.0,
            assignment_mean=0.0,
            assignment_min=0.0,
            num_source_blocks=len(source_block_records),
            num_target_blocks=len(target_block_records),
            passed=False,
        )

    sim_matrix = compute_blocks_similarity_matrix(
        [record["block"] for record in source_block_records],
        [record["block"] for record in target_block_records],
    )
    coverage_values = best_coverage_by_source_block(sim_matrix)
    assigned_values = assignment_values(sim_matrix)
    best_source_rows_by_target_block = np.argmax(sim_matrix, axis=0)
    function_concentration, function_spread, _ = function_concentration_scores(
        target_block_records,
        source_block_records,
        best_source_rows_by_target_block,
    )

    coverage_mean = safe_mean(coverage_values)
    coverage_min = safe_min(coverage_values)
    assignment_mean = safe_mean(assigned_values)
    assignment_min = safe_min(assigned_values)
    coverage_mean_threshold = (
        block_threshold
        if block_coverage_mean_threshold is None
        else block_coverage_mean_threshold
    )
    coverage_ratio = float(np.mean(coverage_values >= block_threshold))
    passed = (
        coverage_mean >= coverage_mean_threshold
        and coverage_ratio >= block_min_coverage_ratio
        and assignment_mean >= block_assignment_threshold
        and function_concentration >= min_function_concentration
        and function_spread >= min_function_spread
    )

    return BlockMatchResult(
        score=coverage_mean,
        coverage_mean=coverage_mean,
        coverage_min=coverage_min,
        coverage_ratio=coverage_ratio,
        assignment_mean=assignment_mean,
        assignment_min=assignment_min,
        num_source_blocks=len(source_block_records),
        num_target_blocks=len(target_block_records),
        passed=passed,
        function_concentration=function_concentration,
        function_spread=function_spread,
    )


def evaluate_block_presence(
    source_unit: asm.CodeUnit,
    target_unit: asm.CodeUnit,
    block_threshold: float,
    block_assignment_threshold: float,
    block_min_coverage_ratio: float,
    block_coverage_mean_threshold: float | None = None,
    locality_window_multiplier: float = 3.0,
    locality_window_padding: int = 2,
    min_edge_locality_ratio: float = 0.5,
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
    min_function_concentration: float = 0.45,
    min_function_spread: float = 0.50,
) -> BlockMatchResult:
    """Evaluate whether a target CU's blocks are present anywhere in an ELF.

    Unlike `evaluate_block_match`, this does not require function-level pairs.
    Coverage is measured from target-library blocks to source-ELF blocks inside
    a compact source-function window, so a small library CU can match inside a
    much larger executable without allowing arbitrary far-away block matches.
    """
    target_block_records, target_spans = indexed_block_records(
        target_unit.functions,
        min_block_instructions,
    )
    source_block_records, source_spans = indexed_block_records(
        source_unit.functions,
        min_block_instructions,
    )

    if not source_block_records or not target_block_records:
        return BlockMatchResult(
            score=0.0,
            coverage_mean=0.0,
            coverage_min=0.0,
            coverage_ratio=0.0,
            assignment_mean=0.0,
            assignment_min=0.0,
            num_source_blocks=len(target_block_records),
            num_target_blocks=len(source_block_records),
            passed=False,
        )

    full_sim_matrix = compute_blocks_similarity_matrix(
        [record["block"] for record in target_block_records],
        [record["block"] for record in source_block_records],
    )
    coverage_mean_threshold = (
        block_threshold
        if block_coverage_mean_threshold is None
        else block_coverage_mean_threshold
    )

    source_function_count = len(source_unit.functions)
    target_function_count = max(1, len(target_unit.functions))
    window_size = min(
        source_function_count,
        max(
            target_function_count,
            int(np.ceil(target_function_count * locality_window_multiplier))
            + locality_window_padding,
        ),
    )

    best_result = BlockMatchResult(
        score=0.0,
        coverage_mean=0.0,
        coverage_min=0.0,
        coverage_ratio=0.0,
        assignment_mean=0.0,
        assignment_min=0.0,
        num_source_blocks=len(target_block_records),
        num_target_blocks=len(source_block_records),
        passed=False,
        locality_span=window_size,
        edge_locality_ratio=1.0,
        function_concentration=0.0,
        function_spread=0.0,
    )
    best_sort_key = (-1.0, -1.0, -1.0, -1.0, -1.0)

    for window_start in range(0, max(1, source_function_count - window_size + 1)):
        window_stop = window_start + window_size
        block_start = source_spans[window_start][0]
        block_stop = source_spans[window_stop - 1][1]
        if block_stop <= block_start:
            continue

        sim_matrix = full_sim_matrix[:, block_start:block_stop]
        coverage_values = best_coverage_by_source_block(sim_matrix)
        assigned_values = assignment_values(sim_matrix)

        coverage_mean = safe_mean(coverage_values)
        coverage_min = safe_min(coverage_values)
        assignment_mean = safe_mean(assigned_values)
        assignment_min = safe_min(assigned_values)
        coverage_ratio = float(np.mean(coverage_values >= block_threshold))

        best_source_columns = np.argmax(sim_matrix, axis=1) + block_start
        function_concentration, function_spread, source_function_by_target = (
            function_concentration_scores(
                target_block_records,
                source_block_records,
                best_source_columns,
            )
        )
        edge_locality_ratio = internal_call_edge_locality_ratio(
            target_unit.functions,
            source_function_by_target,
            locality_window_multiplier,
            locality_window_padding,
        )
        passed = (
            coverage_mean >= coverage_mean_threshold
            and coverage_ratio >= block_min_coverage_ratio
            and assignment_mean >= block_assignment_threshold
            and edge_locality_ratio >= min_edge_locality_ratio
            and function_concentration >= min_function_concentration
            and function_spread >= min_function_spread
        )
        result = BlockMatchResult(
            score=coverage_mean,
            coverage_mean=coverage_mean,
            coverage_min=coverage_min,
            coverage_ratio=coverage_ratio,
            assignment_mean=assignment_mean,
            assignment_min=assignment_min,
            num_source_blocks=len(target_block_records),
            num_target_blocks=block_stop - block_start,
            passed=passed,
            locality_span=window_size,
            edge_locality_ratio=edge_locality_ratio,
            function_concentration=function_concentration,
            function_spread=function_spread,
        )

        sort_key = (
            coverage_mean,
            assignment_mean,
            coverage_ratio,
            function_concentration,
            function_spread,
        )
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_result = result

    return best_result


def select_block_candidates(
    matches: list[Match],
    candidate_threshold: float,
    candidate_top_k: int,
    candidate_low_threshold: float | None = None,
) -> list[Match]:
    """Select function-level CU matches to refine with block-level matching.

    `candidate_threshold` marks strong function-level candidates.  The optional
    `candidate_low_threshold` opens a recovery band: function-level matches below the
    strong threshold but still above the lower threshold are also evaluated by
    the block matcher instead of being discarded immediately.
    """
    effective_threshold = candidate_threshold
    if candidate_low_threshold is not None:
        effective_threshold = min(candidate_threshold, candidate_low_threshold)

    candidate_matches = [
        match
        for match in matches
        if match.get_score() >= effective_threshold
    ]
    candidate_matches.sort(key=lambda match: match.get_score(), reverse=True)

    if candidate_top_k > 0:
        return candidate_matches[:candidate_top_k]

    return candidate_matches


def internal_call_edges(functions: list[asm.Function]) -> set[tuple[int, int]]:
    """Return call edges whose caller and callee are both in `functions`."""
    addresses = {function.address for function in functions}
    edges = set()

    for caller in functions:
        for callee_addr in caller.resolved_call_targets:
            if callee_addr in addresses:
                edges.add((caller.address, callee_addr))

    return edges


def call_graph_assignment_score(
    source_functions: list[asm.Function],
    target_functions: list[asm.Function],
    target_to_source: dict[int, int],
) -> float:
    """Score whether matched functions preserve the target CU call edges.

    The score uses only numeric call targets resolved to local function addresses.
    Symbols and function names are not consulted.
    """
    target_edges = internal_call_edges(target_functions)
    if not target_edges:
        return 0.0

    source_edges = internal_call_edges(source_functions)

    matched_edges = 0
    for target_caller, target_callee in target_edges:
        source_caller = target_to_source.get(target_caller)
        source_callee = target_to_source.get(target_callee)

        if source_caller is None or source_callee is None:
            continue

        if (source_caller, source_callee) in source_edges:
            matched_edges += 1

    return matched_edges / len(target_edges)


def function_assignment_locality_score(
    target_functions: list[asm.Function],
    source_global_indices: np.ndarray,
    target_indices: np.ndarray,
    locality_multiplier: float = FUNCTION_LOCALITY_WINDOW_MULTIPLIER,
    locality_padding: int = FUNCTION_LOCALITY_WINDOW_PADDING,
) -> float:
    """Score whether target internal call edges map to nearby source functions."""
    source_function_by_target = {
        int(target_index): int(source_index)
        for source_index, target_index in zip(source_global_indices, target_indices)
    }
    return internal_call_edge_locality_ratio(
        target_functions,
        source_function_by_target,
        locality_multiplier,
        locality_padding,
        empty_score=0.0,
    )


def combined_similarity(
    base_similarity: float,
    call_graph_similarity: float,
    function_locality_similarity: float = 0.0,
) -> float:
    return min(
        1.0,
        base_similarity
        + (MODIFIER_CALL_GRAPH * call_graph_similarity)
        + (MODIFIER_FUNCTION_LOCALITY * function_locality_similarity),
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
            window_locality_scores = []
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
                call_graph_score = call_graph_assignment_score(
                    source_functions,
                    target_cu_functions,
                    target_to_source,
                )
                locality_score = function_assignment_locality_score(
                    target_cu_functions,
                    source_global_rows,
                    target_cols,
                )
                assigned_similarities = [
                    combined_similarity(
                        float(source_window_matrix[source_row][target_col]),
                        call_graph_score,
                        locality_score,
                    )
                    for source_row, target_col in zip(source_rows, target_cols)
                ]
                window_similarity_sums.append(sum(assigned_similarities))
                window_call_graph_scores.append(call_graph_score)
                window_locality_scores.append(locality_score)

            # Pick the source-binary window with the best combined score.
            best_idx = np.argmax(window_similarity_sums)
            best_source_window = source_candidate_windows[best_idx]
            best_source_to_target_assignment = source_to_target_assignments[best_idx]
            best_sum_similarity = float(window_similarity_sums[best_idx])
            best_call_graph_score = float(window_call_graph_scores[best_idx])
            best_locality_score = float(window_locality_scores[best_idx])
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
                        combined_similarity(
                            float(sim_matrix[source_idx][target_idx]),
                            best_call_graph_score,
                            best_locality_score,
                        ),
                        best_call_graph_score,
                        best_locality_score,
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
                f"call_graph={best_call_graph_score:.4f}, "
                f"locality={best_locality_score:.4f}"
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
