import typing
from collections import Counter


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

import asm

DEFAULT_MIN_BLOCK_INSTRUCTIONS = 3
DEFAULT_RODATA_NGRAM_SIZE = 16
MIN_RODATA_BYTES_FOR_BYTE_SCORE = 64
MIN_RODATA_NGRAMS_FOR_BYTE_SCORE = 4


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
    windows_total: int = 0
    windows_evaluated: int = 0
    windows_skipped: int = 0


class RodataIndex(typing.NamedTuple):
    """Precomputed .rodata features for one source ELF."""

    data: bytes
    strings: Counter[str]
    ngrams: set[bytes]
    byte_size: int
    string_count: int
    ngram_count: int


class RodataMatchResult(typing.NamedTuple):
    """Containment-style .rodata match between a target CU and a source ELF."""

    score: float
    string_score: float
    byte_score: float
    target_bytes: int
    source_bytes: int
    matched_strings: int
    target_strings: int
    matched_ngrams: int
    target_ngrams: int
    has_rodata: bool


RodataEvidenceStatus = typing.Literal[
    "confirm",
    "penalty",
    "neutral",
    "disabled",
    "skipped",
]


class RodataEvidence(typing.NamedTuple):
    """Decision-oriented interpretation of a .rodata match."""

    status: RodataEvidenceStatus
    informative: bool


def rodata_ngrams(data: bytes, ngram_size: int = DEFAULT_RODATA_NGRAM_SIZE) -> set[bytes]:
    """Return byte n-grams for containment matching, skipping all-zero grams."""
    if len(data) < ngram_size:
        return {data} if any(data) else set()

    return {
        data[index:index + ngram_size]
        for index in range(0, len(data) - ngram_size + 1)
        if any(data[index:index + ngram_size])
    }


def build_rodata_index(
    code_unit: asm.CodeUnit,
    ngram_size: int = DEFAULT_RODATA_NGRAM_SIZE,
) -> RodataIndex:
    """Build reusable .rodata features for one source code unit."""
    strings = Counter(code_unit.rodata_strings)
    ngrams = rodata_ngrams(code_unit.rodata_bytes, ngram_size)
    return RodataIndex(
        data=code_unit.rodata_bytes,
        strings=strings,
        ngrams=ngrams,
        byte_size=len(code_unit.rodata_bytes),
        string_count=sum(strings.values()),
        ngram_count=len(ngrams),
    )


def rodata_string_weight(value: str) -> float:
    """Weight longer strings more than short/common-looking fragments."""
    if len(value) < 4:
        return 0.0
    if len(value) < 6:
        return 0.5
    return float(min(len(value), 80))


def evaluate_rodata_match(
    source_index: RodataIndex,
    target_unit: asm.CodeUnit,
    ngram_size: int = DEFAULT_RODATA_NGRAM_SIZE,
) -> RodataMatchResult:
    """Score how much target CU .rodata is contained in the source ELF .rodata."""
    target_byte_size = len(target_unit.rodata_bytes)
    has_rodata = bool(target_byte_size or target_unit.rodata_strings)
    if not has_rodata:
        return RodataMatchResult(
            score=0.0,
            string_score=0.0,
            byte_score=0.0,
            target_bytes=0,
            source_bytes=source_index.byte_size,
            matched_strings=0,
            target_strings=0,
            matched_ngrams=0,
            target_ngrams=0,
            has_rodata=False,
        )

    target_strings = Counter(target_unit.rodata_strings)
    target_ngrams = rodata_ngrams(target_unit.rodata_bytes, ngram_size)

    string_total = 0.0
    string_matched = 0.0
    matched_strings = 0
    for value, target_count in target_strings.items():
        weight = rodata_string_weight(value)
        if weight <= 0:
            continue

        source_count = source_index.strings.get(value, 0)
        matched_count = min(target_count, source_count)
        string_total += target_count * weight
        string_matched += matched_count * weight
        matched_strings += matched_count

    string_score = string_matched / string_total if string_total else 0.0

    byte_score_is_informative = (
        target_byte_size >= MIN_RODATA_BYTES_FOR_BYTE_SCORE
        and len(target_ngrams) >= MIN_RODATA_NGRAMS_FOR_BYTE_SCORE
    )
    if byte_score_is_informative:
        matched_ngrams = len(target_ngrams & source_index.ngrams)
        byte_score = matched_ngrams / len(target_ngrams)
    else:
        matched_ngrams = 0
        byte_score = 0.0

    if string_total and byte_score_is_informative:
        score = (0.70 * string_score) + (0.30 * byte_score)
    elif string_total:
        score = string_score
    elif byte_score_is_informative:
        score = 0.50 * byte_score
    else:
        score = 0.0

    return RodataMatchResult(
        score=float(score),
        string_score=float(string_score),
        byte_score=float(byte_score),
        target_bytes=target_byte_size,
        source_bytes=source_index.byte_size,
        matched_strings=matched_strings,
        target_strings=sum(target_strings.values()),
        matched_ngrams=matched_ngrams,
        target_ngrams=len(target_ngrams),
        has_rodata=has_rodata,
    )


def classify_rodata_evidence(
    rodata_result: RodataMatchResult | None,
    min_bytes: int = 128,
    min_strings: int = 2,
    min_ngrams: int = 32,
    penalty_threshold: float = 0.10,
    confirm_threshold: float = 0.70,
) -> RodataEvidence:
    """Classify .rodata as confirm/penalty/neutral for a block-level CU match."""
    if rodata_result is None or not rodata_result.has_rodata:
        return RodataEvidence(status="neutral", informative=False)

    informative = (
        rodata_result.target_bytes >= min_bytes
        or rodata_result.target_strings >= min_strings
        or rodata_result.target_ngrams >= min_ngrams
    )
    if not informative:
        return RodataEvidence(status="neutral", informative=False)

    if rodata_result.score >= confirm_threshold:
        status = "confirm"
    elif rodata_result.score <= penalty_threshold:
        status = "penalty"
    else:
        status = "neutral"

    return RodataEvidence(status=status, informative=True)


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
                    "function_index": function_index,
                    "block": block,
                }
            )
        spans.append((start, len(records)))

    return records, spans


def compute_blocks_similarity_matrix(
    target_blocks: list[asm.Block],
    source_blocks: list[asm.Block],
) -> np.ndarray:
    """Compute cosine similarity between two lists of embedded basic blocks."""
    target_embeddings = np.stack([np.squeeze(block.embedding) for block in target_blocks])
    source_embeddings = np.stack([np.squeeze(block.embedding) for block in source_blocks])
    return cosine_similarity(target_embeddings, source_embeddings)


def assignment_values(sim_matrix: np.ndarray) -> np.ndarray:
    """Return maximum-similarity linear-assignment values."""
    row_indices, column_indices = linear_sum_assignment(sim_matrix, maximize=True)
    return sim_matrix[row_indices, column_indices]


def function_max_similarity_matrix(
    block_similarity_matrix: np.ndarray,
    function_spans: list[tuple[int, int]],
) -> np.ndarray:
    """Return each target block's best similarity inside each source function."""
    function_matrix = np.full(
        (block_similarity_matrix.shape[0], len(function_spans)),
        -np.inf,
        dtype=block_similarity_matrix.dtype,
    )

    for function_index, (block_start, block_stop) in enumerate(function_spans):
        if block_stop <= block_start:
            continue
        function_matrix[:, function_index] = np.max(
            block_similarity_matrix[:, block_start:block_stop],
            axis=1,
        )

    return function_matrix


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


def evaluate_block_presence(
    source_unit: asm.CodeUnit,
    target_unit: asm.CodeUnit,
    block_threshold: float,
    block_assignment_threshold: float,
    min_coverage_ratio: float,
    min_coverage_mean: float,
    locality_window_multiplier: float = 3.0,
    locality_window_padding: int = 2,
    min_edge_locality_ratio: float = 0.5,
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
    min_function_concentration: float = 0.45,
    min_function_spread: float = 0.50,
    enable_window_prefilter: bool = True,
) -> BlockMatchResult:
    """Evaluate whether a target CU's blocks are present anywhere in an ELF.

    Coverage is measured from target-library blocks to source-ELF blocks inside
    a compact source-function window, so a small library CU can match inside a
    much larger executable without allowing arbitrary far-away block matches.
    A cheap function-level prefilter keeps the expensive assignment/locality
    checks for windows that can still satisfy the coverage gates.
    """
    target_block_records, _ = indexed_block_records(
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
            num_source_blocks=len(source_block_records),
            num_target_blocks=len(target_block_records),
            passed=False,
        )

    full_sim_matrix = compute_blocks_similarity_matrix(
        [record["block"] for record in target_block_records],
        [record["block"] for record in source_block_records],
    )
    source_function_count = len(source_unit.functions)
    target_function_count = len(target_unit.functions)
    window_size = min(
        source_function_count,
        max(
            target_function_count,
            int(np.ceil(target_function_count * locality_window_multiplier))
            + locality_window_padding,
        ),
    )
    window_total = max(1, source_function_count - window_size + 1)
    function_sim_matrix = function_max_similarity_matrix(
        full_sim_matrix,
        source_spans,
    )

    best_result = BlockMatchResult(
        score=0.0,
        coverage_mean=0.0,
        coverage_min=0.0,
        coverage_ratio=0.0,
        assignment_mean=0.0,
        assignment_min=0.0,
        num_source_blocks=len(source_block_records),
        num_target_blocks=len(target_block_records),
        passed=False,
        locality_span=window_size,
        edge_locality_ratio=1.0,
        function_concentration=0.0,
        function_spread=0.0,
        windows_total=window_total,
    )
    best_sort_key = (False, -1.0, -1.0, -1.0, -1.0, -1.0)
    windows_evaluated = 0
    windows_skipped = 0

    for window_start in range(window_total):
        window_stop = window_start + window_size
        block_start = source_spans[window_start][0]
        block_stop = source_spans[window_stop - 1][1]
        if block_stop <= block_start:
            windows_skipped += 1
            continue

        coverage_values = np.max(
            function_sim_matrix[:, window_start:window_stop],
            axis=1,
        )
        coverage_mean = float(np.mean(coverage_values))
        coverage_min = float(np.min(coverage_values))
        coverage_ratio = float(np.mean(coverage_values >= block_threshold))

        should_skip_window = (
            enable_window_prefilter
            and (
                coverage_mean < min_coverage_mean
                or coverage_ratio < min_coverage_ratio
            )
        )
        if should_skip_window:
            windows_skipped += 1
            result = BlockMatchResult(
                score=coverage_mean,
                coverage_mean=coverage_mean,
                coverage_min=coverage_min,
                coverage_ratio=coverage_ratio,
                assignment_mean=0.0,
                assignment_min=0.0,
                num_source_blocks=block_stop - block_start,
                num_target_blocks=len(target_block_records),
                passed=False,
                locality_span=window_size,
                edge_locality_ratio=0.0,
                function_concentration=0.0,
                function_spread=0.0,
                windows_total=window_total,
            )
            sort_key = (
                False,
                coverage_mean,
                0.0,
                coverage_ratio,
                0.0,
                0.0,
            )
            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_result = result
            continue

        windows_evaluated += 1
        sim_matrix = full_sim_matrix[:, block_start:block_stop]
        assigned_values = assignment_values(sim_matrix)
        assignment_mean = float(np.mean(assigned_values))
        assignment_min = float(np.min(assigned_values))

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
            coverage_mean >= min_coverage_mean
            and coverage_ratio >= min_coverage_ratio
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
            num_source_blocks=block_stop - block_start,
            num_target_blocks=len(target_block_records),
            passed=passed,
            locality_span=window_size,
            edge_locality_ratio=edge_locality_ratio,
            function_concentration=function_concentration,
            function_spread=function_spread,
        )

        sort_key = (
            passed,
            coverage_mean,
            assignment_mean,
            coverage_ratio,
            function_concentration,
            function_spread,
        )
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_result = result

    return best_result._replace(
        windows_total=window_total,
        windows_evaluated=windows_evaluated,
        windows_skipped=windows_skipped,
    )
