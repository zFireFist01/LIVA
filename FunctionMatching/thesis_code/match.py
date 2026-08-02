import typing
import math
import statistics
from collections import Counter
from collections.abc import Iterable


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

import asm

DEFAULT_MIN_BLOCK_INSTRUCTIONS = 3
DEFAULT_RODATA_NGRAM_SIZE = 16
MIN_RODATA_BYTES_FOR_BYTE_SCORE = 64
MIN_RODATA_NGRAMS_FOR_BYTE_SCORE = 4

LIBRARY_SCORE_AGGREGATORS = (
    "mean",
    "max",
    "top3_mean",
    "top3_noisy_or",
)


def aggregate_library_score(
    scores: Iterable[float],
    mode: str,
    *,
    score_floor: float = 0.0,
) -> float:
    """Reduce accepted CU scores to one score for a library decision.

    ``score_floor`` is used only by ``top3_noisy_or`` to map the accepted
    score interval ``[score_floor, 1]`` to probability-like evidence. The
    other modes operate directly on the CU block scores.
    """
    ranked = sorted((float(score) for score in scores), reverse=True)
    if not ranked:
        return 0.0
    if mode == "mean":
        return statistics.fmean(ranked)
    if mode == "max":
        return ranked[0]

    top = ranked[:3]
    if mode == "top3_mean":
        return statistics.fmean(top)
    if mode == "top3_noisy_or":
        scale = max(1e-12, 1.0 - float(score_floor))
        probabilities = [
            min(1.0, max(0.0, (score - float(score_floor)) / scale))
            for score in top
        ]
        return 1.0 - math.prod(1.0 - probability for probability in probabilities)
    raise ValueError(
        f"Unknown library score aggregator {mode!r}; expected one of "
        + ", ".join(LIBRARY_SCORE_AGGREGATORS)
)


class FunctionMatch(typing.NamedTuple):
    """Block-derived mapping of one reference function into the source ELF."""

    target_function_index: int
    source_function_index: int
    dominant_ratio: float
    coverage_mean: float
    coverage_ratio: float


class BlockWindowResult(typing.NamedTuple):
    """Threshold-independent metrics for one fully evaluated source window."""

    window_start: int
    window_stop: int
    block_start: int
    block_stop: int
    num_source_blocks: int
    coverage_mean: float
    coverage_min: float
    coverage_ratio: float
    assignment_quality: float
    assignment_ratio: float
    call_edge_ratio: float
    call_edges_evaluated: int
    call_edges_total: int
    function_concentration: float
    function_spread: float
    function_matches: tuple[FunctionMatch, ...] = ()


class BlockMatchResult(typing.NamedTuple):
    """Result of a block-level refinement for one CU match."""

    score: float
    coverage_mean: float
    coverage_min: float
    coverage_ratio: float
    assignment_quality: float
    assignment_ratio: float
    num_source_blocks: int
    num_target_blocks: int
    passed: bool
    locality_span: int = 0
    call_edge_ratio: float = 1.0
    call_edges_evaluated: int = 0
    call_edges_total: int = 0
    function_concentration: float = 1.0
    function_spread: float = 1.0
    windows_total: int = 0
    windows_evaluated: int = 0
    windows_skipped: int = 0
    windows: tuple[BlockWindowResult, ...] = ()
    function_matches: tuple[FunctionMatch, ...] = ()


class CrossCUCallEvidence(typing.NamedTuple):
    """Preservation statistics for direct calls between candidate CUs."""

    matched_edges: int
    evaluable_edges: int
    expected_edges: int
    ratio: float
    coverage: float


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
    string_informative: bool = False
    byte_informative: bool = False


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
            string_informative=False,
            byte_informative=False,
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

    string_informative = bool(string_total)
    string_score = string_matched / string_total if string_informative else 0.0

    byte_informative = (
        target_byte_size >= MIN_RODATA_BYTES_FOR_BYTE_SCORE
        and len(target_ngrams) >= MIN_RODATA_NGRAMS_FOR_BYTE_SCORE
    )
    if byte_informative:
        matched_ngrams = len(target_ngrams & source_index.ngrams)
        byte_score = matched_ngrams / len(target_ngrams)
    else:
        matched_ngrams = 0
        byte_score = 0.0

    if string_informative and byte_informative:
        score = (0.70 * string_score) + (0.30 * byte_score)
    elif string_informative:
        score = string_score
    elif byte_informative:
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
        string_informative=string_informative,
        byte_informative=byte_informative,
    )


def classify_rodata_evidence(
    rodata_result: RodataMatchResult | None,
    min_bytes: int = 128,
    min_strings: int = 2,
    min_ngrams: int = 32,
    penalty_threshold: float = 0.0,
    confirm_threshold: float = 0.70,
) -> RodataEvidence:
    """Classify .rodata as confirm/penalty/neutral for a block-level CU match."""
    if rodata_result is None or not rodata_result.has_rodata:
        return RodataEvidence(status="neutral", informative=False)

    # Keep the original threshold semantics: any one of the three evidence
    # quantities is sufficient to make .rodata informative.  In particular,
    # min_strings=0 deliberately makes every non-empty .rodata section
    # informative, even when no string was extracted.
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


def apply_rodata_bonus(
    block_score: float,
    rodata_result: RodataMatchResult | None,
    rodata_evidence: RodataEvidence,
    bonus_weight: float = 0.30,
    confirm_threshold: float = 0.70,
) -> float:
    """Apply a bounded positive adjustment to a .rodata-confirmed CU score.

    Neutral, unavailable and disabled evidence leaves the block score intact.
    Penalties are handled by the caller because they reject the CU.  The
    interpolation keeps the adjusted score in ``[block_score, 1]`` and makes
    the bonus grow continuously from zero at ``confirm_threshold``.
    """
    score = min(1.0, max(0.0, float(block_score)))
    if (
        rodata_evidence.status != "confirm"
        or rodata_result is None
        or bonus_weight <= 0.0
    ):
        return score

    threshold = min(1.0, max(0.0, float(confirm_threshold)))
    if threshold >= 1.0:
        confidence = 1.0
    else:
        confidence = (
            float(rodata_result.score) - threshold
        ) / (1.0 - threshold)
    confidence = min(1.0, max(0.0, confidence))
    adjusted = score + float(bonus_weight) * confidence * (1.0 - score)
    return min(1.0, max(score, adjusted))


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


def assignment_scores(
    sim_matrix: np.ndarray,
    match_threshold: float,
) -> tuple[float, float]:
    """Return quality and target coverage of a one-to-one block assignment.

    The assignment is lexicographic: it first maximizes the number of distinct
    pairs reaching ``match_threshold``, then their total cosine similarity.
    This prevents a single excellent pair from replacing two merely good pairs.

    ``linear_sum_assignment`` emits only ``min(targets, sources)`` pairs for a
    rectangular matrix.  Quality is computed on those pairs, while ratio uses
    every target block as its denominator.  Unassigned targets and pairs below
    threshold both reduce the ratio; extra source blocks do not.
    """
    pair_count = min(sim_matrix.shape)
    bounded_similarities = np.clip(sim_matrix, -1.0, 1.0)
    good_pair_bonus = (2.0 * pair_count) + 1.0
    assignment_objective = bounded_similarities + (
        (bounded_similarities >= match_threshold) * good_pair_bonus
    )
    row_indices, column_indices = linear_sum_assignment(
        assignment_objective,
        maximize=True,
    )
    values = sim_matrix[row_indices, column_indices]
    quality = float(np.mean(values))
    ratio = float(np.count_nonzero(values >= match_threshold) / sim_matrix.shape[0])
    return quality, ratio


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


def internal_call_edge_ratio(
    source_functions: list[asm.Function],
    target_functions: list[asm.Function],
    source_function_by_target: dict[int, int],
    empty_score: float = 1.0,
) -> tuple[float, int, int]:
    """Measure preservation of numeric direct-call edges after function mapping.

    Function identities are represented only by recovered entry addresses.  No
    symbol or function name participates in this score, so it remains usable on
    stripped ELF files.  A target edge is evaluable only when both endpoint
    functions have a block-derived source mapping.  With no evaluable edges the
    result is neutral rather than an artificial failure.
    """
    target_index_by_address = {
        function.address: index
        for index, function in enumerate(target_functions)
    }
    matched_edges = 0
    evaluated_edges = 0
    total_edges = 0

    for caller_index, caller in enumerate(target_functions):
        source_caller_index = source_function_by_target.get(caller_index)

        for callee_address in caller.resolved_call_targets:
            callee_index = target_index_by_address.get(callee_address)
            if callee_index is None:
                continue

            total_edges += 1

            source_callee_index = source_function_by_target.get(callee_index)
            if (
                source_caller_index is None
                or source_callee_index is None
                or not 0 <= source_caller_index < len(source_functions)
                or not 0 <= source_callee_index < len(source_functions)
            ):
                continue

            evaluated_edges += 1
            source_caller = source_functions[source_caller_index]
            source_callee = source_functions[source_callee_index]
            if source_callee.address in source_caller.resolved_call_targets:
                matched_edges += 1

    if not evaluated_edges:
        return empty_score, 0, total_edges

    return matched_edges / evaluated_edges, evaluated_edges, total_edges


def function_concentration_scores(
    target_block_records: list[dict],
    source_block_records: list[dict],
    matched_source_block_indices: np.ndarray,
    matched_source_similarities: np.ndarray | None = None,
    match_threshold: float = 0.0,
) -> tuple[
    float,
    float,
    dict[int, int],
    tuple[FunctionMatch, ...],
]:
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
        return 0.0, 0.0, {}, ()

    dominant_total = 0
    block_total = 0
    source_function_by_target: dict[int, int] = {}
    function_matches: list[FunctionMatch] = []

    for target_function_index, source_indices in source_indices_by_target_function.items():
        values, counts = np.unique(source_indices, return_counts=True)
        dominant_position = int(np.argmax(counts))
        dominant_source_index = int(values[dominant_position])
        dominant_count = int(counts[dominant_position])

        source_function_by_target[target_function_index] = dominant_source_index
        dominant_total += dominant_count
        block_total += len(source_indices)
        target_positions = [
            index
            for index, record in enumerate(target_block_records)
            if record["function_index"] == target_function_index
        ]
        similarities = (
            np.asarray(matched_source_similarities)[target_positions]
            if matched_source_similarities is not None
            else np.ones(len(target_positions), dtype=float)
        )
        function_matches.append(
            FunctionMatch(
                target_function_index=int(target_function_index),
                source_function_index=dominant_source_index,
                dominant_ratio=float(dominant_count / len(source_indices)),
                coverage_mean=float(np.mean(similarities)),
                coverage_ratio=float(
                    np.mean(similarities >= match_threshold)
                ),
            )
        )

    concentration = dominant_total / block_total if block_total else 0.0
    mapped_target_functions = len(source_function_by_target)
    spread = (
        len(set(source_function_by_target.values())) / mapped_target_functions
        if mapped_target_functions else 0.0
    )

    return (
        float(concentration),
        float(spread),
        source_function_by_target,
        tuple(sorted(function_matches)),
    )


def cross_cu_call_evidence(
    source_functions: list[asm.Function],
    target_units: list[asm.CodeUnit],
    block_results: list[BlockMatchResult],
    call_edges: typing.Iterable[asm.InterCUCallEdge],
    included_cu_indices: set[int] | None = None,
    require_both_included: bool = True,
) -> CrossCUCallEvidence:
    """Measure reference inter-CU calls preserved by block-derived mappings.

    Endpoint mappings come exclusively from block similarity. Archive symbols
    are used beforehand only to create integer reference edges; target ELF
    names and symbols never participate.
    """
    included = (
        set(range(len(target_units)))
        if included_cu_indices is None
        else set(included_cu_indices)
    )
    source_index_by_address = {
        function.address: index
        for index, function in enumerate(source_functions)
    }
    source_edges = {
        (caller_index, source_index_by_address[callee_address])
        for caller_index, caller in enumerate(source_functions)
        for callee_address in caller.resolved_call_targets
        if callee_address in source_index_by_address
    }
    mapping_by_cu = [
        {
            match.target_function_index: match.source_function_index
            for match in result.function_matches
        }
        for result in block_results
    ]

    matched_edges = 0
    evaluable_edges = 0
    expected_edges = 0
    for edge in call_edges:
        selected_endpoint_count = sum(
            index in included
            for index in (edge.caller_cu_index, edge.callee_cu_index)
        )
        if (
            (
                require_both_included
                and selected_endpoint_count != 2
            )
            or (
                not require_both_included
                and selected_endpoint_count == 0
            )
            or not 0 <= edge.caller_cu_index < len(mapping_by_cu)
            or not 0 <= edge.callee_cu_index < len(mapping_by_cu)
        ):
            continue
        expected_edges += 1
        source_caller = mapping_by_cu[edge.caller_cu_index].get(
            edge.caller_function_index
        )
        source_callee = mapping_by_cu[edge.callee_cu_index].get(
            edge.callee_function_index
        )
        if source_caller is None or source_callee is None:
            continue
        evaluable_edges += 1
        if (source_caller, source_callee) in source_edges:
            matched_edges += 1

    ratio = matched_edges / evaluable_edges if evaluable_edges else 0.0
    coverage = evaluable_edges / expected_edges if expected_edges else 0.0
    return CrossCUCallEvidence(
        matched_edges=matched_edges,
        evaluable_edges=evaluable_edges,
        expected_edges=expected_edges,
        ratio=float(ratio),
        coverage=float(coverage),
    )


def apply_cross_cu_call_adjustment(
    score: float,
    evidence: CrossCUCallEvidence,
    bonus_weight: float = 0.0,
    penalty_weight: float = 0.0,
    saturation_edges: int = 3,
) -> float:
    """Apply bounded, reliability-weighted cross-CU evidence to a score.

    Positive preservation is stronger evidence than a missing edge, which can
    result from optimization. Both adjustments remain disabled until they have
    been validated on a program-disjoint panel.
    """
    bounded_score = min(1.0, max(0.0, float(score)))
    if evidence.evaluable_edges <= 0 or evidence.expected_edges <= 0:
        return bounded_score

    saturation = max(1, int(saturation_edges))
    reliability = min(1.0, evidence.evaluable_edges / saturation)
    reliability *= evidence.coverage
    positive = (
        max(0.0, float(bonus_weight))
        * reliability
        * evidence.ratio
        * (1.0 - bounded_score)
    )
    negative = (
        max(0.0, float(penalty_weight))
        * reliability
        * (1.0 - evidence.ratio)
        * bounded_score
    )
    return min(1.0, max(0.0, bounded_score + positive - negative))


def block_window_gate_values(window: BlockWindowResult) -> tuple[float, ...]:
    """Return the seven maximized metrics used by the block-presence decision."""
    return (
        window.coverage_mean,
        window.coverage_ratio,
        window.assignment_quality,
        window.assignment_ratio,
        window.call_edge_ratio,
        window.function_concentration,
        window.function_spread,
    )


def pareto_block_windows(
    windows: typing.Iterable[BlockWindowResult],
) -> tuple[BlockWindowResult, ...]:
    """Return a compact frontier sufficient for threshold-conjunction replay.

    All seven decision metrics are maximized. Equal metric vectors are represented
    by their first window because later duplicates cannot change a replayed
    decision. Windows with non-finite metrics are omitted: they cannot be
    compared safely and must never be interpreted as measured zeroes.
    """
    frontier: list[BlockWindowResult] = []

    for candidate in windows:
        candidate_values = block_window_gate_values(candidate)
        if not all(np.isfinite(value) for value in candidate_values):
            continue

        existing_values = [block_window_gate_values(window) for window in frontier]
        if any(
            values == candidate_values
            or (
                all(left >= right for left, right in zip(values, candidate_values))
                and any(left > right for left, right in zip(values, candidate_values))
            )
            for values in existing_values
        ):
            continue

        frontier = [
            window
            for window, values in zip(frontier, existing_values)
            if not (
                all(left >= right for left, right in zip(candidate_values, values))
                and any(left > right for left, right in zip(candidate_values, values))
            )
        ]
        frontier.append(candidate)

    return tuple(frontier)


def evaluate_block_presence(
    source_unit: asm.CodeUnit,
    target_unit: asm.CodeUnit,
    block_threshold: float,
    min_assignment_quality: float,
    min_assignment_ratio: float,
    min_coverage_ratio: float,
    min_coverage_mean: float,
    locality_window_multiplier: float = 3.0,
    locality_window_padding: int = 2,
    min_call_edge_ratio: float = 0.5,
    min_block_instructions: int = DEFAULT_MIN_BLOCK_INSTRUCTIONS,
    min_function_concentration: float = 0.45,
    min_function_spread: float = 0.50,
    enable_window_prefilter: bool = True,
) -> BlockMatchResult:
    """Evaluate whether a target CU's blocks are present anywhere in an ELF.

    Coverage is measured from target-library blocks to source-ELF blocks inside
    a compact source-function window, so a small library CU can match inside a
    much larger executable without allowing arbitrary far-away block matches.
    A cheap function-level prefilter keeps the expensive assignment and
    call-graph checks for windows that can still satisfy the coverage gates.
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
            assignment_quality=0.0,
            assignment_ratio=0.0,
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
        assignment_quality=0.0,
        assignment_ratio=0.0,
        num_source_blocks=len(source_block_records),
        num_target_blocks=len(target_block_records),
        passed=False,
        locality_span=window_size,
        call_edge_ratio=1.0,
        function_concentration=0.0,
        function_spread=0.0,
        windows_total=window_total,
    )
    best_sort_key = (False, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    windows_evaluated = 0
    windows_skipped = 0
    evaluated_windows: list[BlockWindowResult] = []

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
            sim_matrix = full_sim_matrix[:, block_start:block_stop]
            best_source_columns = np.argmax(sim_matrix, axis=1) + block_start
            best_source_similarities = sim_matrix[
                np.arange(sim_matrix.shape[0]),
                best_source_columns - block_start,
            ]
            (
                _concentration,
                _spread,
                _source_function_by_target,
                function_matches,
            ) = function_concentration_scores(
                target_block_records,
                source_block_records,
                best_source_columns,
                best_source_similarities,
                block_threshold,
            )
            result = BlockMatchResult(
                score=coverage_mean,
                coverage_mean=coverage_mean,
                coverage_min=coverage_min,
                coverage_ratio=coverage_ratio,
                assignment_quality=0.0,
                assignment_ratio=0.0,
                num_source_blocks=block_stop - block_start,
                num_target_blocks=len(target_block_records),
                passed=False,
                locality_span=window_size,
                call_edge_ratio=0.0,
                function_concentration=0.0,
                function_spread=0.0,
                windows_total=window_total,
                function_matches=function_matches,
            )
            sort_key = (
                False,
                coverage_mean,
                coverage_ratio,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_result = result
            continue

        windows_evaluated += 1
        sim_matrix = full_sim_matrix[:, block_start:block_stop]
        assignment_quality, assignment_ratio = assignment_scores(
            sim_matrix,
            block_threshold,
        )

        best_source_columns = np.argmax(sim_matrix, axis=1) + block_start
        best_source_similarities = sim_matrix[
            np.arange(sim_matrix.shape[0]),
            best_source_columns - block_start,
        ]
        (
            function_concentration,
            function_spread,
            source_function_by_target,
            function_matches,
        ) = (
            function_concentration_scores(
                target_block_records,
                source_block_records,
                best_source_columns,
                best_source_similarities,
                block_threshold,
            )
        )
        call_edge_ratio, call_edges_evaluated, call_edges_total = (
            internal_call_edge_ratio(
                source_unit.functions,
                target_unit.functions,
                source_function_by_target,
            )
        )
        window_result = BlockWindowResult(
            window_start=window_start,
            window_stop=window_stop,
            block_start=block_start,
            block_stop=block_stop,
            num_source_blocks=block_stop - block_start,
            coverage_mean=coverage_mean,
            coverage_min=coverage_min,
            coverage_ratio=coverage_ratio,
            assignment_quality=assignment_quality,
            assignment_ratio=assignment_ratio,
            call_edge_ratio=call_edge_ratio,
            call_edges_evaluated=call_edges_evaluated,
            call_edges_total=call_edges_total,
            function_concentration=function_concentration,
            function_spread=function_spread,
            function_matches=function_matches,
        )
        evaluated_windows.append(window_result)
        passed = (
            coverage_mean >= min_coverage_mean
            and coverage_ratio >= min_coverage_ratio
            and assignment_quality >= min_assignment_quality
            and assignment_ratio >= min_assignment_ratio
            and call_edge_ratio >= min_call_edge_ratio
            and function_concentration >= min_function_concentration
            and function_spread >= min_function_spread
        )
        result = BlockMatchResult(
            score=coverage_mean,
            coverage_mean=coverage_mean,
            coverage_min=coverage_min,
            coverage_ratio=coverage_ratio,
            assignment_quality=assignment_quality,
            assignment_ratio=assignment_ratio,
            num_source_blocks=block_stop - block_start,
            num_target_blocks=len(target_block_records),
            passed=passed,
            locality_span=window_size,
            call_edge_ratio=call_edge_ratio,
            call_edges_evaluated=call_edges_evaluated,
            call_edges_total=call_edges_total,
            function_concentration=function_concentration,
            function_spread=function_spread,
            function_matches=function_matches,
        )

        sort_key = (
            passed,
            coverage_mean,
            coverage_ratio,
            assignment_quality,
            assignment_ratio,
            call_edge_ratio,
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
        windows=pareto_block_windows(evaluated_windows),
    )
