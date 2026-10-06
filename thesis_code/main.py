import time
import argparse
import gc
import gzip
import json
from contextlib import nullcontext
from pathlib import Path
import re
import tempfile

from model import PALMTREE_POOLING_MODES
from analysis_cache import CachedCodeUnitLoader, DEFAULT_CACHE_DIR
from archive_utils import extract_archive_members
from asm import (
    ASM_NORMALIZATION_MODES,
    CodeUnit,
    InterCUCallEdge,
    resolve_archive_inter_cu_calls,
)
from match import (
    BlockMatchResult,
    LIBRARY_SCORE_AGGREGATORS,
    RodataEvidence,
    RodataMatchResult,
    aggregate_library_score,
    apply_cross_cu_call_adjustment,
    build_rodata_index,
    apply_rodata_bonus,
    classify_rodata_evidence,
    cross_cu_call_evidence,
    evaluate_block_presence,
    evaluate_rodata_match,
    prepare_block_unit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR / "Test"
REPO_ROOT = SCRIPT_DIR.parent

PATH_TO_CALCULATOR = str(
    REPO_ROOT / "Exploration" / "easy" / "gcc11" / "calculator_static_opt"
)

PATH_TO_LIBRARIES = str(REPO_ROOT / "Exploration" / "easy" / "libraries")

OUTPUT = str(TEST_DIR / "output.txt")

def get_args():
    parser = argparse.ArgumentParser(description="Compilation-unit block matching")

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
        "--features_output",
        type=str,
        default=None,
        help="Optional JSONL file with full CU-level matching features"
    )
    parser.add_argument(
        "--offline_ablation_features",
        "--offline-ablation-features",
        dest="offline_ablation_features",
        action="store_true",
        help=(
            "Write replay-complete CU features for a fixed-profile offline "
            "ablation. This increases feature-log size but does not change "
            "the online matching decision."
        ),
    )
    parser.add_argument(
        "--asm_model",
        type=str,
        default="palmtree/model/transformer.ep19",
        help="Path to the PalmTree model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device for PalmTree: auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument(
        "--asm_normalization",
        choices=ASM_NORMALIZATION_MODES,
        default="v2",
        help=(
            "Assembly preprocessing: v2 is stripped-safe and tokenizes Intel "
            "syntax for PalmTree; legacy only removes commas"
        ),
    )
    parser.add_argument(
        "--palmtree_pooling",
        "--palmtree-pooling",
        dest="palmtree_pooling",
        choices=PALMTREE_POOLING_MODES,
        default="masked_mean",
        help=(
            "PalmTree instruction pooling: masked_mean excludes padding and "
            "preserves <eos>; mean reproduces the original unmasked adapter"
        ),
    )
    parser.add_argument(
        "--analysis_cache_dir",
        "--analysis-cache-dir",
        dest="analysis_cache_dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "Persistent radare2 + PalmTree cache directory. Matching-only "
            "parameter changes reuse this cache."
        ),
    )
    parser.add_argument(
        "--no_analysis_cache",
        "--no-analysis-cache",
        dest="no_analysis_cache",
        action="store_true",
        help="Disable reading and writing the persistent analysis cache.",
    )
    parser.add_argument(
        "--analysis_cache_only",
        "--analysis-cache-only",
        dest="analysis_cache_only",
        action="store_true",
        help=(
            "Use the packaged cache identity without invoking radare2 and "
            "fail immediately instead of analyzing any cache miss."
        ),
    )
    parser.add_argument(
        "--library_score_aggregator",
        "--library-score-aggregator",
        dest="library_score_aggregator",
        choices=LIBRARY_SCORE_AGGREGATORS,
        default="top3_noisy_or",
        help=(
            "How accepted CU block scores are reduced to one library score. "
            "mean preserves the previous pipeline behavior"
        ),
    )
    parser.add_argument(
        "--library_min_score",
        type=float,
        default=0.975,
        help=(
            "Minimum aggregated block score across accepted CUs required to declare "
            "a library present; 0 preserves the legacy any-CU rule"
        )
    )
    parser.add_argument(
        "--cross_cu_call_bonus_weight",
        type=float,
        default=0.20,
        help=(
            "Maximum bounded library-score bonus for preserved direct calls "
            "between matched CUs; 0 keeps the new evidence diagnostic-only"
        ),
    )
    parser.add_argument(
        "--cross_cu_call_penalty_weight",
        type=float,
        default=0.0,
        help=(
            "Maximum bounded penalty for contradicted evaluable cross-CU calls; "
            "the default is 0 until negative evidence is validated offline"
        ),
    )
    parser.add_argument(
        "--cross_cu_call_saturation_edges",
        type=int,
        default=8,
        help="Evaluable cross-CU edges required for full relation reliability",
    )
    parser.add_argument(
        "--block_threshold",
        type=float,
        default=0.90,
        help=(
            "Per-block similarity threshold used by coverage_ratio and the "
            "one-to-one assignment_ratio"
        )
    )
    parser.add_argument(
        "--block_coverage_mean_threshold",
        type=float,
        default=0.975,
        help="Minimum mean best-block similarity"
    )
    parser.add_argument(
        "--block_assignment_quality_threshold",
        "--block_assignment_threshold",
        dest="block_assignment_quality_threshold",
        type=float,
        default=0.875,
        help="Minimum mean similarity of the one-to-one Hungarian block pairs"
    )
    parser.add_argument(
        "--block_min_assignment_ratio",
        type=float,
        default=0.40,
        help=(
            "Minimum ratio of target blocks covered by distinct Hungarian pairs "
            "whose similarity reaches block_threshold"
        )
    )
    parser.add_argument(
        "--block_min_coverage_ratio",
        type=float,
        default=0.65,
        help="Minimum ratio of target CU blocks with similarity >= block_threshold"
    )
    parser.add_argument(
        "--block_locality_window_multiplier",
        type=float,
        default=2.5,
        help="Source-function window multiplier for all-CU block matching"
    )
    parser.add_argument(
        "--block_locality_window_padding",
        type=int,
        default=6,
        help="Extra source functions added to the all-CU block-locality window"
    )
    parser.add_argument(
        "--block_min_call_edge_ratio",
        "--block_min_edge_locality_ratio",
        dest="block_min_call_edge_ratio",
        type=float,
        default=0.30,
        help=(
            "Minimum ratio of observable internal direct-call edges preserved "
            "between block-mapped functions; uses numeric addresses, not symbols"
        )
    )
    parser.add_argument(
        "--block_min_instructions",
        type=int,
        default=5,
        help="Minimum number of instructions required for a basic block to be used in block matching"
    )
    parser.add_argument(
        "--block_min_function_concentration",
        type=float,
        default=0.75,
        help="Minimum ratio of target blocks that must stay in each target function's dominant source function"
    )
    parser.add_argument(
        "--block_min_function_spread",
        type=float,
        default=0.30,
        help="Minimum ratio of distinct dominant source functions to mapped target functions"
    )
    parser.add_argument(
        "--cu_min_function_coverage",
        "--cu-min-function-coverage",
        dest="cu_min_function_coverage",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of reference-CU functions represented by the "
            "block-derived mapping; 0 disables this gate"
        ),
    )
    parser.add_argument(
        "--disable_block_window_prefilter",
        action="store_true",
        help="Evaluate assignment and call-graph metrics for every source-function window"
    )
    parser.add_argument(
        "--fast_negative_bound",
        action="store_true",
        help=(
            "Skip diagnostic window replay when global ELF block coverage "
            "proves that a CU cannot meet the mandatory coverage gates."
        ),
    )
    rodata_filter_group = parser.add_mutually_exclusive_group()
    rodata_filter_group.add_argument(
        "--disable_rodata_filter",
        dest="disable_rodata_filter",
        action="store_true",
        help="Disable .rodata-based penalty on block-level CU matches",
    )
    rodata_filter_group.add_argument(
        "--enable_rodata_filter",
        dest="disable_rodata_filter",
        action="store_false",
        help="Enable .rodata-based penalty on block-level CU matches",
    )
    parser.set_defaults(disable_rodata_filter=False)
    parser.add_argument(
        "--rodata_min_bytes",
        type=int,
        default=512,
        help="Minimum target .rodata bytes for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_min_strings",
        type=int,
        default=16,
        help="Minimum target .rodata strings for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_min_ngrams",
        type=int,
        default=32,
        help="Minimum target .rodata byte n-grams for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_penalty_threshold",
        type=float,
        default=0.50,
        help=(
            "Drop a block-passed CU when informative .rodata score is at or "
            "below this value"
        )
    )
    parser.add_argument(
        "--rodata_confirm_threshold",
        type=float,
        default=0.90,
        help="Mark a CU as .rodata-confirmed when informative .rodata score is at or above this value"
    )
    parser.add_argument(
        "--rodata_bonus_weight",
        type=float,
        default=0.30,
        help=(
            "Maximum bounded bonus applied to block scores for confirmed "
            ".rodata; 0 disables the bonus"
        ),
    )
    parser.add_argument("--rodata_string_weight", type=float, default=0.70)
    parser.add_argument("--rodata_byte_only_weight", type=float, default=0.50)
    args = parser.parse_args()
    if args.no_analysis_cache and args.analysis_cache_only:
        parser.error("--analysis-cache-only cannot be used with --no-analysis-cache")
    if not 0.0 <= args.cu_min_function_coverage <= 1.0:
        parser.error("--cu-min-function-coverage must be in [0, 1]")
    if not 0.0 <= args.rodata_string_weight <= 1.0:
        parser.error("--rodata_string_weight must be in [0, 1]")
    if not 0.0 <= args.rodata_byte_only_weight <= 1.0:
        parser.error("--rodata_byte_only_weight must be in [0, 1]")
    return args


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
        output_path.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if str(output_file).endswith(".gz") else open
    with opener(output_file, "wt", encoding="utf-8"):
        pass


def log_parse_debug(code_unit: CodeUnit) -> None:
    print(
        f"[DEBUG] parsed {code_unit.name}: "
        f"type={code_unit.unit_type}, "
        f"functions={code_unit.get_num_functions()}, "
        f"symbol_fallback_count={code_unit.symbol_fallback_count}, "
        f"pseudo_block_fallback_count={code_unit.pseudo_block_fallback_count}, "
        f"rodata_sections={code_unit.rodata_section_count}, "
        f"rodata_bytes={code_unit.get_rodata_size()}, "
        f"rodata_strings={len(code_unit.rodata_strings)}"
    )


def empty_rodata_result(target_unit: CodeUnit) -> RodataMatchResult:
    """Return a neutral .rodata result for CUs where .rodata was not checked."""
    return RodataMatchResult(
        score=0.0,
        string_score=0.0,
        byte_score=0.0,
        target_bytes=target_unit.get_rodata_size(),
        source_bytes=0,
        matched_strings=0,
        target_strings=len(target_unit.rodata_strings),
        matched_ngrams=0,
        target_ngrams=0,
        has_rodata=bool(target_unit.get_rodata_size() or target_unit.rodata_strings),
    )


def cu_status(block_result, rodata_evidence: RodataEvidence, rodata_disabled: bool) -> str:
    """Return the report status for a candidate CU."""
    if not block_result.passed:
        return "DROP"
    if not rodata_disabled and rodata_evidence.status == "penalty":
        return "DROP_RODATA"
    return "PASS"


def should_evaluate_rodata(block_result, rodata_disabled: bool) -> bool:
    """Evaluate .rodata only after the cheaper block gates have passed."""
    return bool(block_result.passed and not rodata_disabled)


def format_block_cu_line(
    index: int,
    target_unit: CodeUnit,
    block_status: str,
    rodata_evidence: RodataEvidence,
) -> str:
    """Format one compact CU-level report line."""
    return (
        f"    BLOCK_CU[{index:03}] {block_status} "
        f"name={target_unit.name} "
        f"rodata_status={rodata_evidence.status} "
        f"target_functions={target_unit.get_num_functions()}"
    )


def block_cu_feature_record(
    binary_path: str,
    library_name: str,
    source_unit: CodeUnit,
    target_unit_index: int,
    target_unit: CodeUnit,
    block_result: BlockMatchResult,
    library_call_edges: tuple[InterCUCallEdge, ...],
    rodata_result: RodataMatchResult,
    block_status: str,
    asm_normalization: str,
    palmtree_pooling: str,
    rodata_status: str | None = None,
    offline_ablation_features: bool = False,
) -> dict:
    """Build an autonomous CU/function record without changing decisions.

    Normal records retain verbose accepted-CU diagnostics and compact rejected
    diagnostics. Replay records instead use indexed function mappings and keep
    the Pareto windows required by the fixed-profile offline ablation.
    """
    def function_descriptor(function, index: int) -> dict:
        return {
            "index": int(index),
            "name": str(function.name),
            "address": int(function.address),
            "size": int(function.size),
            "blocks": int(function.get_num_blocks()),
            "instructions": int(function.get_num_instructions()),
        }

    def serialize_function_matches(function_matches) -> list[dict]:
        serialized = []
        for function_match in function_matches:
            target_index = int(function_match.target_function_index)
            source_index = int(function_match.source_function_index)
            if not 0 <= target_index < len(target_unit.functions):
                continue
            if not 0 <= source_index < len(source_unit.functions):
                continue
            source_function = source_unit.functions[source_index]
            serialized.append({
                "target_function_index": target_index,
                "source_function_index": source_index,
                "reference_function": function_descriptor(
                    target_unit.functions[target_index], target_index
                ),
                "elf_function": function_descriptor(
                    source_function, source_index
                ),
                "dominant_ratio": float(function_match.dominant_ratio),
                "coverage_mean": float(function_match.coverage_mean),
                "coverage_ratio": float(function_match.coverage_ratio),
                "source_call_targets": sorted(
                    int(index)
                    for index, candidate in enumerate(source_unit.functions)
                    if candidate.address in source_function.resolved_call_targets
                ),
            })
        return serialized

    common_record = {
        "schema_version": 3 if offline_ablation_features else 2,
        "type": "block_cu",
        "binary_path": binary_path,
        "library": Path(library_name).name,
        "asm_normalization": asm_normalization,
        "palmtree_pooling": palmtree_pooling,
        "name": target_unit.name,
        "target_cu_index": int(target_unit_index),
        "cu_status": block_status,
        "rodata_status": rodata_status or (
            "penalty" if block_status == "DROP_RODATA" else "unknown"
        ),
    }

    def compact_function_mapping(function_matches) -> list[dict]:
        mappings: list[dict] = []
        for function_match in function_matches:
            target_index = int(function_match.target_function_index)
            source_index = int(function_match.source_function_index)
            if (
                0 <= target_index < len(target_unit.functions)
                and 0 <= source_index < len(source_unit.functions)
            ):
                mappings.append({
                    "target_function_index": target_index,
                    "source_function_index": source_index,
                    "dominant_ratio": float(function_match.dominant_ratio),
                    "coverage_mean": float(function_match.coverage_mean),
                    "coverage_ratio": float(function_match.coverage_ratio),
                })
        return mappings

    def best_compact_function_match(function_matches) -> dict | None:
        mappings = compact_function_mapping(function_matches)
        best = max(
            mappings,
            key=lambda match: (
                match["dominant_ratio"],
                match["coverage_mean"],
                match["coverage_ratio"],
                -match["target_function_index"],
                -match["source_function_index"],
            ),
            default=None,
        )
        if best is None:
            return None
        target_index = int(best["target_function_index"])
        return {
            **best,
            # The ELF descriptor is already present once in the file header.
            # Retaining only the reference descriptor preserves the strongest
            # diagnostic hint for CUs rejected by B without cataloguing every
            # function of every rejected CU.
            "reference_function": function_descriptor(
                target_unit.functions[target_index], target_index
            ),
        }

    def replay_window(window) -> dict:
        return {
            "window_start": int(window.window_start),
            "window_stop": int(window.window_stop),
            "block_start": int(window.block_start),
            "block_stop": int(window.block_stop),
            "coverage_mean": float(window.coverage_mean),
            "coverage_min": float(window.coverage_min),
            "coverage_ratio": float(window.coverage_ratio),
            "assignment_quality": float(window.assignment_quality),
            "assignment_ratio": float(window.assignment_ratio),
            "call_edge_ratio": float(window.call_edge_ratio),
            "call_edges_evaluated": int(window.call_edges_evaluated),
            "call_edges_total": int(window.call_edges_total),
            "function_concentration": float(window.function_concentration),
            "function_spread": float(window.function_spread),
            "function_coverage": float(
                getattr(window, "function_coverage", 0.0)
            ),
            "function_matches": compact_function_mapping(
                window.function_matches
            ),
        }

    if offline_ablation_features:
        replay_windows = [
            replay_window(window)
            for window in (getattr(block_result, "windows", ()) or ())
        ]
        outgoing_inter_cu_calls = [
            {
                "caller_function_index": int(edge.caller_function_index),
                "callee_cu_index": int(edge.callee_cu_index),
                "callee_function_index": int(edge.callee_function_index),
            }
            for edge in library_call_edges
            if edge.caller_cu_index == target_unit_index
        ]
        record = {
            **common_record,
            "record_detail": "replay",
            "target_function_count": int(target_unit.get_num_functions()),
            "best_score": float(block_result.score),
            "windows_total": int(block_result.windows_total),
            "windows_evaluated": int(block_result.windows_evaluated),
            "windows_skipped": int(block_result.windows_skipped),
            "windows": replay_windows,
            "rodata": float(rodata_result.score),
            "rodata_string_score": float(getattr(rodata_result, "string_score", 0.0)),
            "rodata_byte_score": float(getattr(rodata_result, "byte_score", 0.0)),
            "rodata_string_informative": bool(getattr(rodata_result, "string_informative", False)),
            "rodata_byte_informative": bool(getattr(rodata_result, "byte_informative", False)),
            "rodata_has_rodata": bool(rodata_result.has_rodata),
            "rodata_strings": int(rodata_result.target_strings),
            "rodata_ngrams": int(rodata_result.target_ngrams),
            "rodata_bytes": int(rodata_result.target_bytes),
            "rodata_measured": bool(
                replay_windows
            ),
            "inter_cu_calls": outgoing_inter_cu_calls,
            "selected_match": {
                "passed": bool(block_result.passed),
                "score": float(block_result.score),
                "coverage_mean": float(block_result.coverage_mean),
                "coverage_min": float(block_result.coverage_min),
                "coverage_ratio": float(block_result.coverage_ratio),
                "assignment_quality": float(block_result.assignment_quality),
                "assignment_ratio": float(block_result.assignment_ratio),
                "call_edge_ratio": float(block_result.call_edge_ratio),
                "call_edges_evaluated": int(
                    block_result.call_edges_evaluated
                ),
                "call_edges_total": int(block_result.call_edges_total),
                "function_concentration": float(
                    block_result.function_concentration
                ),
                "function_spread": float(block_result.function_spread),
                "function_coverage": float(
                    getattr(block_result, "function_coverage", 0.0)
                ),
                "function_matches": compact_function_mapping(
                    block_result.function_matches
                ) if replay_windows else [],
            },
        }
        if replay_windows:
            # Function names and size/shape fields are needed by the planned
            # function-mapping quality experiment. They are stored once per
            # B-qualified CU and referenced by index from every Pareto window.
            record["target_functions"] = [
                function_descriptor(function, index)
                for index, function in enumerate(target_unit.functions)
            ]
        else:
            record["best_function_match"] = best_compact_function_match(
                block_result.function_matches
            )
        return record

    if block_status != "PASS":
        function_matches = serialize_function_matches(
            block_result.function_matches
        )
        best_function_match = max(
            function_matches,
            key=lambda match: (
                match["dominant_ratio"],
                match["coverage_mean"],
                match["coverage_ratio"],
                -match["target_function_index"],
                -match["source_function_index"],
            ),
            default=None,
        )
        return {
            **common_record,
            "record_detail": "compact",
            "target_function_count": int(target_unit.get_num_functions()),
            "rejection_reason": (
                "rodata_penalty"
                if block_status == "DROP_RODATA"
                else "block_gates"
            ),
            "best_score": float(block_result.score),
            "coverage_mean": float(block_result.coverage_mean),
            "coverage_min": float(block_result.coverage_min),
            "coverage_ratio": float(block_result.coverage_ratio),
            "assignment_quality": float(block_result.assignment_quality),
            "assignment_ratio": float(block_result.assignment_ratio),
            "call_edge_ratio": float(block_result.call_edge_ratio),
            "call_edges_evaluated": int(block_result.call_edges_evaluated),
            "call_edges_total": int(block_result.call_edges_total),
            "function_concentration": float(
                block_result.function_concentration
            ),
            "function_spread": float(block_result.function_spread),
            "window_start": int(block_result.selected_window_start),
            "window_stop": int(block_result.selected_window_stop),
            "block_start": int(block_result.selected_block_start),
            "block_stop": int(block_result.selected_block_stop),
            "windows_total": int(block_result.windows_total),
            "windows_evaluated": int(block_result.windows_evaluated),
            "windows_skipped": int(block_result.windows_skipped),
            "best_function_match": best_function_match,
            "rodata": float(rodata_result.score),
        }

    windows = []
    for window in (getattr(block_result, "windows", ()) or ()):
        windows.append(
            {
                "window_start": int(window.window_start),
                "window_stop": int(window.window_stop),
                "block_start": int(window.block_start),
                "block_stop": int(window.block_stop),
                "coverage_mean": float(window.coverage_mean),
                "coverage_ratio": float(window.coverage_ratio),
                "assignment_quality": float(window.assignment_quality),
                "assignment_ratio": float(window.assignment_ratio),
                "call_edge_ratio": float(window.call_edge_ratio),
                "function_concentration": float(window.function_concentration),
                "function_spread": float(window.function_spread),
                "function_coverage": float(
                    getattr(window, "function_coverage", 0.0)
                ),
                "function_matches": serialize_function_matches(
                    window.function_matches
                ),
            }
        )

    outgoing_inter_cu_calls = [
        {
            "caller_function_index": edge.caller_function_index,
            "callee_cu_index": edge.callee_cu_index,
            "callee_function_index": edge.callee_function_index,
        }
        for edge in library_call_edges
        if edge.caller_cu_index == target_unit_index
    ]
    record = {
        **common_record,
        "record_detail": "full",
        "target_functions": [
            function_descriptor(function, index)
            for index, function in enumerate(target_unit.functions)
        ],
        "rodata": float(rodata_result.score),
        "rodata_string_score": float(getattr(rodata_result, "string_score", 0.0)),
        "rodata_byte_score": float(getattr(rodata_result, "byte_score", 0.0)),
        "rodata_string_informative": bool(getattr(rodata_result, "string_informative", False)),
        "rodata_byte_informative": bool(getattr(rodata_result, "byte_informative", False)),
        "rodata_has_rodata": bool(rodata_result.has_rodata),
        "rodata_strings": int(rodata_result.target_strings),
        "rodata_ngrams": int(rodata_result.target_ngrams),
        "rodata_bytes": int(rodata_result.target_bytes),
        "windows": windows,
        "inter_cu_calls": outgoing_inter_cu_calls,
        "selected_match": {
            "passed": bool(block_result.passed),
            "score": float(block_result.score),
            "coverage_mean": float(block_result.coverage_mean),
            "coverage_min": float(block_result.coverage_min),
            "coverage_ratio": float(block_result.coverage_ratio),
            "assignment_quality": float(block_result.assignment_quality),
            "assignment_ratio": float(block_result.assignment_ratio),
            "call_edge_ratio": float(block_result.call_edge_ratio),
            "call_edges_evaluated": int(block_result.call_edges_evaluated),
            "call_edges_total": int(block_result.call_edges_total),
            "function_concentration": float(
                block_result.function_concentration
            ),
            "function_spread": float(block_result.function_spread),
            "function_coverage": float(
                getattr(block_result, "function_coverage", 0.0)
            ),
            "window_start": int(block_result.selected_window_start),
            "window_stop": int(block_result.selected_window_stop),
            "block_start": int(block_result.selected_block_start),
            "block_stop": int(block_result.selected_block_stop),
            "function_matches": serialize_function_matches(
                block_result.function_matches
            ),
        },
    }
    if not windows:
        record.update(
            {
                "coverage_mean": float(block_result.coverage_mean),
                "coverage_ratio": float(block_result.coverage_ratio),
                "assignment_quality": float(block_result.assignment_quality),
                "assignment_ratio": float(block_result.assignment_ratio),
                "call_edge_ratio": float(block_result.call_edge_ratio),
                "function_concentration": float(
                    block_result.function_concentration
                ),
                "function_spread": float(block_result.function_spread),
                "function_coverage": float(
                    getattr(block_result, "function_coverage", 0.0)
                ),
                "function_matches": serialize_function_matches(
                    block_result.function_matches
                ),
            }
        )
    return record


def release_library_units(*containers: list) -> None:
    """Release mmap-backed CUs before the library generator advances."""
    for container in containers:
        container.clear()
    gc.collect()


def process_bin(
    binary: CodeUnit,
    lib,
    library_call_edges: dict[str, tuple[InterCUCallEdge, ...]],
    args,
):
    """Process a parsed binary against all libraries."""
    output_file = args.output if args.output else None
    features_opener = (
        gzip.open
        if args.features_output and str(args.features_output).endswith(".gz")
        else open
    )
    features_context = (
        features_opener(args.features_output, "at", encoding="utf-8")
        if args.features_output
        else nullcontext(None)
    )
    binary_path = str(Path(args.path_to_binary).resolve())
    binary_rodata_index = None
    prepared_binary = prepare_block_unit(
        binary,
        args.block_min_instructions,
    )

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    with features_context as features_file:
        features_requested = features_file is not None
        source_call_targets: dict[int, list[int]] = {}
        if features_requested:
            source_index_by_address = {
                function.address: index
                for index, function in enumerate(binary.functions)
            }
            source_call_targets = {
                index: sorted(
                    source_index_by_address[address]
                    for address in function.resolved_call_targets
                    if address in source_index_by_address
                )
                for index, function in enumerate(binary.functions)
                if function.resolved_call_targets
            }
            print(
                json.dumps(
                    {
                        "schema_version": (
                            3 if args.offline_ablation_features else 2
                        ),
                        "type": "source_call_targets",
                        "binary_path": binary_path,
                        **({
                            "feature_mode": "replay_complete",
                            "matching_configuration": {
                                "asm_normalization": args.asm_normalization,
                                "palmtree_pooling": args.palmtree_pooling,
                                "library_score_aggregator": args.library_score_aggregator,
                                "library_min_score": args.library_min_score,
                                "cross_cu_call_bonus_weight": args.cross_cu_call_bonus_weight,
                                "cross_cu_call_penalty_weight": args.cross_cu_call_penalty_weight,
                                "cross_cu_call_saturation_edges": args.cross_cu_call_saturation_edges,
                                "block_threshold": args.block_threshold,
                                "block_coverage_mean_threshold": args.block_coverage_mean_threshold,
                                "block_assignment_quality_threshold": args.block_assignment_quality_threshold,
                                "block_min_assignment_ratio": args.block_min_assignment_ratio,
                                "block_min_coverage_ratio": args.block_min_coverage_ratio,
                                "block_locality_window_multiplier": args.block_locality_window_multiplier,
                                "block_locality_window_padding": args.block_locality_window_padding,
                                "block_min_call_edge_ratio": args.block_min_call_edge_ratio,
                                "block_min_instructions": args.block_min_instructions,
                                "block_min_function_concentration": args.block_min_function_concentration,
                                "block_min_function_spread": args.block_min_function_spread,
                                "cu_min_function_coverage": args.cu_min_function_coverage,
                                "disable_block_window_prefilter": bool(
                                    args.disable_block_window_prefilter
                                ),
                                "fast_negative_bound": bool(
                                    args.fast_negative_bound
                                ),
                                "rodata_filter_enabled": int(
                                    not args.disable_rodata_filter
                                ),
                                "rodata_min_bytes": args.rodata_min_bytes,
                                "rodata_min_strings": args.rodata_min_strings,
                                "rodata_min_ngrams": args.rodata_min_ngrams,
                                "rodata_penalty_threshold": args.rodata_penalty_threshold,
                                "rodata_confirm_threshold": args.rodata_confirm_threshold,
                                "rodata_bonus_weight": args.rodata_bonus_weight,
                                "rodata_string_weight": args.rodata_string_weight,
                                "rodata_byte_only_weight": args.rodata_byte_only_weight,
                            },
                        } if args.offline_ablation_features else {}),
                        "functions": [
                            {
                                "index": int(index),
                                "name": str(function.name),
                                "address": int(function.address),
                                "size": int(function.size),
                                "blocks": int(function.get_num_blocks()),
                                "instructions": int(
                                    function.get_num_instructions()
                                ),
                            }
                            for index, function in enumerate(binary.functions)
                        ],
                        "calls": [
                            [source_index, targets]
                            for source_index, targets in source_call_targets.items()
                        ],
                    },
                    separators=(",", ":"),
                ),
                file=features_file,
            )
        for library_name, comp_units in lib.items():
            start_library = time.time()
            target_comp_units = [
                comp_unit
                for comp_unit in comp_units
                if comp_unit.get_num_functions() > 1
            ]

            if not target_comp_units:
                stop_library = time.time()
                elapsed_time = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(stop_library - start_library),
                )
                line = (
                    f"{'NO':7} | "
                    f"library={Path(library_name).name} | "
                    f"score={0.0:6.2f}% | "
                    f"block_best_any={0.0:6.2f}% | "
                    f"block_best_matched={0.0:6.2f}% | "
                    f"matched_cu=0/0 | "
                    f"matched_functions=0 | "
                    f"time={elapsed_time} | "
                    f"no multi-function compilation units"
                )
                log_line(line, output_file)
                # The generator cannot release its mmap-backed CodeUnits until
                # the next iteration.  Drop the caller's references first so a
                # large archive does not consume the file-descriptor budget
                # while the following archive is being loaded.
                release_library_units(target_comp_units, comp_units)
                continue

            block_results = []
            rodata_evidence_by_cu = {}
            rodata_result_by_cu = {}

            call_edges = library_call_edges.get(library_name, ())
            call_edges_by_cu: dict[int, list[InterCUCallEdge]] = {}
            for edge in call_edges:
                call_edges_by_cu.setdefault(
                    int(edge.caller_cu_index), []
                ).append(edge)
            for target_unit_index, target_unit in enumerate(target_comp_units):
                block_result = evaluate_block_presence(
                    source_unit=binary,
                    target_unit=target_unit,
                    block_threshold=args.block_threshold,
                    min_assignment_quality=(
                        args.block_assignment_quality_threshold
                    ),
                    min_assignment_ratio=args.block_min_assignment_ratio,
                    min_coverage_ratio=args.block_min_coverage_ratio,
                    min_coverage_mean=args.block_coverage_mean_threshold,
                    locality_window_multiplier=args.block_locality_window_multiplier,
                    locality_window_padding=args.block_locality_window_padding,
                    min_call_edge_ratio=args.block_min_call_edge_ratio,
                    min_block_instructions=args.block_min_instructions,
                    min_function_concentration=args.block_min_function_concentration,
                    min_function_spread=args.block_min_function_spread,
                    min_function_coverage=args.cu_min_function_coverage,
                    enable_window_prefilter=not args.disable_block_window_prefilter,
                    fast_negative_bound=args.fast_negative_bound,
                    prepared_source=prepared_binary,
                )
                block_results.append((target_unit, block_result))

                measure_rodata = should_evaluate_rodata(
                    block_result,
                    args.disable_rodata_filter,
                ) or (
                    args.offline_ablation_features
                    and bool(block_result.windows)
                )
                if measure_rodata:
                    if binary_rodata_index is None:
                        binary_rodata_index = build_rodata_index(binary)
                    rodata_result = evaluate_rodata_match(
                        binary_rodata_index,
                        target_unit,
                        string_weight=args.rodata_string_weight,
                        byte_only_weight=args.rodata_byte_only_weight,
                    )
                    classified_rodata_evidence = classify_rodata_evidence(
                        rodata_result,
                        args.rodata_min_bytes,
                        args.rodata_min_strings,
                        args.rodata_min_ngrams,
                        args.rodata_penalty_threshold,
                        args.rodata_confirm_threshold,
                    )
                    if not block_result.passed:
                        # Raw evidence is retained for replay, while the online
                        # pipeline still records that R was never consulted.
                        rodata_evidence = RodataEvidence(
                            status="skipped",
                            informative=False,
                        )
                    elif args.disable_rodata_filter:
                        rodata_evidence = RodataEvidence(
                            status="disabled",
                            informative=classified_rodata_evidence.informative,
                        )
                    else:
                        rodata_evidence = classified_rodata_evidence
                else:
                    rodata_result = empty_rodata_result(target_unit)
                    rodata_evidence = RodataEvidence(
                        status="disabled" if args.disable_rodata_filter else "skipped",
                        informative=False,
                    )

                rodata_evidence_by_cu[id(target_unit)] = rodata_evidence
                rodata_result_by_cu[id(target_unit)] = rodata_result
                block_status = cu_status(
                    block_result,
                    rodata_evidence,
                    args.disable_rodata_filter,
                )
                if features_file and not args.offline_ablation_features:
                    print(
                        json.dumps(
                            block_cu_feature_record(
                                binary_path,
                                library_name,
                                binary,
                                target_unit_index,
                                target_unit,
                                block_result,
                                tuple(
                                    call_edges_by_cu.get(
                                        target_unit_index, ()
                                    )
                                ),
                                rodata_result,
                                block_status,
                                args.asm_normalization,
                                args.palmtree_pooling,
                                rodata_status=rodata_evidence.status,
                                offline_ablation_features=(
                                    args.offline_ablation_features
                                ),
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        file=features_file,
                    )

            if features_file and args.offline_ablation_features:
                # B remains present in every supported ablation. A CU with no
                # B-qualified window can therefore never participate in a
                # replayed cross-CU edge. Keeping only the induced subgraph
                # avoids duplicating the complete static archive call graph in
                # every ELF feature log.
                b_qualified_cu_indices = {
                    index
                    for index, (_target_unit, block_result) in enumerate(
                        block_results
                    )
                    if block_result.windows
                }
                for target_unit_index, (target_unit, block_result) in enumerate(
                    block_results
                ):
                    rodata_evidence = rodata_evidence_by_cu[id(target_unit)]
                    replay_edges = (
                        tuple(
                            edge
                            for edge in call_edges_by_cu.get(
                                target_unit_index, ()
                            )
                            if edge.callee_cu_index in b_qualified_cu_indices
                        )
                        if target_unit_index in b_qualified_cu_indices
                        else ()
                    )
                    print(
                        json.dumps(
                            block_cu_feature_record(
                                binary_path,
                                library_name,
                                binary,
                                target_unit_index,
                                target_unit,
                                block_result,
                                replay_edges,
                                rodata_result_by_cu[id(target_unit)],
                                cu_status(
                                    block_result,
                                    rodata_evidence,
                                    args.disable_rodata_filter,
                                ),
                                args.asm_normalization,
                                args.palmtree_pooling,
                                rodata_status=rodata_evidence.status,
                                offline_ablation_features=True,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        file=features_file,
                    )

            successful_block_results = [
                (target_unit, block_result)
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and (
                        args.disable_rodata_filter
                        or rodata_evidence_by_cu[id(target_unit)].status != "penalty"
                    )
                )
            ]
            block_best_score = max(
                (block_result.score for _, block_result in block_results),
                default=0.0,
            )
            successful_block_best_score = max(
                (block_result.score for _, block_result in successful_block_results),
                default=0.0,
            )
            adjusted_successful_scores = [
                apply_rodata_bonus(
                    block_result.score,
                    rodata_result_by_cu[id(target_unit)],
                    rodata_evidence_by_cu[id(target_unit)],
                    args.rodata_bonus_weight,
                    args.rodata_confirm_threshold,
                )
                for target_unit, block_result in successful_block_results
            ]
            aggregation_scores = adjusted_successful_scores
            if args.library_score_aggregator == "top3_mean":
                # Fix the denominator before applying .rodata. A rejected CU
                # contributes zero, so negative evidence cannot increase the
                # library score by removing a low-scoring value from the mean.
                aggregation_scores = [
                    (
                        0.0
                        if rodata_evidence_by_cu[id(target_unit)].status
                        == "penalty"
                        else apply_rodata_bonus(
                            block_result.score,
                            rodata_result_by_cu[id(target_unit)],
                            rodata_evidence_by_cu[id(target_unit)],
                            args.rodata_bonus_weight,
                            args.rodata_confirm_threshold,
                        )
                    )
                    for target_unit, block_result in block_results
                    if block_result.passed
                ]
            score_tot = aggregate_library_score(
                aggregation_scores,
                args.library_score_aggregator,
                score_floor=args.block_coverage_mean_threshold,
            )
            base_score_tot = score_tot
            accepted_cu_indices = {
                index
                for index, (target_unit, block_result) in enumerate(block_results)
                if (
                    block_result.passed
                    and (
                        args.disable_rodata_filter
                        or rodata_evidence_by_cu[id(target_unit)].status != "penalty"
                    )
                )
            }
            cross_cu_evidence = cross_cu_call_evidence(
                binary.functions,
                target_comp_units,
                [result for _, result in block_results],
                call_edges,
                included_cu_indices=accepted_cu_indices,
            )
            anchored_cross_cu_evidence = cross_cu_call_evidence(
                binary.functions,
                target_comp_units,
                [result for _, result in block_results],
                call_edges,
                included_cu_indices=accepted_cu_indices,
                require_both_included=False,
            )
            score_tot = apply_cross_cu_call_adjustment(
                score_tot,
                cross_cu_evidence,
                bonus_weight=args.cross_cu_call_bonus_weight,
                penalty_weight=args.cross_cu_call_penalty_weight,
                saturation_edges=args.cross_cu_call_saturation_edges,
            )

            stop_library = time.time()
            elapsed_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(stop_library - start_library),
            )

            matched_cu = len(successful_block_results)
            matched_functions = sum(
                len(block_result.function_matches)
                for _, block_result in successful_block_results
            )
            total_cu = len(target_comp_units)
            percentage = score_tot * 100.0

            status = (
                "YES"
                if matched_cu >= 1 and score_tot >= args.library_min_score
                else "NO"
            )

            rodata_confirmed_count = sum(
                1
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and rodata_evidence_by_cu[id(target_unit)].status == "confirm"
                )
            )
            rodata_penalty_count = sum(
                1
                for target_unit, block_result in block_results
                if (
                    block_result.passed
                    and rodata_evidence_by_cu[id(target_unit)].status == "penalty"
                )
            )
            line = (
                f"{status:7} | "
                f"library={Path(library_name).name} | "
                f"aggregator={args.library_score_aggregator} | "
                f"score={percentage:6.2f}% | "
                f"base_score={base_score_tot * 100.0:6.2f}% | "
                f"cross_cu_edges={cross_cu_evidence.matched_edges}/"
                f"{cross_cu_evidence.evaluable_edges}/"
                f"{cross_cu_evidence.expected_edges} | "
                f"cross_cu_ratio={cross_cu_evidence.ratio:.4f} | "
                f"cross_cu_coverage={cross_cu_evidence.coverage:.4f} | "
                f"cross_cu_anchored_edges="
                f"{anchored_cross_cu_evidence.matched_edges}/"
                f"{anchored_cross_cu_evidence.evaluable_edges}/"
                f"{anchored_cross_cu_evidence.expected_edges} | "
                f"block_best_any={block_best_score * 100.0:6.2f}% | "
                f"block_best_matched={successful_block_best_score * 100.0:6.2f}% | "
                f"rodata_confirmed_cu={rodata_confirmed_count} | "
                f"rodata_penalty_cu={rodata_penalty_count} | "
                f"matched_cu={matched_cu}/{total_cu} | "
                f"matched_functions={matched_functions} | "
                f"time={elapsed_time}"
            )

            for i, (target_unit, block_result) in enumerate(block_results):
                rodata_evidence = rodata_evidence_by_cu[id(target_unit)]
                log_line(
                    format_block_cu_line(
                        i,
                        target_unit,
                        cu_status(
                            block_result,
                            rodata_evidence,
                            args.disable_rodata_filter,
                        ),
                        rodata_evidence,
                    ),
                    output_file
                )

            log_line(line, output_file)
            # Each cached CodeUnit owns one embedding mmap.  All containers
            # holding CodeUnits must be emptied before LazyLibraries resumes
            # and starts loading the next archive.  In particular, accepted
            # results are a subset with their own references; retaining that
            # list can exhaust RLIMIT_NOFILE when two large archives are
            # adjacent and surface as a misleading cache-only miss.
            release_library_units(
                successful_block_results,
                block_results,
                target_comp_units,
                comp_units,
            )


def load_library_archive(
    library_file: Path,
    code_loader: CachedCodeUnitLoader,
) -> tuple[list[CodeUnit], bool]:
    """Load only multi-function CUs, extracting solely on cache miss."""
    comp_units = code_loader.load_archive(library_file, min_functions=2)
    if comp_units is not None:
        return comp_units, True

    comp_units = []
    archive_member_identities: list[tuple[str, int, dict]] = []
    extracted_objects = []
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            extracted_objects = extract_archive_members(library_file, Path(tmpdir))
        except Exception as error:
            print(f"[ERROR] ar failed on {library_file.name}")
            print(error)
            return comp_units, False

        for member in extracted_objects:
            obj_file = member.path
            try:
                identity = code_loader.identity(obj_file, CodeUnit.TYPE_CU)
                unit = code_loader.load(
                    obj_file,
                    unit_type=CodeUnit.TYPE_CU,
                    identity=identity,
                )
                archive_member_identities.append(
                    (member.name, member.occurrence, identity)
                )
                log_parse_debug(unit)
                if unit.get_num_functions() > 1:
                    comp_units.append(unit)
            except Exception as error:
                print(f"[WARN] Failed to parse {obj_file}: {error}")
        if len(archive_member_identities) == len(extracted_objects):
            code_loader.store_archive_index(library_file, archive_member_identities)
    return comp_units, False


class LazyLibraries:
    """Yield one library at a time so mmap-backed CUs do not accumulate in RAM."""

    def __init__(
        self,
        library_files: list[Path],
        code_loader: CachedCodeUnitLoader,
        call_edges: dict[str, tuple[InterCUCallEdge, ...]],
    ):
        self.library_files = library_files
        self.code_loader = code_loader
        self.call_edges = call_edges

    def items(self):
        for library_file in self.library_files:
            comp_units, archive_cache_hit = load_library_archive(
                library_file, self.code_loader
            )
            key = library_file.as_posix()
            self.call_edges[key] = resolve_archive_inter_cu_calls(comp_units)
            print(
                f"[DEBUG] {library_file.name}: loaded {len(comp_units)} "
                "multi-function object files "
                f"(archive-cache={'hit' if archive_cache_hit else 'miss'})"
            )
            print(
                f"[DEBUG] {library_file.name}: resolved "
                f"{len(self.call_edges[key])} direct cross-CU function edge(s)"
            )
            try:
                yield key, comp_units
            finally:
                self.call_edges.pop(key, None)
                del comp_units
                gc.collect()


if __name__ == "__main__":
    start_main = time.time()

    args = get_args()
    initialize_output_file(args.output)
    initialize_output_file(args.features_output)

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
        
    asm_model_path = Path(args.asm_model).expanduser()
    if not asm_model_path.is_file() and not asm_model_path.is_absolute():
        asm_model_path = SCRIPT_DIR / asm_model_path
    code_loader = CachedCodeUnitLoader(
        asm_model_path,
        device=args.device,
        pooling=args.palmtree_pooling,
        asm_normalization=args.asm_normalization,
        cache_dir=(None if args.no_analysis_cache else args.analysis_cache_dir),
        cache_only=args.analysis_cache_only,
    )
    print(f"PalmTree requested device: {args.device}")
    print(f"PalmTree pooling: {args.palmtree_pooling}")
    print(
        "Analysis cache: "
        + (
            "disabled"
            if code_loader.cache is None
            else code_loader.cache.root.as_posix()
        )
    )
    if args.analysis_cache_only:
        print("Analysis cache mode: cache-only (read-only; radare2 disabled)")

    library_call_edges: dict[str, tuple[InterCUCallEdge, ...]] = {}
    lib = LazyLibraries(library_files, code_loader, library_call_edges)

    log_line(f"Found {len(library_files)} libraries to process", args.output)
    log_line(f"Target binary: {binary_path}", args.output)
    log_line(f"Assembly normalization: {args.asm_normalization}", args.output)
    log_line(f"PalmTree pooling: {args.palmtree_pooling}", args.output)
    log_line(f"Library score aggregator: {args.library_score_aggregator}", args.output)
    log_line(
        "Feature collection mode: "
        + (
            "replay_complete"
            if args.offline_ablation_features
            else "standard"
        ),
        args.output,
    )
    log_line(f"Library minimum score: {args.library_min_score}", args.output)
    log_line(
        f"CU minimum function coverage: {args.cu_min_function_coverage}",
        args.output,
    )
    log_line(
        f"Cross-CU call bonus weight: {args.cross_cu_call_bonus_weight}",
        args.output,
    )
    log_line(
        f"Cross-CU call penalty weight: {args.cross_cu_call_penalty_weight}",
        args.output,
    )
    log_line(
        f"Cross-CU call saturation edges: {args.cross_cu_call_saturation_edges}",
        args.output,
    )
    log_line(f"Rodata penalty threshold: {args.rodata_penalty_threshold}", args.output)
    log_line(f"Rodata confirm threshold: {args.rodata_confirm_threshold}", args.output)
    log_line(f"Rodata bonus weight: {args.rodata_bonus_weight}", args.output)
    log_line(f"Rodata string weight: {args.rodata_string_weight}", args.output)
    log_line(f"Rodata byte-only weight: {args.rodata_byte_only_weight}", args.output)
    log_line(f"Start processing: {time.strftime('%H:%M:%S', time.gmtime())}\n", args.output)

    binary = code_loader.load(
        binary_path,
        unit_type=CodeUnit.TYPE_ELF,
    )
    log_parse_debug(binary)

    process_bin(binary, lib, library_call_edges, args)

    cache_stats = code_loader.stats
    print(
        "Analysis cache stats: "
        f"hits={cache_stats.hits}, misses={cache_stats.misses}, "
        f"writes={cache_stats.writes}, model_loaded={code_loader.model_loaded}"
    )

    elapsed_main = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_main))
    log_line(f"\nDone processing in {elapsed_main}", args.output)
