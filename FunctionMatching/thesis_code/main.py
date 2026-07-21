import time
import os
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import re
import tempfile
import subprocess

from model import PALMTREE_POOLING_MODES, PalmTree
from asm import ASM_NORMALIZATION_MODES, CodeUnit, parse_r2_file
from match import (
    BlockMatchResult,
    LIBRARY_SCORE_AGGREGATORS,
    RodataEvidence,
    RodataMatchResult,
    aggregate_library_score,
    build_rodata_index,
    apply_rodata_bonus,
    classify_rodata_evidence,
    evaluate_block_presence,
    evaluate_rodata_match,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR / "Test"
REPO_ROOT = SCRIPT_DIR.parents[1]

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
        "--library_score_aggregator",
        "--library-score-aggregator",
        dest="library_score_aggregator",
        choices=LIBRARY_SCORE_AGGREGATORS,
        default="mean",
        help=(
            "How accepted CU block scores are reduced to one library score. "
            "mean preserves the previous pipeline behavior"
        ),
    )
    parser.add_argument(
        "--library_min_score",
        type=float,
        default=0.0,
        help=(
            "Minimum aggregated block score across accepted CUs required to declare "
            "a library present; 0 preserves the legacy any-CU rule"
        )
    )
    parser.add_argument(
        "--block_threshold",
        type=float,
        default=0.70,
        help=(
            "Per-block similarity threshold used by coverage_ratio and the "
            "one-to-one assignment_ratio"
        )
    )
    parser.add_argument(
        "--block_coverage_mean_threshold",
        type=float,
        default=0.80,
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
        default=0.50,
        help=(
            "Minimum ratio of target blocks covered by distinct Hungarian pairs "
            "whose similarity reaches block_threshold"
        )
    )
    parser.add_argument(
        "--block_min_coverage_ratio",
        type=float,
        default=0.50,
        help="Minimum ratio of target CU blocks with similarity >= block_threshold"
    )
    parser.add_argument(
        "--block_locality_window_multiplier",
        type=float,
        default=5.0,
        help="Source-function window multiplier for all-CU block matching"
    )
    parser.add_argument(
        "--block_locality_window_padding",
        type=int,
        default=2,
        help="Extra source functions added to the all-CU block-locality window"
    )
    parser.add_argument(
        "--block_min_call_edge_ratio",
        "--block_min_edge_locality_ratio",
        dest="block_min_call_edge_ratio",
        type=float,
        default=0.50,
        help=(
            "Minimum ratio of observable internal direct-call edges preserved "
            "between block-mapped functions; uses numeric addresses, not symbols"
        )
    )
    parser.add_argument(
        "--block_min_instructions",
        type=int,
        default=3,
        help="Minimum number of instructions required for a basic block to be used in block matching"
    )
    parser.add_argument(
        "--block_min_function_concentration",
        type=float,
        default=0.45,
        help="Minimum ratio of target blocks that must stay in each target function's dominant source function"
    )
    parser.add_argument(
        "--block_min_function_spread",
        type=float,
        default=0.30,
        help="Minimum ratio of distinct dominant source functions to mapped target functions"
    )
    parser.add_argument(
        "--disable_block_window_prefilter",
        action="store_true",
        help="Evaluate assignment and call-graph metrics for every source-function window"
    )
    parser.add_argument(
        "--disable_rodata_filter",
        action="store_true",
        help="Disable .rodata-based penalty on block-level CU matches"
    )
    parser.add_argument(
        "--rodata_min_bytes",
        type=int,
        default=512,
        help="Minimum target .rodata bytes for .rodata evidence to be informative"
    )
    parser.add_argument(
        "--rodata_min_strings",
        type=int,
        default=0,
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
        default=0.0,
        help=(
            "Drop a block-passed CU when informative .rodata score is at or "
            "below this value; the offline-selected default 0.0 keeps the "
            "low-evidence branch without rejecting nonzero scores"
        )
    )
    parser.add_argument(
        "--rodata_confirm_threshold",
        type=float,
        default=0.70,
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
    return parser.parse_args()


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

    with open(output_file, "w", encoding="utf-8"):
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
    target_unit: CodeUnit,
    block_result: BlockMatchResult,
    rodata_result: RodataMatchResult,
    rodata_evidence: RodataEvidence,
    block_status: str,
    asm_normalization: str,
    palmtree_pooling: str,
) -> dict:
    """Build the full CU-level feature row used by threshold tuning."""
    windows = [
        window._asdict() if hasattr(window, "_asdict") else window
        for window in (getattr(block_result, "windows", ()) or ())
    ]
    return {
        "type": "block_cu",
        "binary_path": binary_path,
        "library": Path(library_name).name,
        "asm_normalization": asm_normalization,
        "palmtree_pooling": palmtree_pooling,
        "name": target_unit.name,
        "status": block_status,
        "block": float(block_result.score),
        "coverage_mean": float(block_result.coverage_mean),
        "coverage_min": float(block_result.coverage_min),
        "coverage_ratio": float(block_result.coverage_ratio),
        "assignment_quality": float(block_result.assignment_quality),
        "assignment_ratio": float(block_result.assignment_ratio),
        "rodata": float(rodata_result.score),
        "rodata_has_rodata": bool(rodata_result.has_rodata),
        "rodata_string_score": float(rodata_result.string_score),
        "rodata_byte_score": float(rodata_result.byte_score),
        "rodata_informative": bool(rodata_evidence.informative),
        "rodata_string_informative": bool(
            getattr(rodata_result, "string_informative", False)
        ),
        "rodata_byte_informative": bool(
            getattr(rodata_result, "byte_informative", False)
        ),
        "rodata_strings": int(rodata_result.target_strings),
        "rodata_matched_strings": int(rodata_result.matched_strings),
        "rodata_ngrams": int(rodata_result.target_ngrams),
        "rodata_matched_ngrams": int(rodata_result.matched_ngrams),
        "rodata_bytes": int(rodata_result.target_bytes),
        "rodata_status": rodata_evidence.status,
        "blocks_source": int(block_result.num_source_blocks),
        "blocks_target": int(block_result.num_target_blocks),
        "locality_span": int(block_result.locality_span),
        "windows_evaluated": int(block_result.windows_evaluated),
        "windows_total": int(block_result.windows_total),
        "windows_skipped": int(block_result.windows_skipped),
        "call_edge_ratio": float(block_result.call_edge_ratio),
        "call_edges_evaluated": int(block_result.call_edges_evaluated),
        "call_edges_total": int(block_result.call_edges_total),
        "function_concentration": float(block_result.function_concentration),
        "function_spread": float(block_result.function_spread),
        "windows": windows,
        "target_functions": int(target_unit.get_num_functions()),
    }


def process_bin(binary: CodeUnit, lib: dict[str, list[CodeUnit]], args):
    """Process a parsed binary against all libraries."""
    output_file = args.output if args.output else None
    features_context = (
        open(args.features_output, "a", encoding="utf-8")
        if args.features_output
        else nullcontext(None)
    )
    binary_path = str(Path(args.path_to_binary).resolve())
    binary_rodata_index = None

    if output_file:
        output_path = Path(output_file)
        if output_path.parent and not output_path.parent.exists():
            raise ValueError(f"Output directory '{output_path.parent}' does not exist")

    with features_context as features_file:
        features_requested = features_file is not None
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
                    f"time={elapsed_time} | "
                    f"no multi-function compilation units"
                )
                log_line(line, output_file)
                continue

            block_results = []
            rodata_evidence_by_cu = {}
            rodata_result_by_cu = {}

            for target_unit in target_comp_units:
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
                    enable_window_prefilter=not args.disable_block_window_prefilter,
                )
                block_results.append((target_unit, block_result))

                should_evaluate_rodata = features_requested or (
                    block_result.passed and not args.disable_rodata_filter
                )
                if should_evaluate_rodata:
                    if binary_rodata_index is None:
                        binary_rodata_index = build_rodata_index(binary)
                    rodata_result = evaluate_rodata_match(
                        binary_rodata_index,
                        target_unit,
                    )
                    classified_rodata_evidence = classify_rodata_evidence(
                        rodata_result,
                        args.rodata_min_bytes,
                        args.rodata_min_strings,
                        args.rodata_min_ngrams,
                        args.rodata_penalty_threshold,
                        args.rodata_confirm_threshold,
                    )
                    rodata_evidence = (
                        RodataEvidence(
                            status="disabled",
                            informative=classified_rodata_evidence.informative,
                        )
                        if args.disable_rodata_filter
                        else classified_rodata_evidence
                    )
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
                if features_file:
                    print(
                        json.dumps(
                            block_cu_feature_record(
                                binary_path,
                                library_name,
                                target_unit,
                                block_result,
                                rodata_result,
                                rodata_evidence,
                                block_status,
                                args.asm_normalization,
                                args.palmtree_pooling,
                            ),
                            sort_keys=True,
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

            stop_library = time.time()
            elapsed_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(stop_library - start_library),
            )

            matched_cu = len(successful_block_results)
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
                f"block_best_any={block_best_score * 100.0:6.2f}% | "
                f"block_best_matched={successful_block_best_score * 100.0:6.2f}% | "
                f"rodata_confirmed_cu={rodata_confirmed_count} | "
                f"rodata_penalty_cu={rodata_penalty_count} | "
                f"matched_cu={matched_cu}/{total_cu} | "
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
        
    asm_model = PalmTree("Palm Tree")
    asm_model.load(
        args.asm_model,
        device=args.device,
        pooling=args.palmtree_pooling,
    )
    print(f"PalmTree device: {asm_model.device}")
    print(f"PalmTree pooling: {asm_model.pooling}")

    lib: dict[str, list[CodeUnit]] = {}
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

            for obj_file in extracted_objects:
                try:
                    b = parse_r2_file(
                        obj_file.as_posix(),
                        asm_model=asm_model,
                        unit_type=CodeUnit.TYPE_CU,
                        asm_normalization=args.asm_normalization,
                    )
                    log_parse_debug(b)
                    if b.get_num_functions() > 0:
                        comp_units.append(b)
                except Exception as e:
                    print(f"[WARN] Failed to parse {obj_file}: {e}")

        lib[library_file.as_posix()] = comp_units
        print(f"[DEBUG] {library_file.name}: loaded {len(comp_units)} object files")

    log_line(f"Found {len(library_files)} libraries to process", args.output)
    log_line(f"Target binary: {binary_path}", args.output)
    log_line(f"Assembly normalization: {args.asm_normalization}", args.output)
    log_line(f"PalmTree pooling: {args.palmtree_pooling}", args.output)
    log_line(f"Library score aggregator: {args.library_score_aggregator}", args.output)
    log_line(f"Library minimum score: {args.library_min_score}", args.output)
    log_line(f"Rodata penalty threshold: {args.rodata_penalty_threshold}", args.output)
    log_line(f"Rodata confirm threshold: {args.rodata_confirm_threshold}", args.output)
    log_line(f"Rodata bonus weight: {args.rodata_bonus_weight}", args.output)
    log_line(f"Start processing: {time.strftime('%H:%M:%S', time.gmtime())}\n", args.output)

    binary = parse_r2_file(
        binary_path.as_posix(),
        asm_model=asm_model,
        unit_type=CodeUnit.TYPE_ELF,
        asm_normalization=args.asm_normalization,
    )
    log_parse_debug(binary)

    process_bin(binary, lib, args)

    elapsed_main = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_main))
    log_line(f"\nDone processing in {elapsed_main}", args.output)
