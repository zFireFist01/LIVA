#!/usr/bin/env python3
"""Replay function coverage for one compiler on a shared coordinate panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import gzip
import hashlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path
import re
import statistics
import sys
import time
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED = REPO_ROOT / "libseeker-unified"


def install_match_stub() -> None:
    stub = types.ModuleType("match")
    stub.LIBRARY_SCORE_AGGREGATORS = ("mean", "max", "top3_mean", "top3_noisy_or")

    def aggregate(scores, mode, *, score_floor=0.0):
        ranked = sorted((float(value) for value in scores), reverse=True)
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
                min(1.0, max(0.0, (value - float(score_floor)) / scale))
                for value in top
            ]
            return 1.0 - math.prod(1.0 - value for value in probabilities)
        raise ValueError(mode)

    stub.aggregate_library_score = aggregate
    sys.modules["match"] = stub


install_match_stub()
sys.path.insert(0, str(REPO_ROOT / "thesis_code"))
import optuna_threshold_search as replay  # noqa: E402


@contextlib.contextmanager
def filtered_feature_stream(path: Path, included: set[str]):
    wanted = {value.encode() for value in included}
    pattern = re.compile(rb'"library":"([^"]+)"')
    with gzip.open(path, "rb") as stream:
        def lines():
            for raw in stream:
                if b'"type":"source_call_targets"' in raw or any(
                    match.group(1) in wanted for match in pattern.finditer(raw)
                ):
                    yield raw.decode("utf-8", "replace")
        yield lines()


source = inspect.getsource(replay.parse_feature_jsonl)
source = source.replace(
    'opener = gzip.open if path.name.endswith(".gz") else open',
    'opener = lambda path, mode, encoding, errors: filtered_feature_stream(path, included_libraries)',
)
namespace = dict(replay.__dict__)
namespace["filtered_feature_stream"] = filtered_feature_stream
exec(source, namespace)
fast_parse = namespace["parse_feature_jsonl"]


STATUS_RE = re.compile(r"^(YES \[W\]|YES|NO)\s+\|\s+library=([^ |]+)")
BASE_RE = re.compile(r"\bbase_score=\s*([0-9.]+)%")
SCORE_RE = re.compile(r"\bscore=\s*([0-9.]+)%")
SCENARIOS = ("exact_build", "family")


def library_family(name: str) -> str:
    lowered = Path(name).name.lower()
    patterns = (
        (r"^libc\.a(?:\.|$)", "glibc"),
        (r"^libpcre2-8\.a(?:\.|$)", "pcre2"),
        (r"^libpcre2-posix\.a(?:\.|$)", "pcre2-posix"),
        (r"^libiconv\.a(?:\.|$)", "iconv"),
        (r"^libcharset\.a(?:\.|$)", "charset"),
        (r"^libgcc_eh\.a(?:\.|$)", "libgcc_eh"),
        (r"^libgcc\.a(?:\.|$)", "libgcc"),
    )
    for pattern, family in patterns:
        if re.match(pattern, lowered):
            return family
    match = re.match(r"^(lib[^.]+)", lowered)
    return match.group(1) if match else lowered


def truth_family(archive: dict) -> str:
    library = str(archive.get("library", "")).lower()
    source_name = str(archive.get("source", "")).lower()
    if library == "compiler-runtime":
        return library_family(source_name)
    if library == "glibc":
        return "glibc"
    if library == "pcre2":
        return "pcre2"
    if library in {"iconv", "libiconv"}:
        return "iconv"
    return library_family(str(archive.get("archive", library)))


def coordinate(report: Path) -> tuple[str, str, tuple[str, ...]]:
    with report.open(encoding="utf-8", errors="replace") as stream:
        stream.readline()
        target = stream.readline().removeprefix("Target binary: ").strip()
    parts = Path(target).parts[-4:]
    return parts[0], parts[2], parts


def tally(counts: Counter, actual: bool, predicted: bool) -> None:
    counts["TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"] += 1


def scenario_metrics(status: dict[str, bool], exact_labels: set[str]):
    result = {key: Counter() for key in SCENARIOS}
    for label, predicted in status.items():
        tally(result["exact_build"], label in exact_labels, predicted)
    expected_families = {LABELS[label]["family"] for label in exact_labels}
    predicted_families = {
        LABELS[label]["family"] for label, predicted in status.items() if predicted
    }
    for family in FAMILIES:
        tally(
            result["family"],
            family in expected_families,
            family in predicted_families,
        )
    return result


def worker(report_string: str):
    report = Path(report_string)
    feature = report.with_name(report.name.replace(".report.txt", ".features.jsonl.gz"))
    _program, _optimization, parts = coordinate(report)
    payload = json.loads((TRUTH_ROOT / parts[0] / parts[1] / parts[2] / "ground_truth.json").read_text())
    exact_labels: set[str] = set()
    incorporated = 0
    for archive in payload.get("archives", []):
        if int(archive.get("included_compilation_units", 0) or 0) <= 0:
            continue
        incorporated += 1
        archive_path = str(archive.get("archive", ""))
        if "Dataset/builds/libraries/" in archive_path:
            label = BY_PATH.get(archive_path.split("Dataset/builds/libraries/", 1)[1])
            if label in KNOWN:
                exact_labels.add(label)

    old: dict[str, bool] = {}
    selected: set[str] = set()
    near_negative = 0
    with report.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = STATUS_RE.match(line)
            if not match:
                continue
            label = match.group(2)
            predicted = match.group(1).startswith("YES")
            old[label] = predicted
            if predicted:
                selected.add(label)
            else:
                base = BASE_RE.search(line)
                score = SCORE_RE.search(line)
                if (base and float(base.group(1)) >= 96.86) or (
                    not base and score and float(score.group(1)) >= 96.86
                ):
                    selected.add(label)
                    near_negative += 1
    if len(old) != len(KNOWN):
        raise ValueError(f"{report}: incomplete report ({len(old)}/{len(KNOWN)})")

    with gzip.open(feature, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
    params = {**replay.DECISION_DEFAULTS, **header["matching_configuration"]}
    baseline_coverage = float(params["cu_min_function_coverage"])
    new = dict(old)
    disagreement = turned_on = turned_off = 0
    if baseline_coverage != COVERAGE:
        _binary, libraries, _pooling, calls = fast_parse(feature, selected)
        for label in selected:
            records = libraries.get(label, [])
            accepted, score = replay.library_match_evidence(records, params, calls)
            baseline = bool(accepted) and score >= float(params["library_min_score"])
            if baseline != old[label]:
                disagreement += 1
            params["cu_min_function_coverage"] = COVERAGE
            accepted, score = replay.library_match_evidence(records, params, calls)
            updated = bool(accepted) and score >= float(params["library_min_score"])
            params["cu_min_function_coverage"] = baseline_coverage
            new[label] = updated
            turned_on += updated and not old[label]
            turned_off += old[label] and not updated
    return (
        scenario_metrics(old, exact_labels),
        scenario_metrics(new, exact_labels),
        Counter(
            reports=1,
            selected=len(selected),
            near_negative=near_negative,
            baseline_disagreement=disagreement,
            turned_on=turned_on,
            turned_off=turned_off,
            incorporated_archives=incorporated,
            positive_exact=len(exact_labels),
            unsupported_positive=incorporated - len(exact_labels),
        ),
    )


def initialize(compiler: str, coverage: float):
    global REPORT_DIR, TRUTH_ROOT, LABELS, BY_PATH, KNOWN, BY_ARCHIVE, REFERENCE, COVERAGE, FAMILIES
    REPORT_DIR = UNIFIED / compiler / "reports/current"
    TRUTH_ROOT = UNIFIED / "ground_truth/libseeker/binaries"
    COVERAGE = coverage
    matrix_rows = [
        row for row in csv.DictReader((UNIFIED / "library_matrix.tsv").open(), delimiter="\t")
        if row["status"] == "selected" and row["path"]
    ]
    name_counts = Counter(row["archive"] for row in matrix_rows)
    LABELS, BY_PATH = {}, {}
    for row in matrix_rows:
        label = row["archive"]
        if name_counts[row["archive"]] > 1:
            identity = "Dataset/builds/libraries/" + row["path"]
            label += "." + hashlib.sha256(identity.encode()).hexdigest()[:16]
        LABELS[label] = {**row, "family": library_family(row["archive"])}
        BY_PATH[row["path"]] = label
    with next(REPORT_DIR.glob("*.report.txt")).open(encoding="utf-8", errors="replace") as stream:
        KNOWN = {match.group(2) for line in stream if (match := STATUS_RE.match(line))}
    BY_ARCHIVE = defaultdict(list)
    for label in KNOWN:
        BY_ARCHIVE[LABELS[label]["archive"]].append(label)
    FAMILIES = {LABELS[label]["family"] for label in KNOWN}

    def reference_rank(label: str):
        row = LABELS[label]
        return (
            row["role"] != "current",
            row["toolchain"] != "gcc-13-13.3.0",
            row["optimization"] != "O2",
            row["path"],
        )
    REFERENCE = {
        archive: min(labels, key=reference_rank)
        for archive, labels in BY_ARCHIVE.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="clang-18-18.1.3")
    parser.add_argument("--coverage", type=float, default=0.60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    initialize(args.compiler, args.coverage)
    base = UNIFIED / "gcc-13-13.3.0/reports/current"
    common = {coordinate(path)[:2] for path in base.glob("*.report.txt")}
    reports = sorted(
        path for path in REPORT_DIR.glob("*.report.txt") if coordinate(path)[:2] in common
    )
    if len(reports) != 646:
        raise ValueError(f"expected 646 common reports, found {len(reports)}")
    if args.limit > 0:
        reports = reports[:args.limit]
    old = {key: Counter() for key in SCENARIOS}
    new = {key: Counter() for key in SCENARIOS}
    stats = Counter()
    started = time.monotonic()
    with multiprocessing.Pool(args.workers) as pool:
        for index, (before, after, item_stats) in enumerate(
            pool.imap_unordered(worker, map(str, reports), chunksize=1), start=1
        ):
            for key in SCENARIOS:
                old[key].update(before[key])
                new[key].update(after[key])
            stats.update(item_stats)
            if index % 25 == 0:
                print(f"progress {index}/{len(reports)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    result = {"compiler": args.compiler, "coverage": args.coverage, "stats": dict(stats), "scenarios": {}}
    for key in SCENARIOS:
        result["scenarios"][key] = {}
        for name, counts in (("before", old[key]), ("after", new[key])):
            tp, fp, fn = (counts[label] for label in ("TP", "FP", "FN"))
            result["scenarios"][key][name] = {
                **dict(counts), "f1": 2 * tp / (2 * tp + fp + fn)
            }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
