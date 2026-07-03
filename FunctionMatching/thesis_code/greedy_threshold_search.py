#!/usr/bin/env python3
"""Hybrid parameter search for the randomized static-ELF dataset.

Structural block-matching parameters are sampled through expensive pipeline
runs. Decision thresholds are then optimized offline with restarted greedy
coordinate search over the emitted BLOCK_CU records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "GroundTruth"
DEFAULT_LIB_BUILD_ROOT = REPO_ROOT / "Dataset/builds/lib_builds"
DEFAULT_GCC_LIB_ROOT = DEFAULT_LIB_BUILD_ROOT / "gcc-16.1.1"
DEFAULT_NEGATIVE_LIB_ROOT = (
    REPO_ROOT / "Exploration/libseeker_repo/build_lib/all_libs"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "greedy_threshold_results"
OPTIMIZATIONS = ("O0", "O1", "O2", "O3", "Os")
DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES = 600_000
FEATURE_SCHEMA_VERSION = 2

PREFERRED_NEGATIVE_ARCHIVES = (
    "libpcre2-posix.a",
    "libpanelw.a",
    "libmenuw.a",
    "libformw.a",
    "libbz2.a",
    "libcap.a",
    "libattr.a",
    "libuuid.a",
    "libexpat.a",
    "liblzma.a",
    "libssl.a",
    "libcrypto.a",
    "libmount.a",
    "libblkid.a",
    "libsodium.a",
)

DECISION_DEFAULTS = {
    "block_coverage_mean_threshold": 0.80,
    "block_assignment_threshold": 0.875,
    "block_min_coverage_ratio": 0.50,
    "block_min_edge_locality_ratio": 0.70,
    "block_min_function_concentration": 0.45,
    "block_min_function_spread": 0.30,
    "rodata_filter_enabled": 1,
    "rodata_min_bytes": 512,
    "rodata_min_strings": 0,
    "rodata_min_ngrams": 32,
    "rodata_penalty_threshold": 0.20,
}

DECISION_GRIDS = {
    "block_coverage_mean_threshold": tuple(
        round(0.60 + (index * 0.025), 3) for index in range(13)
    ),
    "block_assignment_threshold": tuple(
        round(0.60 + (index * 0.025), 3) for index in range(13)
    ),
    "block_min_coverage_ratio": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "block_min_edge_locality_ratio": tuple(
        round(index * 0.10, 3) for index in range(10)
    ),
    "block_min_function_concentration": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "block_min_function_spread": tuple(
        round(0.25 + (index * 0.05), 3) for index in range(11)
    ),
    "rodata_filter_enabled": (0, 1),
    "rodata_min_bytes": (0, 32, 64, 128, 256, 512),
    "rodata_min_strings": (0, 1, 2, 3, 5),
    "rodata_min_ngrams": (0, 16, 32, 64, 128),
    "rodata_penalty_threshold": (0.00, 0.05, 0.10, 0.15, 0.20, 0.30),
}

STRUCTURAL_DEFAULTS = {
    "block_threshold": 0.70,
    "block_locality_window_multiplier": 5.0,
    "block_locality_window_padding": 2,
}

STRUCTURAL_GRIDS = {
    "block_threshold": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    "block_locality_window_multiplier": (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    "block_locality_window_padding": (0, 1, 2, 3, 4, 6),
}

SUMMARY_RE = re.compile(
    r"^(?:YES \[W\]|YES|NO)\s+\|\s+library=(?P<library>\S+)"
)
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect block-matching features from the randomized ELF builds, "
            "then tune the decision thresholds with greedy coordinate search."
        )
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--lib-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Compiler-specific library root. Repeatable. Defaults to "
            "gcc-16.1.1 plus any clang-* roots found under Dataset/builds/lib_builds."
        ),
    )
    parser.add_argument(
        "--negative-libs-dir",
        type=Path,
        default=DEFAULT_NEGATIVE_LIB_ROOT,
        help="libseeker_repo directory containing versioned negative archives.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260615,
        help="Controls the positive optimization mix and negative versions.",
    )
    parser.add_argument(
        "--program",
        action="append",
        choices=("grep", "less", "sed", "gawk", "nano"),
        help="Restrict the search to selected programs. Repeatable.",
    )
    parser.add_argument(
        "--compiler",
        action="append",
        choices=("gcc", "clang"),
        help="Restrict the search to selected ELF compilers. Repeatable.",
    )
    parser.add_argument(
        "--include-glibc",
        action="store_true",
        help="Include libc.a. It is universal in this dataset and very expensive.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--structural-trials",
        type=int,
        default=19,
        help="Number of expensive structural matching profiles to evaluate.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Greedy coordinate-search restarts for each structural profile.",
    )
    parser.add_argument(
        "--train-elf-count",
        type=int,
        default=6,
        help=(
            "Number of ELF files used to tune thresholds. Defaults to 6 to "
            "keep the expensive matching stage manageable."
        ),
    )
    parser.add_argument(
        "--positive-lib-count",
        type=int,
        default=6,
        help=(
            "Number of present libraries used by the search. The same number "
            "of absent libraries is selected from libseeker_repo."
        ),
    )
    parser.add_argument(
        "--max-positive-archive-bytes",
        type=int,
        default=DEFAULT_MAX_POSITIVE_ARCHIVE_BYTES,
        help=(
            "Prefer positive archives no larger than this many bytes, using "
            "the smallest available compiler/optimization build. 0 disables "
            "the size filter. Defaults to 600000 to avoid very slow libraries "
            "such as libiconv.a during threshold tuning."
        ),
    )
    parser.add_argument(
        "--objective-order",
        choices=("cu_first", "lib_first"),
        default="cu_first",
        help=(
            "Metric priority for greedy selection. cu_first optimizes CU "
            "F1/precision before library metrics; lib_first optimizes "
            "library F1/precision before CU metrics."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete feature collections already present in output-dir.",
    )
    parser.add_argument(
        "--reuse-feature-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Reuse compatible per-library features from a previous "
            "greedy_threshold_search output directory. Repeatable. This lets "
            "a new library selection reuse already collected profile/ELF/"
            "library records and compute only missing pairs."
        ),
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Do not run matching; use previously collected feature files.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect feature files without running the greedy search.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def archive_family(name: str) -> str:
    lowered = Path(name).name.lower()
    special = {
        "libc.a": "glibc",
        "libpcre2-8.a": "pcre2",
        "libiconv.a": "iconv",
    }
    return special.get(lowered, lowered.removesuffix(".a"))


def normalize_cu_name(name: str) -> str:
    normalized = Path(name).name
    if normalized.endswith(".o"):
        normalized = normalized[:-2]
    if normalized.endswith(".c"):
        normalized = normalized[:-2]
    prefixes = (
        "libpcre2_8_la-",
        "libpcre2_posix_la-",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


LINKER_MAP_MEMBER_RE = re.compile(
    r"(?P<archive>\S+?\.a)\((?P<member>[^()]+\.o)\)"
)


def included_cus_from_linker_map(linker_map: Path) -> dict[str, set[str]]:
    included: dict[str, set[str]] = {}
    if not linker_map.is_file():
        return included

    for match in LINKER_MAP_MEMBER_RE.finditer(
        linker_map.read_text(encoding="utf-8", errors="replace")
    ):
        family = archive_family(Path(match.group("archive")).name)
        included.setdefault(family, set()).add(
            normalize_cu_name(match.group("member"))
        )
    return included


def included_cus_from_ground_truth(data: dict[str, Any]) -> dict[str, set[str]]:
    included: dict[str, set[str]] = {}
    for archive in data.get("archives", []):
        family = archive_family(Path(str(archive["archive"])).name)
        compilation_units = archive.get("compilation_units")
        if compilation_units is None:
            continue
        included[family] = {
            normalize_cu_name(str(unit["compilation_unit"]))
            for unit in compilation_units
            if unit.get("included")
        }

    linker_map = data.get("linker_map")
    if linker_map:
        for family, names in included_cus_from_linker_map(Path(str(linker_map))).items():
            included.setdefault(family, set()).update(names)

    return included


def archive_is_present_in_ground_truth(archive: dict[str, Any]) -> bool:
    if "included_compilation_units" not in archive:
        return True
    return int(archive.get("included_compilation_units", 0) or 0) > 0


def resolve_lib_roots(args: argparse.Namespace) -> list[Path]:
    if args.lib_root:
        roots = [path.resolve() for path in args.lib_root]
    else:
        roots = []
        if DEFAULT_GCC_LIB_ROOT.is_dir():
            roots.append(DEFAULT_GCC_LIB_ROOT.resolve())
        roots.extend(
            path.resolve()
            for path in sorted(DEFAULT_LIB_BUILD_ROOT.glob("clang-*"))
            if path.is_dir()
        )

    if not roots:
        raise FileNotFoundError("No compiler-specific library roots found")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Library root not found: {root}")
    return roots


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    programs = set(args.program or ("grep", "less", "sed", "gawk", "nano"))
    compilers = set(args.compiler or ("gcc", "clang"))
    cases = []

    for program in sorted(programs):
        for path in sorted((args.ground_truth_dir / program).glob("*/ground_truth.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("elf_optimization"):
                continue
            compiler = str(data.get("compiler", "")).split("-", maxsplit=1)[0]
            if compiler not in compilers:
                continue
            binary = Path(str(data["binary"])).resolve()
            if not binary.is_file():
                raise FileNotFoundError(f"Ground-truth ELF not found: {binary}")
            present_archives = [
                archive
                for archive in data.get("archives", [])
                if archive_is_present_in_ground_truth(archive)
            ]
            expected = {
                archive_family(Path(str(archive["archive"])).name)
                for archive in present_archives
            }
            archive_sources: dict[str, set[str]] = {}
            for archive in present_archives:
                archive_path = Path(str(archive["archive"]))
                archive_sources.setdefault(archive_path.name, set()).add(
                    archive_path.parents[3].name
                )
            expected_cus = included_cus_from_ground_truth(data)
            cases.append(
                {
                    "variant": str(data["variant"]),
                    "program": program,
                    "compiler": compiler,
                    "elf_optimization": str(data["elf_optimization"]),
                    "binary": binary,
                    "expected": expected,
                    "expected_cus": expected_cus,
                    "archive_names": {
                        Path(str(archive["archive"])).name
                        for archive in present_archives
                    },
                    "archive_sources": archive_sources,
                }
            )

    if not cases:
        raise ValueError("No randomized ELF ground truths selected")
    return cases


def candidate_archive_names(
    cases: list[dict[str, Any]], include_glibc: bool
) -> list[str]:
    names = {
        archive_name
        for case in cases
        for archive_name in case["archive_names"]
    }
    if not include_glibc:
        names.discard("libc.a")
    return sorted(names)


def find_candidate_archive(
    lib_root: Path,
    source: str,
    optimization: str,
    archive_name: str,
) -> Path:
    matches = sorted(
        path.resolve()
        for path in (lib_root / source / optimization / "install/lib").glob(
            archive_name
        )
        if path.is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {archive_name} in {lib_root.name}/{source}/"
            f"{optimization}, found {len(matches)}"
        )
    return matches[0]


def preferred_sources_for_archive(
    cases: list[dict[str, Any]], archive_name: str
) -> set[str]:
    preferred: set[str] = set()
    for case in cases:
        preferred.update(case.get("archive_sources", {}).get(archive_name, set()))
    return preferred


def source_for_archive(
    lib_roots: list[Path],
    archive_name: str,
    preferred_sources: set[str] | None = None,
) -> str:
    matches = sorted(
        path.resolve()
        for lib_root in lib_roots
        for optimization in OPTIMIZATIONS
        for path in lib_root.glob(
            f"*/{optimization}/install/lib/{archive_name}"
        )
        if path.is_file()
    )
    sources = {path.parents[3].name for path in matches}
    if preferred_sources:
        preferred_matches = sources & preferred_sources
        if len(preferred_matches) == 1:
            return preferred_matches.pop()
    if len(sources) != 1:
        raise FileNotFoundError(
            f"Expected one source library for {archive_name}, found "
            f"{sorted(sources)}; preferred={sorted(preferred_sources or [])}"
        )
    return sources.pop()


def available_optimizations(
    lib_root: Path, source: str, archive_names: list[str]
) -> list[str]:
    return [
        optimization
        for optimization in OPTIMIZATIONS
        if all(
            (
                lib_root
                / source
                / optimization
                / "install/lib"
                / archive_name
            ).is_file()
            for archive_name in archive_names
        )
    ]


def available_optimizations_with_size_limit(
    args: argparse.Namespace,
    lib_root: Path,
    source: str,
    archive_names: list[str],
) -> list[str]:
    optimizations = available_optimizations(lib_root, source, archive_names)
    if args.max_positive_archive_bytes <= 0:
        return optimizations
    return [
        optimization
        for optimization in optimizations
        if all(
            archive_size(lib_root, source, optimization, archive_name)
            <= args.max_positive_archive_bytes
            for archive_name in archive_names
        )
    ]


def archive_size(lib_root: Path, source: str, optimization: str, archive_name: str) -> int:
    return (
        lib_root
        / source
        / optimization
        / "install/lib"
        / archive_name
    ).stat().st_size


def smallest_archive_size(
    lib_roots: list[Path], source: str, archive_name: str
) -> int:
    sizes = [
        archive_size(lib_root, source, optimization, archive_name)
        for lib_root in lib_roots
        for optimization in OPTIMIZATIONS
        if (
            lib_root
            / source
            / optimization
            / "install/lib"
            / archive_name
        ).is_file()
    ]
    if not sizes:
        raise FileNotFoundError(f"No build found for {archive_name}")
    return min(sizes)


def select_positive_archives(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, str]]:
    if args.positive_lib_count < 1:
        raise ValueError("--positive-lib-count must be at least 1")

    archive_names = candidate_archive_names(cases, args.include_glibc)
    preferred_sources_by_archive = {
        archive_name: preferred_sources_for_archive(cases, archive_name)
        for archive_name in archive_names
    }
    if args.max_positive_archive_bytes > 0:
        small_archive_names = []
        skipped_archive_names = []
        for archive_name in archive_names:
            source = source_for_archive(
                args.lib_roots,
                archive_name,
                preferred_sources_by_archive.get(archive_name),
            )
            size = smallest_archive_size(args.lib_roots, source, archive_name)
            if size <= args.max_positive_archive_bytes:
                small_archive_names.append(archive_name)
            else:
                skipped_archive_names.append((archive_name, size))

        if len(small_archive_names) >= args.positive_lib_count:
            archive_names = small_archive_names
        else:
            print(
                f"[WARN] Only {len(small_archive_names)} positive libraries "
                f"fit --max-positive-archive-bytes="
                f"{args.max_positive_archive_bytes}; using them anyway"
            )
            archive_names = small_archive_names

        for archive_name, size in skipped_archive_names:
            print(
                f"[positive-skip] {archive_name} size={size} exceeds "
                f"--max-positive-archive-bytes={args.max_positive_archive_bytes}"
            )

    archives_by_source: dict[str, list[str]] = {}
    for archive_name in archive_names:
        source = source_for_archive(
            args.lib_roots,
            archive_name,
            preferred_sources_by_archive.get(archive_name),
        )
        archives_by_source.setdefault(source, []).append(archive_name)

    source_candidates = []
    for source, source_archive_names in sorted(archives_by_source.items()):
        archive_name = min(
            source_archive_names,
            key=lambda name: (
                smallest_archive_size(args.lib_roots, source, name),
                name,
            ),
        )
        source_candidates.append(
            (
                smallest_archive_size(args.lib_roots, source, archive_name),
                source,
                archive_name,
            )
        )
    source_candidates.sort()

    selected_names = []
    source_order = [source for _, source, _ in source_candidates]
    for _, _, archive_name in source_candidates:
        selected_names.append(archive_name)
        if len(selected_names) == args.positive_lib_count:
            break

    if len(selected_names) < args.positive_lib_count:
        already_selected = set(selected_names)
        remaining = [
            archive_name
            for source in source_order
            for archive_name in sorted(archives_by_source[source])
            if archive_name not in already_selected
        ]
        remaining.sort(
            key=lambda name: (
                smallest_archive_size(
                    args.lib_roots,
                    source_for_archive(
                        args.lib_roots,
                        name,
                        preferred_sources_by_archive.get(name),
                    ),
                    name,
                ),
                name,
            )
        )
        selected_names.extend(
            remaining[: args.positive_lib_count - len(selected_names)]
        )

    if len(selected_names) < args.positive_lib_count:
        print(
            f"[WARN] Requested {args.positive_lib_count} positive libraries, "
            f"but only {len(selected_names)} are available for the selected ELF set"
        )

    archives_by_source = {}
    for archive_name in selected_names:
        source = source_for_archive(
            args.lib_roots,
            archive_name,
            preferred_sources_by_archive.get(archive_name),
        )
        archives_by_source.setdefault(source, []).append(archive_name)

    sources = sorted(archives_by_source)
    optimization_cycle = list(OPTIMIZATIONS)
    rng.shuffle(optimization_cycle)
    root_cycle = list(args.lib_roots)
    rng.shuffle(root_cycle)
    selected = []

    for index, source in enumerate(sources):
        source_archives = sorted(archives_by_source[source])
        root_candidates = [
            lib_root
            for lib_root in args.lib_roots
            if available_optimizations_with_size_limit(
                args, lib_root, source, source_archives
            )
        ]
        if not root_candidates:
            raise FileNotFoundError(
                f"No compiler root has all archives for {source_archives} "
                f"within --max-positive-archive-bytes="
                f"{args.max_positive_archive_bytes}"
            )
        preferred_root = root_cycle[index % len(root_cycle)]
        lib_root = (
            preferred_root
            if preferred_root in root_candidates
            else min(root_candidates, key=lambda root: root.name)
        )
        available = available_optimizations_with_size_limit(
            args, lib_root, source, source_archives
        )
        if not available:
            raise FileNotFoundError(
                f"No common optimization available for {source_archives}"
            )
        preferred = optimization_cycle[index % len(optimization_cycle)]
        optimization = preferred if preferred in available else rng.choice(available)

        for archive_name in source_archives:
            archive = find_candidate_archive(
                lib_root, source, optimization, archive_name
            )
            selected.append(
                {
                    "name": archive_name,
                    "path": str(archive),
                    "kind": "positive",
                    "source": source,
                    "variant": f"{lib_root.name}/{optimization}",
                    "compiler_root": lib_root.name,
                    "optimization": optimization,
                    "size": archive.stat().st_size,
                }
            )
    return sorted(selected, key=lambda item: item["name"])


def versioned_archive_name(path: Path) -> str | None:
    match = re.match(r"^(?P<archive>.+\.a)\.(?P<version>.+)$", path.name)
    return match.group("archive") if match else None


def negative_archive_pool(directory: Path) -> dict[str, list[Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Negative libraries directory not found: {directory}")

    pool: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        archive_name = versioned_archive_name(path)
        if archive_name is None or path.stat().st_size < 4096:
            continue
        pool.setdefault(archive_name, []).append(path.resolve())
    return pool


def smallest_negative_size(paths: list[Path]) -> int:
    return min(path.stat().st_size for path in paths)


def select_negative_archives(
    args: argparse.Namespace,
    positive_archives: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    pool = negative_archive_pool(args.negative_libs_dir)
    positive_names = {item["name"] for item in positive_archives}
    count = len(positive_archives)
    eligible_names = [
        name
        for name in PREFERRED_NEGATIVE_ARCHIVES
        if name in pool and name not in positive_names
    ]
    eligible_names.sort(key=lambda name: (smallest_negative_size(pool[name]), name))

    if len(eligible_names) < count:
        fallback = sorted(
            name
            for name in pool
            if name not in positive_names
            and name not in eligible_names
            and not any(
                token in name
                for token in ("_pic.a", "_g.a", "nonshared", "BrokenLocale")
            )
        )
        fallback.sort(key=lambda name: (smallest_negative_size(pool[name]), name))
        eligible_names.extend(fallback[: count - len(eligible_names)])

    if len(eligible_names) < count:
        raise ValueError(
            f"Need {count} negative libraries, found {len(eligible_names)}"
        )

    selected = []
    for archive_name in eligible_names[:count]:
        candidates = sorted(pool[archive_name], key=lambda path: (path.stat().st_size, path.name))
        archive = candidates[0]
        version = archive.name.removeprefix(f"{archive_name}.")
        selected.append(
            {
                "name": archive_name,
                "path": str(archive),
                "kind": "negative",
                "source": "libseeker_repo",
                "variant": version,
                "compiler_root": "libseeker_repo",
                "optimization": "",
                "size": archive.stat().st_size,
            }
        )
    return sorted(selected, key=lambda item: item["name"])


def selection_signature(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def archive_key_matches(library_key: str, archive_name: str) -> bool:
    return library_key == archive_name or library_key.startswith(f"{archive_name}.")


def canonical_library_key(library_key: str, archive_names: list[str]) -> str:
    for archive_name in archive_names:
        if archive_key_matches(library_key, archive_name):
            return archive_name
    return library_key


def find_archive_records(
    library_records: dict[str, list[dict[str, Any]]],
    archive_name: str,
) -> tuple[bool, list[dict[str, Any]]]:
    if archive_name in library_records:
        return True, library_records[archive_name]

    merged_records = []
    found = False
    for library_key, records in library_records.items():
        if archive_key_matches(library_key, archive_name):
            found = True
            merged_records.extend(records)
    return found, merged_records


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def prepare_inputs(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    selected_archives: list[dict[str, str]],
) -> tuple[Path, Path]:
    elf_dir = args.output_dir / "inputs/elfs"
    lib_dir = args.output_dir / "inputs/libs"
    elf_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    for stale in (*elf_dir.iterdir(), *lib_dir.iterdir()):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)

    for case in cases:
        replace_symlink(elf_dir / case["variant"], case["binary"])
    for archive in selected_archives:
        replace_symlink(
            lib_dir / archive["name"],
            Path(archive["path"]),
        )
    return elf_dir, lib_dir


def prepare_subset_inputs(
    args: argparse.Namespace,
    profile_id: str,
    archive_name: str,
    cases: list[dict[str, Any]],
    archive: dict[str, str],
) -> tuple[Path, Path]:
    subset_root = (
        args.output_dir
        / "inputs"
        / "missing"
        / profile_id
        / safe_path_part(archive_name)
    )
    elf_dir = subset_root / "elfs"
    lib_dir = subset_root / "libs"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    elf_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)

    for case in cases:
        replace_symlink(elf_dir / case["variant"], case["binary"])
    replace_symlink(lib_dir / archive["name"], Path(archive["path"]))
    return elf_dir, lib_dir


def balanced_sample(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("ELF counts cannot be negative")
    if count == 0:
        return []

    buckets = {
        key: sorted(value, key=lambda case: case["variant"])
        for key, value in groups.items()
    }
    rng = random.Random(seed)
    for key, bucket in buckets.items():
        local_rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        local_rng.shuffle(bucket)

    selected = []
    group_order = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        rng.shuffle(group_order)
        progressed = False
        for key in group_order:
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop())
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break

    if len(selected) < count:
        raise ValueError(f"Requested {count} ELF files, found only {len(selected)}")
    return sorted(selected, key=lambda case: case["variant"])


def select_train_cases(
    cases: list[dict[str, Any]],
    train_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((case["program"], case["compiler"]), []).append(case)

    return balanced_sample(groups, train_count, seed)


def structural_profiles(count: int, seed: int) -> list[dict[str, float | int | str]]:
    default_tuple = tuple(STRUCTURAL_DEFAULTS.values())
    coverage_combinations = [default_tuple]
    for parameter_index, parameter in enumerate(STRUCTURAL_DEFAULTS):
        default_value = STRUCTURAL_DEFAULTS[parameter]
        for candidate in STRUCTURAL_GRIDS[parameter]:
            if candidate == default_value:
                continue
            values = list(default_tuple)
            values[parameter_index] = candidate
            coverage_combinations.append(tuple(values))

    minimum_count = len(coverage_combinations)
    if count < minimum_count:
        raise ValueError(
            f"--structural-trials must be at least {minimum_count} "
            "to cover every structural parameter value"
        )

    interaction_combinations = [
        combination
        for combination in itertools.product(
            *(STRUCTURAL_GRIDS[name] for name in STRUCTURAL_DEFAULTS)
        )
        if combination not in set(coverage_combinations)
    ]
    rng = random.Random(seed)
    rng.shuffle(interaction_combinations)
    chosen = [
        *coverage_combinations,
        *interaction_combinations[: count - minimum_count],
    ]

    profiles = []
    for index, values in enumerate(chosen):
        profile = dict(zip(STRUCTURAL_DEFAULTS, values))
        profile["profile_id"] = f"profile_{index:03d}"
        profiles.append(profile)
    return profiles


def feature_path(output_dir: Path, profile_id: str) -> Path:
    return output_dir / "features" / f"{profile_id}.json.gz"


def collection_command(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    elf_dir: Path,
    lib_dir: Path,
    batch_dir: Path,
    resume: bool,
) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "run_libseeker_batch.py"),
        "--pipeline",
        "current",
        "--dataset-dir",
        str(elf_dir),
        "--libs-dir",
        str(lib_dir),
        "--output-dir",
        str(batch_dir),
        "--all-elfs",
        "--all-libs",
        "--python",
        args.python,
        "--block-threshold",
        str(profile["block_threshold"]),
        "--block-locality-window-multiplier",
        str(profile["block_locality_window_multiplier"]),
        "--block-locality-window-padding",
        str(profile["block_locality_window_padding"]),
        "--block-coverage-mean-threshold",
        "0",
        "--block-assignment-threshold",
        "0",
        "--block-min-coverage-ratio",
        "0",
        "--block-min-edge-locality-ratio",
        "0",
        "--block-min-function-concentration",
        "0",
        "--block-min-function-spread",
        "0",
        "--disable-rodata-filter",
    ]
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])
    if resume:
        command.append("--resume")
    return command


def fraction_right(value: str) -> int:
    try:
        return int(value.split("/", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def parse_report(path: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    binary_path = ""
    pending: list[dict[str, Any]] = []
    libraries: dict[str, list[dict[str, Any]]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Target binary: "):
            binary_path = str(Path(line.removeprefix("Target binary: ")).resolve())
            continue
        if line.lstrip().startswith("BLOCK_CU["):
            fields = dict(FIELD_RE.findall(line))
            pending.append(
                {
                    "name": fields.get("name", ""),
                    "coverage_mean": float(fields["coverage_mean"]),
                    "coverage_ratio": float(fields["coverage_ratio"]),
                    "assignment_mean": float(fields["assignment_mean"]),
                    "edge_locality_ratio": float(fields["edge_locality_ratio"]),
                    "function_concentration": float(fields["function_concentration"]),
                    "function_spread": float(fields["function_spread"]),
                    "rodata": float(fields["rodata"]),
                    "rodata_strings": fraction_right(fields["rodata_strings"]),
                    "rodata_ngrams": fraction_right(fields["rodata_ngrams"]),
                    "rodata_bytes": int(fields["rodata_bytes"]),
                    "matched_functions": int(fields["matched_functions"]),
                }
            )
            continue
        summary = SUMMARY_RE.match(line)
        if summary:
            libraries[summary.group("library")] = pending
            pending = []

    if not binary_path:
        raise ValueError(f"Target binary missing from report: {path}")
    return binary_path, libraries


def parse_feature_collection(
    profile: dict[str, float | int | str],
    reports_dir: Path,
    expected_report_count: int,
    input_signature: str,
    archive_names: list[str],
) -> dict[str, Any]:
    reports = sorted(reports_dir.glob("*.report.txt"))
    if len(reports) != expected_report_count:
        raise RuntimeError(
            f"Expected {expected_report_count} reports in {reports_dir}, "
            f"found {len(reports)}"
        )
    cases = {}
    for report in reports:
        binary_path, libraries = parse_report(report)
        normalized_libraries: dict[str, list[dict[str, Any]]] = {}
        for library_key, records in libraries.items():
            canonical_key = canonical_library_key(library_key, archive_names)
            normalized_libraries.setdefault(canonical_key, []).extend(records)
        cases[binary_path] = normalized_libraries
    return {
        "profile": profile,
        "input_signature": input_signature,
        "cases": cases,
    }


def write_gzip_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def structural_profile_matches(
    expected: dict[str, float | int | str],
    collection: dict[str, Any],
) -> bool:
    actual = collection.get("profile", {})
    return all(actual.get(key) == expected.get(key) for key in STRUCTURAL_DEFAULTS)


def normalize_existing_collection(
    collection: dict[str, Any],
    profile: dict[str, float | int | str],
    input_signature: str,
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, Any]:
    normalized_cases = {}
    existing_cases = collection.get("cases", {})
    for case in cases:
        binary_path = str(case["binary"])
        existing_libraries = existing_cases.get(binary_path, {})
        normalized_libraries = {}
        for archive_name in archive_names:
            found, records = find_archive_records(existing_libraries, archive_name)
            if found:
                normalized_libraries[archive_name] = records
        normalized_cases[binary_path] = normalized_libraries
    return {
        "profile": profile,
        "input_signature": input_signature,
        "cases": normalized_cases,
    }


def count_cached_pairs(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> int:
    count = 0
    for case in cases:
        library_records = collection["cases"].get(str(case["binary"]), {})
        for archive_name in archive_names:
            if archive_name in library_records:
                count += 1
    return count


def reusable_feature_paths(
    args: argparse.Namespace,
    profile_id: str,
    destination: Path,
) -> list[Path]:
    paths = []
    seen = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        paths.append(resolved)

    add(destination)
    for root in args.reuse_feature_dir:
        root = root.resolve()
        if root.is_file():
            add(root)
        elif (root / "features").is_dir():
            add(root / "features" / f"{profile_id}.json.gz")
        elif root.is_dir():
            add(root / f"{profile_id}.json.gz")
    return paths


def find_partial_reuse_collection(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    destination: Path,
    input_signature: str,
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, Any] | None:
    best_collection = None
    best_count = 0
    profile_id = str(profile["profile_id"])

    for path in reusable_feature_paths(args, profile_id, destination):
        try:
            existing = read_gzip_json(path)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as error:
            print(f"[features] cannot reuse {path}: {error}")
            continue
        if not structural_profile_matches(profile, existing):
            continue

        normalized = normalize_existing_collection(
            existing, profile, input_signature, cases, archive_names
        )
        cached_count = count_cached_pairs(normalized, cases, archive_names)
        if cached_count > best_count:
            best_collection = normalized
            best_count = cached_count
            print(
                f"[features] partial reuse candidate {profile_id}: "
                f"{cached_count}/{len(cases) * len(archive_names)} pairs from {path}"
            )

    return best_collection


def missing_cases_by_archive(
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    missing: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        binary_path = str(case["binary"])
        library_records = collection["cases"].setdefault(binary_path, {})
        for archive_name in archive_names:
            if archive_name not in library_records:
                missing.setdefault(archive_name, []).append(case)
    return missing


def merge_collection(
    destination: dict[str, Any],
    addition: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_name: str,
) -> None:
    for case in cases:
        binary_path = str(case["binary"])
        added_libraries = addition["cases"].get(binary_path, {})
        found, records = find_archive_records(added_libraries, archive_name)
        if not found:
            raise KeyError(
                f"Missing collected records for {archive_name} in {binary_path}"
            )
        destination["cases"].setdefault(binary_path, {})[archive_name] = records


def collect_missing_archive_features(
    args: argparse.Namespace,
    profile: dict[str, float | int | str],
    collection: dict[str, Any],
    archive_name: str,
    missing_cases: list[dict[str, Any]],
    selected_archives: dict[str, dict[str, str]],
    input_signature: str,
) -> None:
    profile_id = str(profile["profile_id"])
    archive = selected_archives[archive_name]
    elf_dir, lib_dir = prepare_subset_inputs(
        args, profile_id, archive_name, missing_cases, archive
    )
    batch_dir = (
        args.output_dir
        / "collections_missing"
        / profile_id
        / safe_path_part(archive_name)
    )
    missing_signature = selection_signature(
        [
            {
                "input": input_signature,
                "profile": profile,
                "archive": archive,
                "cases": [case["variant"] for case in missing_cases],
            }
        ]
    )
    batch_signature = batch_dir / "input_signature.txt"
    resume_batch = (
        args.resume
        and batch_signature.is_file()
        and batch_signature.read_text(encoding="utf-8").strip()
        == missing_signature
    )
    if batch_dir.exists() and not resume_batch:
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_signature.write_text(missing_signature + "\n", encoding="utf-8")

    command = collection_command(
        args, profile, elf_dir, lib_dir, batch_dir, resume=resume_batch
    )
    print(
        f"[features] collect missing {profile_id} {archive_name}: "
        f"{len(missing_cases)} ELF(s)"
    )
    print("+", " ".join(command), flush=True)
    if args.dry_run:
        return

    result = subprocess.run(command, cwd=SCRIPT_DIR)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)

    addition = parse_feature_collection(
        profile,
        batch_dir / "reports/current",
        len(missing_cases),
        missing_signature,
        [archive_name],
    )
    merge_collection(collection, addition, missing_cases, archive_name)


def collect_features(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    elf_dir: Path,
    lib_dir: Path,
    input_signature: str,
    selected_archives: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    collections = {}
    archive_names = [archive["name"] for archive in selected_archives]
    selected_archives_by_name = {archive["name"]: archive for archive in selected_archives}
    profiles = structural_profiles(args.structural_trials, args.seed)
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        profile_signature = selection_signature(
            [{"input": input_signature, **profile}]
        )
        destination = feature_path(args.output_dir, profile_id)
        if (args.resume or args.search_only) and destination.is_file():
            existing = read_gzip_json(destination)
            if existing.get("input_signature") == profile_signature:
                print(f"[features] reuse {profile_id}")
                collections[profile_id] = normalize_existing_collection(
                    existing, profile, profile_signature, cases, archive_names
                )
                continue
            if args.search_only:
                partial = find_partial_reuse_collection(
                    args,
                    profile,
                    destination,
                    profile_signature,
                    cases,
                    archive_names,
                )
                if partial is None:
                    raise ValueError(
                        f"Feature selection mismatch in {destination}; recollect it"
                    )
                missing = missing_cases_by_archive(partial, cases, archive_names)
                if missing:
                    raise ValueError(
                        f"Feature selection mismatch in {destination}; "
                        f"missing {sorted(missing)}"
                    )
                collections[profile_id] = partial
                continue
            print(f"[features] selection changed, try partial reuse {profile_id}")
        if args.search_only:
            raise FileNotFoundError(f"Missing feature file: {destination}")

        partial = find_partial_reuse_collection(
            args, profile, destination, profile_signature, cases, archive_names
        )
        if partial is not None:
            missing = missing_cases_by_archive(partial, cases, archive_names)
            if missing:
                print(
                    f"[features] partial reuse {profile_id}: "
                    f"{count_cached_pairs(partial, cases, archive_names)}/"
                    f"{len(cases) * len(archive_names)} pairs cached"
                )
                for archive_name, missing_cases in sorted(missing.items()):
                    collect_missing_archive_features(
                        args,
                        profile,
                        partial,
                        archive_name,
                        missing_cases,
                        selected_archives_by_name,
                        profile_signature,
                    )
            else:
                print(f"[features] partial reuse complete {profile_id}")
            if not args.dry_run:
                write_gzip_json(destination, partial)
                collections[profile_id] = partial
            continue

        batch_dir = (
            args.output_dir / "collections" / profile_id
        )
        batch_signature = batch_dir / "input_signature.txt"
        resume_batch = (
            args.resume
            and batch_signature.is_file()
            and batch_signature.read_text(encoding="utf-8").strip()
            == profile_signature
        )
        if batch_dir.exists() and not resume_batch:
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_signature.write_text(profile_signature + "\n", encoding="utf-8")
        command = collection_command(
            args, profile, elf_dir, lib_dir, batch_dir, resume=resume_batch
        )
        print("+", " ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=SCRIPT_DIR)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

        collection = parse_feature_collection(
            profile,
            batch_dir / "reports/current",
            len(cases),
            profile_signature,
            archive_names,
        )
        write_gzip_json(destination, collection)
        collections[profile_id] = collection
    return collections


def rodata_is_penalty(record: dict[str, Any], params: dict[str, Any]) -> bool:
    informative = (
        record["rodata_bytes"] >= params["rodata_min_bytes"]
        or record["rodata_strings"] >= params["rodata_min_strings"]
        or record["rodata_ngrams"] >= params["rodata_min_ngrams"]
    )
    return (
        bool(params["rodata_filter_enabled"])
        and informative
        and record["rodata"] <= params["rodata_penalty_threshold"]
    )


def record_passes(record: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        record["coverage_mean"] >= params["block_coverage_mean_threshold"]
        and record["coverage_ratio"] >= params["block_min_coverage_ratio"]
        and record["assignment_mean"] >= params["block_assignment_threshold"]
        and record["edge_locality_ratio"]
        >= params["block_min_edge_locality_ratio"]
        and record["function_concentration"]
        >= params["block_min_function_concentration"]
        and record["function_spread"] >= params["block_min_function_spread"]
        and not rodata_is_penalty(record, params)
    )


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    total = tp + tn + fp + fn

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "evaluated": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, total),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def calculate_cu_metrics(tp: int, fp: int, fn: int, evaluated_pairs: int) -> dict[str, float | int]:
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "cu_evaluated_pairs": evaluated_pairs,
        "cu_tp": tp,
        "cu_fp": fp,
        "cu_fn": fn,
        "cu_precision": ratio(tp, tp + fp),
        "cu_recall": ratio(tp, tp + fn),
        "cu_f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def cu_counts_for_prediction(
    archive_name: str,
    case: dict[str, Any],
    records: list[dict[str, Any]],
    successful: list[dict[str, Any]],
) -> tuple[dict[str, int | bool], bool]:
    family = archive_family(archive_name)
    expected = family in case["expected"]
    expected_cus_by_family = case.get("expected_cus", {})
    expected_cus = set(expected_cus_by_family.get(family, set()))

    has_record_names = all("name" in record for record in records)
    if not has_record_names:
        return {
            "cu_ground_truth": False,
            "expected_cu": 0,
            "cu_tp": 0,
            "cu_fp": 0,
            "cu_fn": 0,
        }, False

    predicted_cus = {
        normalize_cu_name(str(record["name"]))
        for record in successful
        if record.get("name")
    }

    if expected:
        if family not in expected_cus_by_family:
            return {
                "cu_ground_truth": False,
                "expected_cu": 0,
                "cu_tp": 0,
                "cu_fp": 0,
                "cu_fn": 0,
            }, False
        tp = len(predicted_cus & expected_cus)
        fp = len(predicted_cus - expected_cus)
        fn = len(expected_cus - predicted_cus)
    else:
        tp = 0
        fp = len(predicted_cus)
        fn = 0

    return {
        "cu_ground_truth": True,
        "expected_cu": len(expected_cus),
        "cu_tp": tp,
        "cu_fp": fp,
        "cu_fn": fn,
    }, True


def evaluate(
    params: dict[str, Any],
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    collected_cases = collection["cases"]
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    cu_counts = {"tp": 0, "fp": 0, "fn": 0}
    cu_evaluated_pairs = 0
    predictions = []

    for case in cases:
        binary_path = str(case["binary"])
        library_records = collected_cases.get(binary_path)
        if library_records is None:
            raise KeyError(f"No collected features for {binary_path}")

        for archive_name in archive_names:
            _, records = find_archive_records(library_records, archive_name)
            successful = [
                record for record in records if record_passes(record, params)
            ]
            predicted = bool(successful)
            expected = archive_family(archive_name) in case["expected"]
            cu_prediction, cu_evaluated = cu_counts_for_prediction(
                archive_name,
                case,
                records,
                successful,
            )
            if cu_evaluated:
                cu_evaluated_pairs += 1
                cu_counts["tp"] += int(cu_prediction["cu_tp"])
                cu_counts["fp"] += int(cu_prediction["cu_fp"])
                cu_counts["fn"] += int(cu_prediction["cu_fn"])
            classification = (
                "TP" if expected and predicted
                else "FN" if expected
                else "FP" if predicted
                else "TN"
            )
            counts[classification.lower()] += 1
            predictions.append(
                {
                    "variant": case["variant"],
                    "program": case["program"],
                    "compiler": case["compiler"],
                    "elf_optimization": case["elf_optimization"],
                    "library": archive_name,
                    "expected": expected,
                    "predicted": predicted,
                    "classification": classification,
                    "matched_cu": len(successful),
                    "total_cu": len(records),
                    **cu_prediction,
                }
            )

    library_metrics = calculate_metrics(**counts)
    metrics = {
        **library_metrics,
        "lib_accuracy": library_metrics["accuracy"],
        "lib_precision": library_metrics["precision"],
        "lib_recall": library_metrics["recall"],
        "lib_specificity": library_metrics["specificity"],
        "lib_f1": library_metrics["f1"],
        **calculate_cu_metrics(
            cu_counts["tp"],
            cu_counts["fp"],
            cu_counts["fn"],
            cu_evaluated_pairs,
        ),
    }
    return metrics, predictions


def objective(
    metrics: dict[str, float | int],
    objective_order: str,
) -> tuple[float, ...]:
    cu_metrics = (
        float(metrics["cu_f1"]),
        float(metrics["cu_precision"]),
        float(metrics["cu_recall"]),
    )
    lib_metrics = (
        float(metrics["lib_f1"]),
        float(metrics["lib_precision"]),
        float(metrics["lib_recall"]),
    )
    if objective_order == "lib_first":
        return (*lib_metrics, *cu_metrics)
    return (*cu_metrics, *lib_metrics)


def random_decision_params(rng: random.Random) -> dict[str, Any]:
    return {
        parameter: rng.choice(candidates)
        for parameter, candidates in DECISION_GRIDS.items()
    }


def greedy_profile_search(
    args: argparse.Namespace,
    profile_id: str,
    collection: dict[str, Any],
    train_cases: list[dict[str, Any]],
    archive_names: list[str],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, float | int], list[dict[str, Any]]]:
    best_params = dict(DECISION_DEFAULTS)
    best_metrics, _ = evaluate(
        best_params, collection, train_cases, archive_names
    )
    history = []

    for restart in range(args.restarts):
        params = (
            dict(DECISION_DEFAULTS)
            if restart == 0
            else random_decision_params(rng)
        )
        current_metrics, _ = evaluate(
            params, collection, train_cases, archive_names
        )

        for round_number in range(1, args.max_rounds + 1):
            changed = False
            for parameter, candidates in DECISION_GRIDS.items():
                old_value = params[parameter]
                selected_value = old_value
                selected_metrics = current_metrics
                selected_key = objective(current_metrics, args.objective_order)

                for candidate in candidates:
                    trial = dict(params)
                    trial[parameter] = candidate
                    metrics, _ = evaluate(
                        trial, collection, train_cases, archive_names
                    )
                    key = objective(metrics, args.objective_order)
                    history.append(
                        {
                            "profile_id": profile_id,
                            "restart": restart,
                            "round": round_number,
                            "parameter": parameter,
                            "candidate": candidate,
                            **metrics,
                        }
                    )
                    if key > selected_key:
                        selected_value = candidate
                        selected_metrics = metrics
                        selected_key = key

                if selected_value != old_value:
                    changed = True
                    params[parameter] = selected_value
                    current_metrics = selected_metrics
            if not changed:
                break

        if objective(current_metrics, args.objective_order) > objective(
            best_metrics,
            args.objective_order,
        ):
            best_params = params
            best_metrics = current_metrics

    return best_params, best_metrics, history


def search_all_profiles(
    args: argparse.Namespace,
    collections: dict[str, dict[str, Any]],
    train_cases: list[dict[str, Any]],
    archive_names: list[str],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, float | int],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rng = random.Random(args.seed + 1)
    best_profile_id = ""
    best_params: dict[str, Any] = {}
    best_train_metrics: dict[str, float | int] = {}
    all_history = []
    profile_rows = []

    for profile_id, collection in sorted(collections.items()):
        params, train_metrics, history = greedy_profile_search(
            args,
            profile_id,
            collection,
            train_cases,
            archive_names,
            rng,
        )
        all_history.extend(history)
        profile = collection["profile"]
        profile_rows.append(
            {
                **profile,
                **{f"best_{key}": value for key, value in params.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
        )
        print(
            f"[{profile_id}] "
            f"train CU_F1={float(train_metrics['cu_f1']):.4f} "
            f"CU_precision={float(train_metrics['cu_precision']):.4f} "
            f"lib_F1={float(train_metrics['lib_f1']):.4f}"
        )

        if (
            not best_train_metrics
            or objective(train_metrics, args.objective_order)
            > objective(best_train_metrics, args.objective_order)
        ):
            best_profile_id = profile_id
            best_params = params
            best_train_metrics = train_metrics

    return (
        best_profile_id,
        best_params,
        best_train_metrics,
        all_history,
        profile_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_metrics(
    params: dict[str, Any],
    collection: dict[str, Any],
    cases: list[dict[str, Any]],
    archive_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    groups = [
        ("ALL", lambda case: True),
        ("gcc", lambda case: case["compiler"] == "gcc"),
        ("clang", lambda case: case["compiler"] == "clang"),
    ]
    groups.extend(
        (program, lambda case, selected=program: case["program"] == selected)
        for program in sorted({case["program"] for case in cases})
    )
    for group, predicate in groups:
        selected_cases = [case for case in cases if predicate(case)]
        metrics, _ = evaluate(params, collection, selected_cases, archive_names)
        rows.append({"group": group, **metrics})
    return rows


def main() -> int:
    args = parse_args()
    if args.search_only and args.collect_only:
        raise ValueError("--search-only and --collect-only are mutually exclusive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.lib_roots = resolve_lib_roots(args)

    cases = load_cases(args)
    train_cases = select_train_cases(
        cases,
        args.train_elf_count,
        args.seed,
    )
    if not train_cases:
        raise ValueError("At least one training ELF is required")
    search_cases = train_cases
    rng = random.Random(args.seed)
    positive_archives = select_positive_archives(args, search_cases, rng)
    negative_archives = select_negative_archives(args, positive_archives, rng)
    selected_archives = sorted(
        positive_archives + negative_archives,
        key=lambda item: item["name"],
    )
    archive_names = [item["name"] for item in selected_archives]
    input_signature = selection_signature(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "objective_order": args.objective_order,
            "archives": selected_archives,
        }
    )
    selection_report = {
        "seed": args.seed,
        "input_signature": input_signature,
        "positive_count": len(positive_archives),
        "negative_count": len(negative_archives),
        "objective_order": args.objective_order,
        "max_positive_archive_bytes": args.max_positive_archive_bytes,
        "lib_roots": [str(root) for root in args.lib_roots],
        "archives": selected_archives,
    }
    (args.output_dir / "input_selection.json").write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    elf_dir, lib_dir = prepare_inputs(args, search_cases, selected_archives)
    print(
        f"Selected {len(search_cases)} / {len(cases)} ELF files, "
        f"{len(positive_archives)} positive archives and "
        f"{len(negative_archives)} negative archives"
    )
    print(f"Search set: {len(train_cases)} ELF files")
    for archive in selected_archives:
        print(
            f"  [{archive['kind']}] {archive['name']} "
            f"({archive['source']} {archive['variant']}, "
            f"{int(archive['size'])} bytes)"
        )

    collections = collect_features(
        args,
        search_cases,
        elf_dir,
        lib_dir,
        input_signature,
        selected_archives,
    )
    if args.dry_run or args.collect_only:
        return 0

    (
        best_profile_id,
        params,
        train_metrics,
        history,
        profile_rows,
    ) = search_all_profiles(
        args,
        collections,
        train_cases,
        archive_names,
    )
    best_collection = collections[best_profile_id]
    metrics, predictions = evaluate(
        params, best_collection, search_cases, archive_names
    )
    metadata_by_name = {
        archive["name"]: archive for archive in selected_archives
    }
    for prediction in predictions:
        metadata = metadata_by_name[prediction["library"]]
        prediction["candidate_kind"] = metadata["kind"]
        prediction["candidate_source"] = metadata["source"]
        prediction["candidate_variant"] = metadata["variant"]

    structural = {
        key: value
        for key, value in best_collection["profile"].items()
        if key != "profile_id"
    }
    tuned_parameters = {**structural, **params}
    report = {
        "best_profile_id": best_profile_id,
        "parameters": tuned_parameters,
        "metrics": metrics,
        "available_elf_count": len(cases),
        "searched_elf_count": len(search_cases),
        "train_elf_count": len(train_cases),
        "seed": args.seed,
        "objective_order": args.objective_order,
        "input_signature": input_signature,
        "positive_archives": positive_archives,
        "negative_archives": negative_archives,
        "pipeline": "current block/.rodata matching only",
        "ground_truth_rule": "archive listed in randomized ground_truth.json",
        "not_tuned": {
            "rodata_confirm_threshold": (
                "diagnostic only; it does not change YES/NO presence"
            ),
            "asm_model": "model choice, not a matching threshold",
            "timeout": "execution control, not matching behavior",
        },
    }
    (args.output_dir / "best_thresholds.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "greedy_history.csv", history)
    write_csv(args.output_dir / "structural_profiles.csv", profile_rows)
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_csv(
        args.output_dir / "metrics_by_group.csv",
        grouped_metrics(params, best_collection, search_cases, archive_names),
    )

    print(f"\nBest structural profile: {best_profile_id}")
    print("Best parameters:")
    for name, value in tuned_parameters.items():
        print(f"  {name}={value}")
    print(
        f"Train CU_F1={float(train_metrics['cu_f1']):.4f} "
        f"CU_precision={float(train_metrics['cu_precision']):.4f} "
        f"lib_F1={float(train_metrics['lib_f1']):.4f}"
    )
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
