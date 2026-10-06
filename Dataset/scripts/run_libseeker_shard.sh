#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$DATASET_DIR/.." && pwd)"

SHARD_INDEX=""
ARTIFACT_ROOT=""
LIBRARY_ROOT="$DATASET_DIR/builds/libraries"
LIBRARY_MATRIX="$DATASET_DIR/manifests/library_matrix.tsv"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
MATCHING_JOBS=auto
CACHE_WORKERS=8
SEED=20260731
DEVICE=auto
TOOLCHAIN_IMAGE="${LIVA_TOOLCHAIN_IMAGE:-thesis-binary-datasets:2026-08-balanced}"
DRY_RUN=0
CLEAN=0
SKIP_FETCH=0
SKIP_BUILD=0
SKIP_CACHE=0
SKIP_MATCHING=0
OFFLINE_ABLATION_FEATURES=0
CU_FUNCTION_COVERAGE=0.0

usage() {
    cat <<'EOF'
Usage: run_libseeker_shard.sh --shard-index N [options]

Build exactly one compiler shard of LibSeeker: 219 programs x 4
optimizations = 876 ELF records. It caches every ELF and the complete selected
library catalog, then searches every ELF against that complete catalog and
writes one portable compressed result log. Linking remains independent: every
ELF contains only archives needed to resolve its own symbols.

  --shard-index N       0..3 (required)
  --artifact-root DIR   output root (default: Dataset/shards/libseeker-shard-N)
  --library-root DIR    shared, read-only versioned library build root
  --library-matrix FILE selected versioned library matrix
  -j, --jobs N          parallel build jobs
  --cache-workers N     parallel ELF-cache shards (default: 8)
  --matching-jobs N     concurrent ELF matching processes or auto (default: auto)
  --seed N              common seed; use the same value on all four PCs
  --device DEVICE       cache embedding device (default: auto)
                         Missing host compiler pairs use $LIVA_TOOLCHAIN_IMAGE
                         (default: thesis-binary-datasets:2026-08-balanced)
  --clean               rebuild this shard instead of resuming it
  --skip-fetch          require ELF sources to be already present
  --skip-build          reuse an existing assembled and validated ELF dataset
  --skip-cache          skip cache prebuild/validation (matching fills misses)
  --skip-matching       do not run the ELF x complete-library-catalog search
  --offline-ablation-features
                        retain replay-complete B/H/S/X/R evidence per ELF
  --cu-function-coverage RATIO
                        require this fraction of reference-CU functions
  --dry-run             print commands without writing
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    (( DRY_RUN )) || "$@"
}

run_parallel_cache() {
    if (( DRY_RUN || CACHE_WORKERS == 1 )); then
        run "$@"
        return
    fi

    local worker status=0
    local -a pids=()
    for ((worker = 0; worker < CACHE_WORKERS; worker++)); do
        printf '+'
        printf ' %q' "$@" --shard-count "$CACHE_WORKERS" --shard-index "$worker"
        printf '\n'
        "$@" \
            --shard-count "$CACHE_WORKERS" \
            --shard-index "$worker" &
        pids+=("$!")
    done
    for worker in "${pids[@]}"; do
        if ! wait "$worker"; then
            status=1
        fi
    done
    (( status == 0 )) || return "$status"
}

canonical_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

while (($#)); do
    case "$1" in
        --shard-index)
            (($# >= 2)) || die "$1 requires a value"
            SHARD_INDEX="$2"; shift 2 ;;
        --artifact-root)
            (($# >= 2)) || die "$1 requires a value"
            ARTIFACT_ROOT="$2"; shift 2 ;;
        --library-root)
            (($# >= 2)) || die "$1 requires a value"
            LIBRARY_ROOT="$2"; shift 2 ;;
        --library-matrix)
            (($# >= 2)) || die "$1 requires a value"
            LIBRARY_MATRIX="$2"; shift 2 ;;
        -j|--jobs)
            (($# >= 2)) || die "$1 requires a value"
            JOBS="$2"; shift 2 ;;
        --cache-workers)
            (($# >= 2)) || die "$1 requires a value"
            CACHE_WORKERS="$2"; shift 2 ;;
        --matching-jobs)
            (($# >= 2)) || die "$1 requires a value"
            MATCHING_JOBS="$2"; shift 2 ;;
        --seed)
            (($# >= 2)) || die "$1 requires a value"
            SEED="$2"; shift 2 ;;
        --device)
            (($# >= 2)) || die "$1 requires a value"
            DEVICE="$2"; shift 2 ;;
        --clean)
            CLEAN=1; shift ;;
        --skip-fetch)
            SKIP_FETCH=1; shift ;;
        --skip-build)
            SKIP_BUILD=1; shift ;;
        --skip-cache)
            SKIP_CACHE=1; shift ;;
        --skip-matching)
            SKIP_MATCHING=1; shift ;;
        --offline-ablation-features)
            OFFLINE_ABLATION_FEATURES=1; shift ;;
        --cu-function-coverage)
            (($# >= 2)) || die "$1 requires a value"
            CU_FUNCTION_COVERAGE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "unknown argument: $1" ;;
    esac
done

[[ "$SHARD_INDEX" =~ ^[0-3]$ ]] || die "--shard-index must be 0, 1, 2 or 3"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be positive"
[[ "$CACHE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "--cache-workers must be positive"
[[ "$MATCHING_JOBS" == auto || "$MATCHING_JOBS" =~ ^[1-9][0-9]*$ ]] \
    || die "--matching-jobs must be auto or a positive integer"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"

COMPILERS=(gcc-11 gcc-13 clang-14 clang-18)
COMPILER="${COMPILERS[$SHARD_INDEX]}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DATASET_DIR/shards/libseeker-shard-$SHARD_INDEX}"
ARTIFACT_ROOT="$(canonical_path "$ARTIFACT_ROOT")"
LIBRARY_ROOT="$(canonical_path "$LIBRARY_ROOT")"
LIBRARY_MATRIX="$(canonical_path "$LIBRARY_MATRIX")"

[[ "$ARTIFACT_ROOT" != / ]] || die "artifact root cannot be /"
[[ -d "$LIBRARY_ROOT" || "$DRY_RUN" == 1 ]] || die "missing library root: $LIBRARY_ROOT"
[[ -f "$LIBRARY_MATRIX" || "$DRY_RUN" == 1 ]] || die "missing library matrix: $LIBRARY_MATRIX"

PYTHON=python3
[[ ! -x "$REPO_DIR/.venv/bin/python" ]] || PYTHON="$REPO_DIR/.venv/bin/python"

case "$COMPILER" in
    gcc-*) CXX_COMPILER="g++-${COMPILER#gcc-}" ;;
    clang-*) CXX_COMPILER="clang++-${COMPILER#clang-}" ;;
    *) die "unsupported compiler command: $COMPILER" ;;
esac

BUILD_PYTHON="$PYTHON"
BUILD_RUNNER=()
if ! command -v "$COMPILER" >/dev/null 2>&1 \
    || ! command -v "$CXX_COMPILER" >/dev/null 2>&1; then
    command -v docker >/dev/null 2>&1 \
        || die "missing $COMPILER/$CXX_COMPILER and docker is unavailable"
    docker image inspect "$TOOLCHAIN_IMAGE" >/dev/null 2>&1 \
        || die "missing toolchain image: $TOOLCHAIN_IMAGE"
    BUILD_PYTHON=python3
    BUILD_RUNNER=(
        docker run --rm
        --user "$(id -u):$(id -g)"
        --env HOME=/tmp
        --volume "$REPO_DIR:$REPO_DIR"
        --workdir "$REPO_DIR"
        "$TOOLCHAIN_IMAGE"
    )
    printf 'Compiler %s/%s unavailable on host; using Docker image %s.\n' \
        "$COMPILER" "$CXX_COMPILER" "$TOOLCHAIN_IMAGE"
fi

printf 'Shard %s/4: compiler=%s, expected ELF=876, output=%s\n' \
    "$SHARD_INDEX" "$COMPILER" "$ARTIFACT_ROOT"

if (( ! SKIP_FETCH )); then
    run "$DATASET_DIR/create_datasets.sh" fetch --artifact-root "$ARTIFACT_ROOT"
fi

if (( ! SKIP_BUILD )); then
    build_args=(
    "${BUILD_RUNNER[@]}"
    "$BUILD_PYTHON" "$SCRIPT_DIR/build_randomized_elf_matrix.py"
    --profile libseeker
    --compiler "$COMPILER"
    --elf-optimizations O0,O2,O3,Os
    --seed "$SEED"
    --lib-root "$LIBRARY_ROOT"
    --library-matrix "$LIBRARY_MATRIX"
    --output-root "$ARTIFACT_ROOT/builds/programs"
    --ground-truth-root "$ARTIFACT_ROOT/ground_truth/legacy-libseeker-primary"
    --jobs "$JOBS"
    --continue-on-error
    --skip-existing
    )
    if (( CLEAN )); then
        build_args=("${build_args[@]/--skip-existing/--clean}")
    fi
    (( DRY_RUN )) && build_args+=(--dry-run)
    run "${build_args[@]}"

    run "$PYTHON" "$SCRIPT_DIR/assemble_datasets.py" \
        --profile libseeker \
        --compiler "$COMPILER" \
        --artifact-root "$ARTIFACT_ROOT" \
        --build-root "$ARTIFACT_ROOT/builds/programs" \
        --copy-mode hardlink \
        --strict

    run "$PYTHON" "$SCRIPT_DIR/validate_datasets.py" \
        --profile libseeker \
        --compiler "$COMPILER" \
        --artifact-root "$ARTIFACT_ROOT"
else
    [[ -f "$ARTIFACT_ROOT/datasets/libseeker/manifest.csv" ]] \
        || die "missing assembled dataset for --skip-build"
fi

CACHE_DIR="$ARTIFACT_ROOT/cache"
if (( ! SKIP_CACHE )); then
    run_parallel_cache "$PYTHON" "$REPO_DIR/thesis_code/build_analysis_cache.py" \
        --dataset-dir "$ARTIFACT_ROOT/datasets/libseeker" \
        --cache-dir "$CACHE_DIR" \
        --device "$DEVICE" \
        --fail-fast
    run "$PYTHON" "$REPO_DIR/thesis_code/build_analysis_cache.py" \
        --library-matrix "$LIBRARY_MATRIX" \
        --library-root "$LIBRARY_ROOT" \
        --cache-dir "$CACHE_DIR" \
        --device "$DEVICE" \
        --provenance-path "$CACHE_DIR/library_provenance.shard$SHARD_INDEX.jsonl" \
        --fail-fast
    run "$PYTHON" "$REPO_DIR/thesis_code/build_experiment_manifest.py" \
        --dataset-root "$ARTIFACT_ROOT" \
        --cache-dir "$CACHE_DIR"
    run "$PYTHON" "$SCRIPT_DIR/validate_libseeker_shard.py" \
        --shard-index "$SHARD_INDEX" \
        --artifact-root "$ARTIFACT_ROOT" \
        --library-matrix "$LIBRARY_MATRIX" \
        --library-root "$LIBRARY_ROOT"
fi

if (( ! SKIP_MATCHING )); then
    MATCHING_EXTRA_ARGS=()
    if (( OFFLINE_ABLATION_FEATURES )); then
        MATCHING_EXTRA_ARGS+=(--offline-ablation-features)
    fi
    MATCHING_EXTRA_ARGS+=(--cu-min-function-coverage "$CU_FUNCTION_COVERAGE")
    run "$PYTHON" "$REPO_DIR/thesis_code/run_libseeker_batch.py" \
        --dataset-dir "$ARTIFACT_ROOT/datasets/libseeker/binaries" \
        --dataset-manifest "$ARTIFACT_ROOT/datasets/libseeker/manifest.json" \
        --library-matrix "$LIBRARY_MATRIX" \
        --library-root "$LIBRARY_ROOT" \
        --output-dir "$ARTIFACT_ROOT/results/libseeker" \
        --result-log "$ARTIFACT_ROOT/results/libseeker/results.jsonl.gz" \
        --ground-truth-dir "$ARTIFACT_ROOT/ground_truth/libseeker" \
        --analysis-cache-dir "$CACHE_DIR" \
        --pipeline current \
        --all-elfs \
        --jobs "$MATCHING_JOBS" \
        --device "$DEVICE" \
        "${MATCHING_EXTRA_ARGS[@]}" \
        --resume
fi

if (( DRY_RUN )); then
    printf 'Dry run complete for shard %s.\n' "$SHARD_INDEX"
else
    printf 'Completed shard %s: %s\n' "$SHARD_INDEX" "$ARTIFACT_ROOT"
fi
