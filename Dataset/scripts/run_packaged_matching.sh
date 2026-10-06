#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$DATASET_DIR/.." && pwd)"

PYTHON="${LIVA_PYTHON:-python3}"
[[ ! -x "$REPO_DIR/.venv/bin/python" ]] || PYTHON="$REPO_DIR/.venv/bin/python"
DEVICE="${LIVA_MATCHING_DEVICE:-auto}"
REQUESTED_JOBS="${LIVA_MATCHING_JOBS:-auto}"
ADAPTIVE_MAX_JOBS="${LIVA_MATCHING_MAX_JOBS:-15}"

shopt -s nullglob
inventories=(
    "$DATASET_DIR"/shards/libseeker-shard-*/cache/shard_inventory.json
)
(( ${#inventories[@]} == 1 )) || {
    printf 'Expected exactly one packaged shard inventory, found %s\n' \
        "${#inventories[@]}" >&2
    exit 1
}

INVENTORY="${inventories[0]}"
CACHE_DIR="$(dirname -- "$INVENTORY")"
SHARD_ROOT="$(dirname -- "$CACHE_DIR")"

cache_valid="$("$PYTHON" - "$INVENTORY" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    inventory = json.load(stream)
valid = (
    inventory.get("valid") is True
    and inventory.get("errors") == []
    and inventory.get("full_library_catalog") is True
    and inventory.get("experiment_cache_records") == inventory.get("elf_records")
    and inventory.get("library_archives_cached")
    == inventory.get("selected_library_archives_total")
)
print("yes" if valid else "no")
PY
)"

[[ "$cache_valid" == yes ]] || {
    printf 'Refusing cache-only matching: packaged cache inventory is incomplete\n' >&2
    exit 1
}

MATCH_CPU="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
[[ "$MATCH_CPU" =~ ^[1-9][0-9]*$ ]] || MATCH_CPU=1
[[ "$ADAPTIVE_MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'LIVA_MATCHING_MAX_JOBS must be a positive integer\n' >&2
    exit 2
}

if [[ "$REQUESTED_JOBS" == auto ]]; then
    MATCH_JOBS=auto
    if (( MATCH_CPU < ADAPTIVE_MAX_JOBS )); then
        DISPLAY_JOBS="$MATCH_CPU"
    else
        DISPLAY_JOBS="$ADAPTIVE_MAX_JOBS"
    fi
    MATCH_THREADS=1
    JOB_REASON="adaptive 1..$DISPLAY_JOBS; continuous CPU/RAM guard"
elif [[ "$REQUESTED_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    MATCH_JOBS="$REQUESTED_JOBS"
    DISPLAY_JOBS="$MATCH_JOBS"
    MATCH_THREADS=$((MATCH_CPU / MATCH_JOBS))
    (( MATCH_THREADS >= 1 )) || MATCH_THREADS=1
    JOB_REASON="LIVA_MATCHING_JOBS override"
else
    printf 'LIVA_MATCHING_JOBS must be auto or a positive integer\n' >&2
    exit 2
fi

export OMP_NUM_THREADS="$MATCH_THREADS"
export OPENBLAS_NUM_THREADS="$MATCH_THREADS"
export MKL_NUM_THREADS="$MATCH_THREADS"
export NUMEXPR_NUM_THREADS="$MATCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$MATCH_THREADS"
export BLIS_NUM_THREADS="$MATCH_THREADS"

# A cached CU retains one mmap descriptor while its archive is being matched.
# The largest packaged archives stay below this budget, and raising the soft
# limit also leaves headroom for Python, gzip and numerical-library files.
MATCH_NOFILE_SOFT="$(ulimit -Sn)"
MATCH_NOFILE_HARD="$(ulimit -Hn)"
MATCH_NOFILE_TARGET=65536
if [[ "$MATCH_NOFILE_HARD" =~ ^[0-9]+$ ]] \
    && (( MATCH_NOFILE_TARGET > MATCH_NOFILE_HARD )); then
    MATCH_NOFILE_TARGET="$MATCH_NOFILE_HARD"
fi
if [[ "$MATCH_NOFILE_SOFT" =~ ^[0-9]+$ ]] \
    && (( MATCH_NOFILE_TARGET > MATCH_NOFILE_SOFT )); then
    if ! ulimit -Sn "$MATCH_NOFILE_TARGET"; then
        printf 'Warning: could not raise open-file limit above %s\n' \
            "$MATCH_NOFILE_SOFT" >&2
    fi
fi

printf 'Shard: %s\n' "$(basename -- "$SHARD_ROOT")"
printf 'Matching: %s process(es), %s numerical thread(s) per process (%s)\n' \
    "$DISPLAY_JOBS" "$MATCH_THREADS" "$JOB_REASON"
printf 'Analysis cache: cache-only (read-only; radare2 disabled)\n'
printf 'Open-file soft limit: %s\n' "$(ulimit -Sn)"

exec "$PYTHON" "$REPO_DIR/thesis_code/run_libseeker_batch.py" \
    --dataset-dir "$SHARD_ROOT/datasets/libseeker/binaries" \
    --dataset-manifest "$SHARD_ROOT/datasets/libseeker/manifest.json" \
    --library-matrix "$DATASET_DIR/manifests/library_matrix.tsv" \
    --library-root "$DATASET_DIR/builds/libraries" \
    --output-dir "$SHARD_ROOT/results/libseeker" \
    --result-log "$SHARD_ROOT/results/libseeker/results.jsonl.gz" \
    --ground-truth-dir "$SHARD_ROOT/ground_truth/libseeker" \
    --analysis-cache-dir "$CACHE_DIR" \
    --analysis-cache-only \
    --pipeline current \
    --all-elfs \
    --jobs "$MATCH_JOBS" \
    --adaptive-max-jobs "$ADAPTIVE_MAX_JOBS" \
    --device "$DEVICE" \
    --resume \
    "$@"
