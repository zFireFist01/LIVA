#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$SCRIPT_DIR/../../.." || exit 1

run_step() {
    local name="$1"
    local log="$2"
    shift 2
    mkdir -p "$(dirname -- "$log")"
    printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    printf '[%s] END %s exit=%s\n' "$(date --iso-8601=seconds)" "$name" "$rc" | tee -a "$log"
    return "$rc"
}

overall=0

run_step \
    task1 \
    libseeker-unified/family_f1_fixed_all.run.log \
    python3 thesis_code/experiments/evaluate_fixed_family_f1.py \
        --workers 8 \
        --output libseeker-unified/family_f1_fixed_all.json \
    || overall=1

run_step \
    task2 \
    libseeker-unified/task2_source_verified/task2_run.log \
    python3 thesis_code/experiments/evaluate_fixed_task2_source.py \
        --workers 8 \
        --limit-per-compiler 0 \
        --output-dir libseeker-unified/task2_source_verified \
    || overall=1

run_step \
    task2_evaluable \
    libseeker-unified/task2_evaluable/task2_evaluable_run.log \
    python3 thesis_code/experiments/reclassify_task2_evaluable.py \
        --output-dir libseeker-unified/task2_evaluable \
    || overall=1

run_step \
    ablation \
    libseeker-unified/ablation_task1_fixed/ablation_run.log \
    python3 thesis_code/experiments/offline_family_ablation_fixed.py \
        --workers 8 \
        --output-dir libseeker-unified/ablation_task1_fixed \
    || overall=1

printf '\n[%s] FINAL_METRICS_FINISHED exit=%s\n' "$(date --iso-8601=seconds)" "$overall"
exit "$overall"
