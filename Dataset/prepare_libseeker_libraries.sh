#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
DRY_RUN=0

while (($#)); do
    case "$1" in
        -j|--jobs)
            (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 1; }
            JOBS="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: Dataset/prepare_libseeker_libraries.sh [-j N] [--dry-run]

Prepare the common read-only input for the four ELF/cache PCs: fetch and build
the current plus explicit minor/major alternative versions, retain the 164 reference archives
and the five extra archives needed for ELF linking, generate library_matrix.tsv
and normalize selected archives.
EOF
            exit 0 ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
    esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' '--jobs must be positive' >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    (( DRY_RUN )) || "$@"
}

historical_args=(historical all --use-covered --jobs "$JOBS")
(( DRY_RUN )) && historical_args+=(--dry-run)
run "$SCRIPT_DIR/scripts/build_libraries.sh" "${historical_args[@]}"
run python3 "$SCRIPT_DIR/scripts/prune_library_archives.py" --apply
run python3 "$SCRIPT_DIR/scripts/materialize_selected_archives.py"

printf 'Common library input ready: %s\n' "$SCRIPT_DIR/builds/libraries"
printf 'Copy that directory and manifests/library_matrix.tsv unchanged to every PC.\n'
