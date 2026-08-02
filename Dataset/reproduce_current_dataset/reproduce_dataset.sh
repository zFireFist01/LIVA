#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATASET_DIR="$REPO_DIR/Dataset"
MATRIX_MANIFEST="$DATASET_DIR/builds/elf_builds/randomized_matrix.json"

ACTION="all"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
DRY_RUN=0
CLEAN=0
SKIP_EXISTING=1

usage() {
    cat <<'EOF'
Usage: reproduce_dataset.sh [action] [options]

Reproduce the current Dataset snapshot from sources:
  1. fetch source code listed in Dataset/manifests/source_manifest.json
  2. build all static libraries at O0,O2,O3,Os (O1 is excluded)
  3. build the exact randomized ELF cases listed in Dataset/builds/elf_builds/randomized_matrix.json
  4. refresh CU/library ground truth from final-linker maps

Actions:
  all         Run fetch, libraries, elves and ground-truth refresh (default)
  fetch       Fetch/extract source archives only
  libraries   Build static libraries only
  elves       Build randomized ELF cases only
  ground-truth
              Relink existing ELF build targets and refresh GroundTruth/matrix metadata

Options:
  -j, --jobs N       Parallel jobs (default: number of CPUs)
      --dry-run      Print commands without running them
      --skip-existing
                     Reuse existing library/ELF outputs where possible (default)
      --clean        Remove and rebuild requested build variants
      --no-clean     Do not remove existing build variants first (default)
  -h, --help         Show this help

Examples:
  Dataset/reproduce_current_dataset/reproduce_dataset.sh --dry-run
  Dataset/reproduce_current_dataset/reproduce_dataset.sh libraries -j 16
  Dataset/reproduce_current_dataset/reproduce_dataset.sh libraries --clean
  Dataset/reproduce_current_dataset/reproduce_dataset.sh elves --skip-existing
  Dataset/reproduce_current_dataset/reproduce_dataset.sh ground-truth
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then
        "$@"
    fi
}

compiler_id() {
    local compiler="$1"
    local version
    version="$("$compiler" -dumpfullversion -dumpversion)"
    printf '%s-%s' "$(basename -- "$compiler")" "$version"
}

select_library_root() {
    local preferred="$DATASET_DIR/builds/lib_builds/gcc-16.1.1"
    local detected="$DATASET_DIR/builds/lib_builds/$(compiler_id gcc)"
    local candidate

    if [[ -d "$preferred" ]]; then
        printf '%s' "$preferred"
        return
    fi
    if [[ -d "$detected" ]]; then
        printf '%s' "$detected"
        return
    fi
    candidate="$(find "$DATASET_DIR/builds/lib_builds" \
        -mindepth 1 -maxdepth 1 -type d -name 'gcc-*' -print \
        2>/dev/null | sort -V | tail -n 1)"
    printf '%s' "${candidate:-$detected}"
}

warn_toolchain() {
    local gcc_id=""
    local clang_id=""

    if command -v gcc >/dev/null 2>&1; then
        gcc_id="$(compiler_id gcc)"
    else
        die "gcc not found"
    fi

    if command -v clang >/dev/null 2>&1; then
        clang_id="$(compiler_id clang)"
    else
        die "clang not found"
    fi

    printf 'Detected toolchains: %s, %s\n' "$gcc_id" "$clang_id"
    if [[ "$gcc_id" != "gcc-16.1.1" || "$clang_id" != "clang-22.1.6" ]]; then
        printf 'Warning: the current dataset was generated with gcc-16.1.1 and clang-22.1.6.\n' >&2
        printf '         Different compiler versions can reproduce the structure, but not byte-identical binaries.\n' >&2
    fi
}

fetch_sources() {
    run_cmd "$DATASET_DIR/scripts/fetch_dataset_sources.sh" --skip-origin-files
}

build_libraries() {
    local compiler_args
    local failures=()
    local -a common_args=(
        -o O0,O2,O3,Os
        -j "$JOBS"
        --continue-on-error
    )

    if (( CLEAN )); then
        common_args+=(--clean)
    fi
    if (( SKIP_EXISTING )); then
        common_args+=(--skip-existing)
    fi

    compiler_args=(env CC=gcc CXX=g++ "$DATASET_DIR/scripts/build_libraries.sh")
    if ! run_cmd "${compiler_args[@]}" "${common_args[@]}"; then
        failures+=(gcc)
    fi

    compiler_args=(env CC=clang CXX=clang++ "$DATASET_DIR/scripts/build_libraries.sh")
    if ! run_cmd "${compiler_args[@]}" "${common_args[@]}"; then
        failures+=(clang)
    fi

    if ((${#failures[@]})); then
        printf 'Library build completed with failures in: %s\n' "${failures[*]}" >&2
        return 1
    fi
}

build_elves() {
    local library_root
    library_root="$(select_library_root)"

    [[ -f "$MATRIX_MANIFEST" ]] || die "missing matrix: $MATRIX_MANIFEST"

    local args=(
        python3 "$DATASET_DIR/scripts/build_randomized_elf_matrix.py"
        --matrix "$MATRIX_MANIFEST"
        --lib-root "$library_root"
        --jobs "$JOBS"
        --continue-on-error
    )

    if (( CLEAN )); then
        args+=(--clean)
    fi
    if (( SKIP_EXISTING )); then
        args+=(--skip-existing)
    fi
    if (( DRY_RUN )); then
        args+=(--dry-run)
    fi

    run_cmd "${args[@]}"
}

refresh_ground_truth() {
    local library_root
    library_root="$(select_library_root)"
    [[ -f "$MATRIX_MANIFEST" ]] || die "missing matrix: $MATRIX_MANIFEST"

    local args=(
        python3 "$DATASET_DIR/scripts/build_randomized_elf_matrix.py"
        --refresh-ground-truth
        --matrix "$MATRIX_MANIFEST"
        --lib-root "$library_root"
        --continue-on-error
    )

    if (( DRY_RUN )); then
        args+=(--dry-run)
    fi

    run_cmd "${args[@]}"
}

parse_args() {
    while (($#)); do
        case "$1" in
            all|fetch|libraries|elves|ground-truth)
                ACTION="$1"
                shift
                ;;
            -j|--jobs)
                [[ $# -ge 2 ]] || die "$1 requires a value"
                JOBS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --skip-existing)
                SKIP_EXISTING=1
                CLEAN=0
                shift
                ;;
            --clean)
                CLEAN=1
                SKIP_EXISTING=0
                shift
                ;;
            --no-clean)
                CLEAN=0
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown argument: $1"
                ;;
        esac
    done

    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
    if (( SKIP_EXISTING )); then
        CLEAN=0
    fi
}

main() {
    parse_args "$@"
    cd "$REPO_DIR"
    warn_toolchain

    case "$ACTION" in
        all)
            fetch_sources
            build_libraries
            build_elves
            refresh_ground_truth
            ;;
        fetch)
            fetch_sources
            ;;
        libraries)
            build_libraries
            ;;
        elves)
            build_elves
            refresh_ground_truth
            ;;
        ground-truth)
            refresh_ground_truth
            ;;
    esac
}

main "$@"
