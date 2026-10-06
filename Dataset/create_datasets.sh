#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
DEFAULT_ARTIFACT_ROOT="$SCRIPT_DIR"

ACTION=all
ARTIFACT_ROOT="$DEFAULT_ARTIFACT_ROOT"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
SEED=20260731
STRICT=1
DRY_RUN=0
CLEAN=0
SKIP_EXISTING=1
INSIDE_CONTAINER=0
LIBRARY_ROOT=""
LIBRARY_MATRIX="$SCRIPT_DIR/manifests/library_matrix.tsv"

TOOLCHAINS=(
    "gcc-11|g++-11"
    "gcc-13|g++-13"
    "clang-14|clang++-14"
    "clang-18|clang++-18"
)

LIBRARIES=(
    zlib-1.3.1
    bzip2
    xz
    openssl-3.5.0
    attr
    acl-2.3.2
    file-5.46
    libcap
    glibc-2.41
    gmp-6.3.0
    libiconv-1.18
    mpfr-4.2.1
    ncurses-6.5
    pcre2-10.47
    readline-8.2
    selinux-3.7
    libunistring-1.4.2
    libidn2-2.3.8
    libpsl-0.21.5
    brotli
    zstd-1.5.7
)

SOURCES=(
    gawk-5.3.2 gzip-1.13 grep-3.11 less-668 nano-8.3
    openssh-portable-V_10_0_P2 sed-4.9 bash-5.3-beta gnuchess-6.2.11
    inetutils-2.6 make-4.4.1 rsync-3.4.1 socat-1.8.0.3 tar-1.35
    wget2-2.2.0 coreutils-9.6 util-linux-v2.39.3 vim-v9.1.1151
    "${LIBRARIES[@]}"
)

usage() {
    cat <<'EOF'
Uso: Dataset/create_datasets.sh [azione] [opzioni]

Azioni:
  all                 costruisce, assembla e valida entrambi i dataset (default)
  libseeker           costruisce e assembla i 3504 ELF del Dataset A
  unseen              costruisce e assembla i 1200 ELF glibc del Dataset B
  collect-libseeker   rigenera Dataset A e ground truth dalle build esistenti
  collect-unseen      rigenera Dataset B e ground truth dalle build esistenti
  collect             rigenera entrambi dalle build esistenti
  validate            esegue soltanto la validazione finale dei due dataset
  fetch               scarica e verifica soltanto i sorgenti congelati

Opzioni:
  -j, --jobs N              job paralleli (default: CPU disponibili)
      --artifact-root DIR   root degli output (default: Dataset)
      --library-root DIR    riusa la root multi-toolchain che contiene
                            <compiler-id>/<sorgente>/<opt>/install/lib/*.a
      --library-matrix FILE matrice versionata delle librerie da bilanciare
      --seed N              seed selezione librerie Dataset A (default: 20260731)
      --clean               ricrea build e output richiesti
      --skip-existing       riprende build complete (default)
      --strict              richiede matrici complete (default)
      --no-strict           consente raccolte parziali per debug/smoke test
      --dry-run             stampa le azioni senza scrivere
  -h, --help                mostra questo testo

Ambiente riproducibile consigliato:
  Dataset/scripts/run_reproduction_container.sh all -j 4

Output di default:
  Dataset/builds/{libraries,programs}
  Dataset/datasets/{libseeker,unseen}
  Dataset/ground_truth/{libseeker,unseen}
EOF
}

die() {
    printf 'Errore: %s\n' "$*" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then
        "$@"
    fi
}

join_csv() {
    local IFS=,
    printf '%s' "$*"
}

compiler_id() {
    local compiler="$1"
    local version
    version="$("$compiler" -dumpfullversion -dumpversion)"
    printf '%s-%s' "$(basename -- "$compiler")" "$version"
}

canonical_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

parse_args() {
    while (($#)); do
        case "$1" in
            all|libseeker|unseen|collect-libseeker|collect-unseen|collect|validate|fetch)
                ACTION="$1"; shift ;;
            -j|--jobs)
                (($# >= 2)) || die "$1 richiede un valore"
                JOBS="$2"; shift 2 ;;
            --artifact-root)
                (($# >= 2)) || die "$1 richiede un valore"
                ARTIFACT_ROOT="$2"; shift 2 ;;
            --library-root)
                (($# >= 2)) || die "$1 richiede un valore"
                LIBRARY_ROOT="$2"; shift 2 ;;
            --library-matrix)
                (($# >= 2)) || die "$1 richiede un valore"
                LIBRARY_MATRIX="$2"; shift 2 ;;
            --seed)
                (($# >= 2)) || die "$1 richiede un valore"
                SEED="$2"; shift 2 ;;
            --clean)
                CLEAN=1; SKIP_EXISTING=0; shift ;;
            --skip-existing)
                SKIP_EXISTING=1; CLEAN=0; shift ;;
            --strict)
                STRICT=1; shift ;;
            --no-strict)
                STRICT=0; shift ;;
            --dry-run)
                DRY_RUN=1; shift ;;
            --inside-container)
                INSIDE_CONTAINER=1; shift ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                die "argomento sconosciuto: $1" ;;
        esac
    done
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs deve essere positivo"
    [[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed deve essere un intero non negativo"
}

validate_paths() {
    ARTIFACT_ROOT="$(canonical_path "$ARTIFACT_ROOT")"
    [[ "$ARTIFACT_ROOT" != / ]] || die "--artifact-root non può essere /"
    [[ "$ARTIFACT_ROOT" != "$(canonical_path "${HOME:?}")" ]] ||
        die "--artifact-root non può essere HOME"
    if [[ -n "$LIBRARY_ROOT" ]]; then
        LIBRARY_ROOT="$(canonical_path "$LIBRARY_ROOT")"
        [[ -d "$LIBRARY_ROOT" || "$DRY_RUN" == 1 ]] ||
            die "--library-root inesistente: $LIBRARY_ROOT"
    fi
    LIBRARY_MATRIX="$(canonical_path "$LIBRARY_MATRIX")"
    [[ -f "$LIBRARY_MATRIX" || "$DRY_RUN" == 1 ]] ||
        die "--library-matrix inesistente: $LIBRARY_MATRIX"
}

fetch_sources() {
    local args=(
        "$SCRIPTS_DIR/fetch_dataset_sources.sh"
        --skip-origin-files
        --only-kind elf
        --only-kind library
        --only-kind unseen-program
        --only-kind unseen-library
        --only-name "$(join_csv "${SOURCES[@]}")"
    )
    (( DRY_RUN )) && args+=(--dry-run)
    run "${args[@]}"
}

build_libraries() {
    local library_build_root="$ARTIFACT_ROOT/builds/libraries"
    local toolchain cc cxx library
    local -a libraries_for_toolchain

    if [[ -n "$LIBRARY_ROOT" ]]; then
        printf 'REUSE librerie multi-toolchain: %s\n' "$LIBRARY_ROOT"
        return
    fi

    for toolchain in "${TOOLCHAINS[@]}"; do
        IFS='|' read -r cc cxx <<< "$toolchain"

        libraries_for_toolchain=("${LIBRARIES[@]}")

        # glibc viene costruita soltanto con GCC.
        # Le altre librerie continuano a essere costruite con tutte le toolchain.
        if [[ "$cc" == clang-* ]]; then
            libraries_for_toolchain=()

            for library in "${LIBRARIES[@]}"; do
                [[ "$library" == "glibc-2.41" ]] && continue
                libraries_for_toolchain+=("$library")
            done
        fi

        local args=(
            "$SCRIPTS_DIR/build_libraries.sh"
            --libraries "$(join_csv "${libraries_for_toolchain[@]}")"
            --optimizations O0,O2,O3,Os
            --jobs "$JOBS"
            --cc "$cc"
            --cxx "$cxx"
            --output "$library_build_root"
            --continue-on-error
        )

        (( CLEAN )) && args+=(--clean)
        (( SKIP_EXISTING )) && args+=(--skip-existing)
        (( DRY_RUN )) && args+=(--dry-run)

        run "${args[@]}"
    done

    # La root passata al generatore contiene tutte le directory compiler-id.
    LIBRARY_ROOT="$library_build_root"
}

build_libseeker() {
    build_libraries
    local compiler optimization_csv program_csv program
    local -a cell_programs
    while IFS='|' read -r compiler optimization_csv program_csv; do
        local args=(
            python3 "$SCRIPTS_DIR/build_randomized_elf_matrix.py"
            --profile libseeker
            --compiler "$compiler"
            --elf-optimizations "$optimization_csv"
            --seed "$SEED"
            --lib-root "$LIBRARY_ROOT"
            --library-matrix "$LIBRARY_MATRIX"
            --output-root "$ARTIFACT_ROOT/builds/programs"
            --ground-truth-root "$ARTIFACT_ROOT/ground_truth/legacy-libseeker-primary-balanced"
            --jobs "$JOBS"
            --continue-on-error
        )
        IFS=',' read -r -a cell_programs <<< "$program_csv"
        for program in "${cell_programs[@]}"; do
            args+=(--program "$program")
        done
        (( CLEAN )) && args+=(--clean)
        (( SKIP_EXISTING )) && args+=(--skip-existing)
        run "${args[@]}"
    done <<'EOF'
gcc-11|O0|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-11|O2|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-11|O3|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-11|Os|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-13|O0|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-13|O2|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-13|O3|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
gcc-13|Os|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-14|O0|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-14|O2|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-14|O3|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-14|Os|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-18|O0|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-18|O2|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-18|O3|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
clang-18|Os|bash,coreutils,gawk,gnuchess,grep,inetutils,less,make,nano,openssh,rsync,sed,socat,tar,util-linux,vim,wget2
EOF
}

build_unseen() {
    build_libseeker
}

collect_profile() {
    local profile="$1"
    local build_root
    if [[ "$profile" == libseeker ]]; then
        build_root="$ARTIFACT_ROOT/builds/programs"
    else
        build_root="$ARTIFACT_ROOT/builds/programs"
    fi
    local args=(
        python3 "$SCRIPTS_DIR/assemble_datasets.py"
        --profile "$profile"
        --artifact-root "$ARTIFACT_ROOT"
        --build-root "$build_root"
        --copy-mode hardlink
    )
    (( CLEAN )) && args+=(--clean)
    (( STRICT )) && args+=(--strict)
    (( DRY_RUN )) && args+=(--dry-run)
    run "${args[@]}"
}

validate_all() {
    local args=(
        python3 "$SCRIPTS_DIR/validate_datasets.py"
        --artifact-root "$ARTIFACT_ROOT"
    )
    run "${args[@]}"
}

validate_profile() {
    local profile="$1"
    run python3 "$SCRIPTS_DIR/validate_datasets.py" \
        --artifact-root "$ARTIFACT_ROOT" --profile "$profile"
}

main() {
    parse_args "$@"
    validate_paths
    cd "$REPO_DIR"

    printf 'ACTION=%s ARTIFACT_ROOT=%s JOBS=%s CONTAINER=%s\n' \
        "$ACTION" "$ARTIFACT_ROOT" "$JOBS" "$INSIDE_CONTAINER"
    case "$ACTION" in
        all)
            fetch_sources
            build_libseeker
            collect_profile libseeker
            collect_profile unseen
            (( STRICT )) && validate_all
            ;;
        libseeker)
            fetch_sources
            build_libseeker
            collect_profile libseeker
            (( STRICT )) && validate_profile libseeker
            ;;
        unseen)
            fetch_sources
            build_unseen
            collect_profile unseen
            (( STRICT )) && validate_profile unseen
            ;;
        collect-libseeker)
            collect_profile libseeker
            (( STRICT )) && validate_profile libseeker
            ;;
        collect-unseen)
            collect_profile unseen
            (( STRICT )) && validate_profile unseen
            ;;
        collect)
            collect_profile libseeker
            collect_profile unseen
            (( STRICT )) && validate_all
            ;;
        validate)
            validate_all
            ;;
        fetch)
            fetch_sources
            ;;
    esac
    printf 'Completato: %s\n' "$ACTION"
}

main "$@"
