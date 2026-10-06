#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$DATASET_DIR/sources/lib_sources"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/builds/libraries}"

CURATED_LIBRARIES=(
    zlib-1.3.1
    bzip2
    xz
    openssl
    openssl-3.5.0
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
EXCLUDED_LIBRARIES=(
    at-spi2-core_2.50.0.orig
    at-spi2-core_2.51.90.orig
    at-spi2-core_2.52.0.orig
    fribidi
    gettext-1.0
    glibc
    gpm
    gtk
    libjpeg8-empty
    libice
    libsm
    libsodium
    libx11
    libxcb
    tcp-wrappers
    util-linux
    zlib
)
DEFAULT_OPTIMIZATIONS=(O0 O2 O3 Os)

LIBRARIES=()
OPTIMIZATIONS=("${DEFAULT_OPTIMIZATIONS[@]}")
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
CLEAN=0
INSTALL=1
CONTINUE_ON_ERROR=0
SKIP_EXISTING=0
DRY_RUN=0
LIST_LIBRARIES=0
STATIC_EXECUTABLES=0
ALLOW_EXCLUDED=0

discover_source_libraries() {
    [[ -d "$SOURCE_DIR" ]] || return 0
    find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        sort
}

is_excluded_library() {
    local library="$1"
    local excluded

    (( ALLOW_EXCLUDED )) && return 1

    for excluded in "${EXCLUDED_LIBRARIES[@]}"; do
        [[ "$library" == "$excluded" ]] && return 0
    done
    return 1
}

default_libraries() {
    local -A seen=()
    local library

    # Keep the libraries already used by the dataset first, then append every
    # other source directory deterministically. This preserves dependency order
    # for the current ELF matrix while still expanding to the full source set.
    for library in "${CURATED_LIBRARIES[@]}"; do
        if [[ -d "$SOURCE_DIR/$library" ]] && ! is_excluded_library "$library"; then
            printf '%s\n' "$library"
            seen["$library"]=1
        fi
    done

    while IFS= read -r library; do
        [[ -n "$library" && -z "${seen[$library]:-}" ]] || continue
        is_excluded_library "$library" && continue
        printf '%s\n' "$library"
    done < <(discover_source_libraries)
}

usage() {
    cat <<'EOF'
Uso: build_libraries.sh [opzioni]

Compila le librerie in Dataset/sources/lib_sources con più livelli di
ottimizzazione, mantenendo build, installazione e log separati.

Opzioni:
  -l, --libraries LIST       Librerie separate da virgola
                             (default: tutte le directory in lib_sources)
      --curated              Usa solo il set curato già usato dagli ELF attuali
  -o, --optimizations LIST   Livelli separati da virgola, con o senza "-"
                             (default: O0,O2,O3,Os; O1 non è ammesso)
  -j, --jobs N               Numero di job paralleli
      --cc COMPILER          Compilatore C (default: gcc o $CC)
      --cxx COMPILER         Compilatore C++ (default: g++ o $CXX)
      --output DIR           Directory di output
      --clean                Ricrea le directory delle varianti richieste
      --skip-existing        Salta le varianti che hanno già install/lib/*.a
      --no-install           Compila senza eseguire "make install"
      --static-executables   Forza il linking statico anche per le utility
                             installate da ciascun progetto
      --continue-on-error    Continua con le altre varianti dopo un errore
      --dry-run              Mostra cosa verrebbe compilato senza eseguire
      --list-libraries       Lista le librerie selezionate e termina
  -h, --help                 Mostra questo messaggio

Esempi:
  ./build_libraries.sh
  ./build_libraries.sh --skip-existing --continue-on-error
  ./build_libraries.sh --curated
  ./build_libraries.sh -l pcre2-10.47,libiconv-1.18 -o O0,O2,O3
  ./build_libraries.sh -l bzip2,xz,pcre2-10.47 --static-executables
  CC=clang CXX=clang++ ./build_libraries.sh -o O2,Os --clean

Nota: glibc non può essere compilata con -O0; tale combinazione viene saltata.
      Lo script supporta ricette dedicate, configure/autogen/autoreconf,
      CMake, Meson e Makefile semplici. Alcune sorgenti extra possono richiedere
      tool esterni come cmake, meson e ninja, oppure fallire se sono solo
      placeholder/packaging.
EOF
}

die() {
    printf 'Errore: %s\n' "$*" >&2
    exit 1
}

split_csv() {
    local value="$1"
    local -n destination="$2"
    IFS=',' read -r -a destination <<< "$value"
}

normalize_optimizations() {
    local index optimization

    for index in "${!OPTIMIZATIONS[@]}"; do
        optimization="${OPTIMIZATIONS[$index]#-}"
        case "$optimization" in
            O0|O2|O3|Os|Oz|Ofast)
                OPTIMIZATIONS[$index]="$optimization"
                ;;
            *)
                die "livello di ottimizzazione non supportato: $optimization"
                ;;
        esac
    done
}

validate_libraries() {
    local library source_root build_system

    for library in "${LIBRARIES[@]}"; do
        is_excluded_library "$library" &&
            die "'$library' è esclusa dal dataset selezionato"
        [[ -d "$SOURCE_DIR/$library" ]] ||
            die "sorgenti non trovati per '$library' in $SOURCE_DIR"
        skip_nonbuildable_source "$library" && continue
        source_root="$(source_root_for_library "$SOURCE_DIR/$library")"
        build_system="$(build_system_for_source "$library" "$source_root")"
        if [[ "$build_system" == unknown ]]; then
            printf 'WARN  %-28s nessun build system riconosciuto\n' "$library" >&2
        fi
    done
}

compiler_id() {
    local name version
    name="$(basename -- "$CC")"
    version="$("$CC" -dumpfullversion -dumpversion 2>/dev/null || true)"
    version="${version:-unknown}"
    printf '%s-%s' "$name" "$version"
}

gmp_dependency_for_mpfr() {
    local library="$1"
    local compiler="${2:-}"
    local optimization="${3:-}"

    case "$library" in
        mpfr-3.1.6)
            printf 'gmp-5.1'
            ;;
        mpfr-4.1.0)
            printf 'gmp-6.2'
            ;;
        *)
            # The historical manifest calls the current source gmp-6.3,
            # while the curated/current dataset calls it gmp-6.3.0. Prefer
            # the historical build when it exists in this output tree.
            if [[ -n "$compiler" && -n "$optimization" &&
                  -d "$OUTPUT_DIR/$compiler/gmp-6.3/$optimization/install" ]]; then
                printf 'gmp-6.3'
            else
                printf 'gmp-6.3.0'
            fi
            ;;
    esac
}

configure_arguments() {
    local library="$1"
    local prefix="$2"
    local compiler="$3"
    local optimization="$4"
    local -n result="$5"
    local dependency_prefix

    result=("--prefix=$prefix")
    case "$library" in
        glib-1.2.*|freetype-1.3.*|gtk-1.2.*)
            # config.guess shipped by these releases predates x86_64 and
            # cannot identify a modern Linux host.
            result+=(
                --build=x86_64-pc-linux-gnu
                --host=x86_64-pc-linux-gnu
                --enable-static
                --disable-shared
            )
            ;;
        acl|acl-*|attr|attr-*|libxcb|libxcb-*)
            result+=(--enable-static --disable-shared)
            ;;
        file-*)
            result+=(
                --enable-static
                --disable-shared
                --enable-zlib
                --disable-bzlib
                --disable-xzlib
                --disable-lzlib
            )
            ;;
        gettext|gettext-*)
            result+=(--enable-static --disable-shared)
            ;;
        glibc|glibc-*)
            result+=(--disable-werror)
            ;;
        gtk-1.2.*)
            printf 'glib-1.2.10-20'
            ;;
        gtk|gtk-*)
            result+=(
                --enable-static
                --disable-shared
                --disable-modules
                --with-included-immodules=yes
            )
            ;;
        libthai|libthai-*)
            result+=(--enable-static --disable-shared --disable-dict)
            ;;
        util-linux|util-linux-*)
            result+=(
                --enable-static
                --disable-shared
                --disable-wall
                --disable-use-tty-group
            )
            ;;
        libiconv|libiconv-*)
            result+=(--enable-static --disable-shared)
            ;;
        gmp|gmp-*)
            result+=(--enable-static --disable-shared)
            ;;
        mpfr|mpfr-*)
            dependency_prefix="$OUTPUT_DIR/$compiler/$(
                gmp_dependency_for_mpfr "$library" "$compiler" "$optimization"
            )/$optimization/install"
            result+=(--enable-static --disable-shared "--with-gmp=$dependency_prefix")
            ;;
        ncurses|ncurses-*)
            result+=(
                --without-shared
                --with-normal
                --without-debug
                --without-ada
                --without-cxx-binding
                --with-termlib
                --enable-pc-files
                "--with-pkg-config-libdir=$prefix/lib/pkgconfig"
            )
            ;;
        pcre2|pcre2-*)
            result+=(--enable-static --disable-shared --enable-pcre2-8)
            ;;
        readline|readline-*)
            result+=(--enable-static --disable-shared --with-curses)
            ;;
        zlib|zlib-*)
            result+=(--static)
            ;;
        *)
            result+=(--enable-static --disable-shared)
            ;;
    esac
}

run_logged() {
    local log_file="$1"
    shift

    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then
        "$@" 2>&1 | tee -a "$log_file"
    fi
}

need_build_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'comando richiesto non trovato: %s\n' "$1" >&2
        return 1
    }
}

join_by_colon() {
    local IFS=:
    printf '%s' "$*"
}

dependency_flags() {
    local compiler="$1"
    local optimization="$2"
    local current_library="$3"
    local dependencies="$4"
    local include_dirs=()
    local lib_dirs=()
    local pkg_dirs=()
    local install_dir dependency
    local -n cppflags_ref="$5"
    local -n ldflags_ref="$6"
    local -n pkg_config_ref="$7"

    cppflags_ref=""
    pkg_config_ref=""
    if (( STATIC_EXECUTABLES )); then
        ldflags_ref="-static"
    else
        ldflags_ref=""
    fi
    [[ -d "$OUTPUT_DIR/$compiler" ]] || return 0
    [[ -n "$dependencies" ]] || return 0

    # Keep the declared dependency order: for static linking and for archives
    # with overlapping names (libidn2 ships a reduced libunistring.a), the
    # order of -L entries is semantically significant.
    for dependency in $dependencies; do
        [[ "$dependency" != "$current_library" ]] || continue
        install_dir="$OUTPUT_DIR/$compiler/$dependency/$optimization/install"
        [[ -d "$install_dir/include" ]] && include_dirs+=("-I$install_dir/include")
        [[ -d "$install_dir/lib" ]] && lib_dirs+=("-L$install_dir/lib")
        [[ -d "$install_dir/lib/pkgconfig" ]] && pkg_dirs+=("$install_dir/lib/pkgconfig")
    done

    cppflags_ref="$(printf '%s ' "${include_dirs[@]}")"
    if (( STATIC_EXECUTABLES )); then
        lib_dirs+=("-static")
    fi
    ldflags_ref="$(printf '%s ' "${lib_dirs[@]}")"
    pkg_config_ref="$(join_by_colon "${pkg_dirs[@]}")"
}

dependency_libraries_for() {
    local library="$1"
    local compiler="${2:-}"
    local optimization="${3:-}"

    case "$library" in
        acl-2.0.8-*)
            printf 'attr-2.0.7-1'
            ;;
        acl-2.2.53-*)
            printf 'attr-2.4.48-6'
            ;;
        acl|acl-*)
            printf 'attr'
            ;;
        file-*)
            printf 'zlib-1.3.1'
            ;;
        freetype|freetype-*)
            printf 'zlib-1.3.1 bzip2 libpng brotli'
            ;;
        fontconfig)
            printf 'freetype libexpat'
            ;;
        glib|glib-*)
            printf 'libffi pcre2-10.47 zlib-1.3.1'
            ;;
        harfbuzz|harfbuzz-*)
            printf 'freetype glib'
            ;;
        gtk|gtk-*)
            # GTK 2's configure tests execute host-linked probes.  Mixing
            # historical GLib/Pango headers with the container's runtime
            # libraries makes those probes report a false version mismatch.
            printf ''
            ;;
        libbsd|libbsd-*)
            printf 'libmd-1.1.0'
            ;;
        libpng|libpng-*)
            printf 'zlib-1.3.1'
            ;;
        libidn2|libidn2-*)
            printf 'libunistring-1.4.2'
            ;;
        libpsl|libpsl-*)
            # Search the complete GNU libunistring before libidn2's private,
            # reduced compatibility archive, which does not export every
            # symbol required by libpsl (for example u8_tolower).
            printf 'libunistring-1.4.2 libidn2-2.3.8'
            ;;
        mpfr|mpfr-*)
            gmp_dependency_for_mpfr "$library" "$compiler" "$optimization"
            ;;
        selinux|selinux-*)
            printf 'pcre2-10.47'
            ;;
        pango|pango-*)
            printf 'fontconfig freetype fribidi glib harfbuzz'
            ;;
        readline|readline-*)
            printf 'ncurses-6.5'
            ;;
    esac
}

has_static_archive() {
    local install_dir="$1"
    find "$install_dir/lib" "$install_dir/lib64" -maxdepth 1 -type f -name '*.a' \
        -print -quit 2>/dev/null | grep -q .
}

collect_build_archives() {
    local install_dir="$1"
    local variant_dir="${install_dir%/install}"
    local archive

    mkdir -p -- "$install_dir/lib"
    while IFS= read -r archive; do
        cp -f -- "$archive" "$install_dir/lib/$(basename -- "$archive")"
    done < <(
        find "$variant_dir/build" "$variant_dir/source" \
            -type f -name '*.a' -print 2>/dev/null | sort -u
    )
}

variant_is_complete() {
    local variant_dir="$1"
    local install_dir="$variant_dir/install"
    local library_name="${variant_dir%/*}"
    library_name="${library_name##*/}"

    [[ -f "$variant_dir/build-info.txt" ]] || return 1
    has_static_archive "$install_dir" || return 1
    # These public archives are required by the ELF linker but were not part
    # of the original 164-name LIVA inventory.  A previously pruned variant
    # that retained only an auxiliary archive must therefore be rebuilt.
    case "$library_name" in
        libiconv|libiconv-*)
            [[ -f "$install_dir/lib/libiconv.a" &&
               -f "$install_dir/lib/libcharset.a" ]] || return 1
            ;;
        libidn2|libidn2-*)
            [[ -f "$install_dir/lib/libidn2.a" ]] || return 1
            ;;
        libpsl|libpsl-*)
            [[ -f "$install_dir/lib/libpsl.a" ]] || return 1
            ;;
        libunistring|libunistring-*)
            [[ -f "$install_dir/lib/libunistring.a" ]] || return 1
            ;;
        zstd|zstd-*)
            [[ -f "$install_dir/lib/libzstd.a" ]] || return 1
            ;;
    esac
    if (( STATIC_EXECUTABLES )); then
        grep -qx 'static_executables=yes' "$variant_dir/build-info.txt" \
            2>/dev/null || return 1
        verify_static_executables "$install_dir" || return 1
    fi
}

verify_static_executables() {
    local install_dir="$1"
    local executable
    local elf_count=0
    local dynamic_count=0

    (( STATIC_EXECUTABLES && INSTALL )) || return 0
    command -v readelf >/dev/null 2>&1 || {
        printf 'readelf è richiesto da --static-executables\n' >&2
        return 1
    }

    while IFS= read -r -d '' executable; do
        readelf -h "$executable" >/dev/null 2>&1 || continue
        ((elf_count += 1))
        if readelf -l "$executable" 2>/dev/null | grep -q 'INTERP'; then
            printf 'ELF dinamico inatteso: %s\n' "$executable" >&2
            ((dynamic_count += 1))
        fi
    done < <(
        find "$install_dir/bin" "$install_dir/sbin" "$install_dir/libexec" \
            -type f -perm /111 -print0 2>/dev/null
    )

    if (( dynamic_count > 0 )); then
        printf 'la variante contiene %d ELF dinamici su %d verificati\n' \
            "$dynamic_count" "$elf_count" >&2
        return 1
    fi
    if (( elf_count > 0 )); then
        printf 'STATIC CHECK: %d ELF verificati in %s\n' \
            "$elf_count" "$install_dir"
    fi
}

copy_lib64_archives() {
    local install_dir="$1"

    [[ -d "$install_dir/lib64" ]] || return 0
    mkdir -p -- "$install_dir/lib"
    find "$install_dir/lib64" -maxdepth 1 -type f -name '*.a' -exec \
        cp -f -- {} "$install_dir/lib/" \;
    if [[ -d "$install_dir/lib64/pkgconfig" ]]; then
        mkdir -p -- "$install_dir/lib/pkgconfig"
        find "$install_dir/lib64/pkgconfig" -maxdepth 1 -type f -exec \
            cp -f -- {} "$install_dir/lib/pkgconfig/" \;
    fi
}

skip_nonbuildable_source() {
    local library="$1"

    case "$library" in
        libjpeg8-empty|zlib)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ensure_static_archive() {
    local library="$1"
    local install_dir="$2"

    copy_lib64_archives "$install_dir"
    # The historical corpus contains both installed public archives and
    # internal/test archives (for example OpenSSL's libtestutil.a).  Preserve
    # every archive produced by the build under the variant's install/lib.
    collect_build_archives "$install_dir"
    has_static_archive "$install_dir" || {
        printf 'nessun archivio statico prodotto per %s in %s\n' \
            "$library" "$install_dir" >&2
        return 1
    }
}

write_build_info() {
    local library="$1"
    local optimization="$2"
    local variant_dir="$3"
    local source_path="$4"
    local build_dir="$5"
    local install_dir="$6"

    {
        printf 'library=%s\n' "$library"
        printf 'optimization=-%s\n' "$optimization"
        printf 'cc=%s\n' "$CC"
        printf 'cxx=%s\n' "$CXX"
        printf 'jobs=%s\n' "$JOBS"
        printf 'source=%s\n' "$source_path"
        printf 'build=%s\n' "$build_dir"
        printf 'install=%s\n' "$install_dir"
        if (( STATIC_EXECUTABLES )); then
            printf 'static_executables=yes\n'
        else
            printf 'static_executables=no\n'
        fi
    } > "$variant_dir/build-info.txt"
}

source_root_for_library() {
    local source_path="$1"

    # Zstandard's supported CMake project lives below build/cmake.  Building
    # the repository root with its generic Makefile also attempts a shared
    # library, which is unnecessary for this static-only dataset.
    if [[ -f "$source_path/build/cmake/CMakeLists.txt" &&
          -f "$source_path/lib/zstd.h" ]]; then
        printf '%s/build/cmake' "$source_path"
        return
    fi

    if [[ -f "$source_path/expat/CMakeLists.txt" ||
          -f "$source_path/expat/configure.ac" ||
          -f "$source_path/expat/configure" ]]; then
        printf '%s/expat' "$source_path"
        return
    fi

    if [[ -f "$source_path/libz/configure" ||
          -f "$source_path/libz/CMakeLists.txt" ||
          -f "$source_path/libz/meson.build" ]]; then
        printf '%s/libz' "$source_path"
        return
    fi

    printf '%s' "$source_path"
}

build_system_for_source() {
    local library="$1"
    local source_root="$2"

    case "$library" in
        bzip2|bzip2-*)
            printf 'bzip2'
            return
            ;;
        brotli|brotli-*)
            printf 'cmake'
            return
            ;;
        glibc|glibc-*)
            printf 'glibc'
            return
            ;;
        ncurses|ncurses-*)
            printf 'ncurses'
            return
            ;;
        openssl|openssl-*)
            printf 'openssl'
            return
            ;;
        selinux|selinux-*)
            printf 'selinux'
            return
            ;;
        tcp-wrappers|tcp-wrappers-*)
            printf 'tcp-wrappers'
            return
            ;;
        xz|xz-*)
            printf 'xz'
            return
            ;;
    esac

    case "$library" in
        freetype|freetype-*)
            if [[ -f "$source_root/meson.build" ]]; then
                printf 'meson'
                return
            fi
            if [[ -f "$source_root/CMakeLists.txt" ]]; then
                printf 'cmake'
                return
            fi
            ;;
    esac

    if [[ -x "$source_root/configure" ]]; then
        printf 'configure'
    elif [[ -f "$source_root/meson.build" ]]; then
        printf 'meson'
    elif [[ -f "$source_root/CMakeLists.txt" ]]; then
        printf 'cmake'
    elif [[ -x "$source_root/autogen.sh" || -x "$source_root/autogen" ]]; then
        printf 'autogen'
    elif [[ -f "$source_root/configure.ac" ||
            -f "$source_root/configure.in" ||
            -f "$source_root/Makefile.am" ]]; then
        printf 'autoreconf'
    elif [[ -f "$source_root/Makefile" ]]; then
        printf 'make'
    else
        printf 'unknown'
    fi
}

prepare_configure_source() {
    local build_system="$1"
    local source_root="$2"
    local log_file="$3"

    case "$build_system" in
        configure)
            return 0
            ;;
        autogen)
            (
                cd -- "$source_root"
                if [[ -x ./autogen.sh ]]; then
                    run_logged "$log_file" env NOCONFIGURE=1 ./autogen.sh
                else
                    run_logged "$log_file" env NOCONFIGURE=1 ./autogen
                fi
            )
            ;;
        autoreconf)
            need_build_cmd autoreconf || return 1
            (
                cd -- "$source_root"
                run_logged "$log_file" autoreconf -fi
            )
            ;;
    esac

    [[ -x "$source_root/configure" ]] || {
        printf 'configure non generato in %s\n' "$source_root" >&2
        return 1
    }
}

apply_historical_compatibility_fixes() {
    local library="$1"
    local source_root="$2"
    local generator

    case "$library" in
        attr-2.4.48-*|acl-2.2.53-*)
            # Prevent Automake from trying to invoke the release-specific
            # aclocal-1.15 binary.  The distributed generated files are valid.
            touch -- "$source_root/aclocal.m4" "$source_root/configure"
            find "$source_root" -type f -name Makefile.in -exec touch -- {} +
            ;;
        glib-1.2.*|freetype-1.3.*)
            # Refresh config.guess/config.sub only in the private copy: the
            # originals predate the x86_64 triplet required by old ltconfig.
            while IFS= read -r config_helper; do
                cp -- "/usr/share/misc/$(basename -- "$config_helper")" \
                    "$config_helper"
            done < <(
                find "$source_root" -type f \
                    \( -name config.guess -o -name config.sub \)
            )
            if [[ "$library" == glib-1.2.* ]]; then
                patch --batch --forward -d "$source_root" -p1 \
                    < "$DATASET_DIR/patches/glib-1.2.10-modern-pretty-function.patch"
            fi
            ;;
        libcap-1.10)
            patch --batch --forward -d "$source_root" -p1 \
                < "$DATASET_DIR/patches/libcap-1.10-modern-syscalls.patch"
            ;;
        libsodium-0.7.0)
            patch --batch --forward -d "$source_root" -p1 \
                < "$DATASET_DIR/patches/libsodium-0.7.0-modern-alignment.patch"
            ;;
        libxcrypt-3.0-*)
            patch --batch --forward -d "$source_root" -p1 \
                < "$DATASET_DIR/patches/libxcrypt-3.0-modern-libc-lock.patch"
            ;;
        libpsl-0.13.0|libpsl-0.20.2)
            # Ubuntu 24.04 intentionally has no /usr/bin/python alias.  Both
            # bundled generators are compatible with Python 3.
            for generator in make_dafsa.py psl-make-dafsa; do
                [[ ! -f "$source_root/src/$generator" ]] || \
                    sed -i '1s|python$|python3|' \
                        "$source_root/src/$generator"
            done
            ;;
        graphite2-0.9.4.dfsg-4)
            # Its tests require the obsolete SIL Graphite and ICU Layout APIs;
            # neither is part of the static library being collected.
            sed -i \
                -e 's/^add_subdirectory(gr2fonttest)/# disabled for library-only build/' \
                -e 's/^add_subdirectory(tests)/# disabled for library-only build/' \
                -e 's/^add_subdirectory(doc)/# disabled for library-only build/' \
                "$source_root/CMakeLists.txt"
            sed -i 's/add_library(graphite2 SHARED/add_library(graphite2 STATIC/' \
                "$source_root/src/CMakeLists.txt"
            sed -i 's/^[[:space:]]*nolib_test(stdc++/# disabled nolib test:/' \
                "$source_root/src/CMakeLists.txt"
            sed -i 's/[[:space:]]-nostdlibs//g' \
                "$source_root/src/CMakeLists.txt"
            ;;
    esac
}

build_bzip2_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local flags="-$optimization -g -fPIC"
    local sources=(blocksort.c bzlib.c compress.c crctable.c decompress.c huffman.c randtable.c)
    local objects=()
    local source object

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    mkdir -p -- "$install_dir/include" "$install_dir/lib" "$install_dir/lib/pkgconfig"

    # bzip2 1.0.x uses the classic Makefile layout.  The current 1.1 source
    # uses bz_version.h.in and is handled by the recipe below.
    if [[ ! -f "$source_path/bz_version.h.in" ]]; then
        rm -rf -- "$build_dir"
        mkdir -p -- "$build_dir"
        cp -a -- "$source_path/." "$build_dir/"
        run_logged "$log_file" make -C "$build_dir" -j "$JOBS" \
            CC="$CC" "CFLAGS=$flags" libbz2.a || return $?
        cp -- "$build_dir/libbz2.a" "$install_dir/lib/" || return $?
        cp -- "$build_dir/bzlib.h" "$install_dir/include/" || return $?
        ensure_static_archive "$library" "$install_dir" || return $?
        write_build_info "$library" "$optimization" "$variant_dir" \
            "$source_path" "$build_dir" "$install_dir"
        return 0
    fi

    sed 's/@BZ_VERSION@/1.1.0/' "$source_path/bz_version.h.in" > "$build_dir/bz_version.h"

    for source in "${sources[@]}"; do
        object="$build_dir/${source%.c}.o"
        objects+=("$object")
        run_logged "$log_file" "$CC" $flags -I"$source_path" -I"$build_dir" \
            -c "$source_path/$source" -o "$object" || return $?
    done
    run_logged "$log_file" ar cr "$install_dir/lib/libbz2.a" "${objects[@]}" ||
        return $?
    run_logged "$log_file" ranlib "$install_dir/lib/libbz2.a" || return $?
    cp -- "$source_path/bzlib.h" "$install_dir/include/" || return $?
    cp -- "$build_dir/bz_version.h" "$install_dir/include/" || return $?
    if (( STATIC_EXECUTABLES )); then
        mkdir -p -- "$install_dir/bin"
        run_logged "$log_file" "$CC" $flags -static \
            -DBZ_UNIX=1 -DBZ_LCCWIN32=0 \
            -I"$source_path" -I"$build_dir" \
            "$source_path/bzip2.c" "$install_dir/lib/libbz2.a" \
            -o "$install_dir/bin/bzip2" || return $?
        run_logged "$log_file" "$CC" $flags -static \
            -DBZ_UNIX=1 -DBZ_LCCWIN32=0 \
            "$source_path/bzip2recover.c" \
            -o "$install_dir/bin/bzip2recover" || return $?
        ln -sf -- bzip2 "$install_dir/bin/bunzip2" || return $?
        ln -sf -- bzip2 "$install_dir/bin/bzcat" || return $?
    fi
    cat > "$install_dir/lib/pkgconfig/bzip2.pc" <<EOF
prefix=$install_dir
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: bzip2
Description: bzip2 compression library
Version: 1.1.0
Libs: -L\${libdir} -lbz2
Cflags: -I\${includedir}
EOF
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_ncurses_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local flavor flavor_build
    local -a flavor_args
    local prepared_source="$variant_dir/source"

    need_build_cmd rsync || return 1
    rm -rf -- "$prepared_source"
    mkdir -p -- "$prepared_source"
    rsync -a --exclude=.git/ "$source_path/" "$prepared_source/" || return $?
    source_path="$prepared_source"
    if [[ "$library" == ncurses-5.9 ]]; then
        # GCC 5+ expands mouse_trafo while MKlib_gen.sh preprocesses the
        # callable wrappers, yielding an invalid function declaration.
        patch --batch --forward -d "$source_path" -p1 \
            < "$DATASET_DIR/patches/ncurses-5.9-mouse-trafo.patch"
    fi

    printf '\nBUILD %-18s %-5s (%s, narrow+wide)\n' \
        "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir"

    for flavor in narrow wide; do
        flavor_build="$build_dir/$flavor"
        mkdir -p -- "$flavor_build"
        flavor_args=()
        [[ "$flavor" == wide ]] && flavor_args+=(--enable-widec)
        (
            cd -- "$flavor_build"
            run_logged "$log_file" env CC="$CC" CXX="$CXX" \
                "CFLAGS=-$optimization -std=gnu17" \
                "CXXFLAGS=-$optimization -std=gnu++17" \
                "$source_path/configure" \
                "--prefix=$install_dir" \
                --without-shared --with-normal --without-debug \
                --without-ada --without-cxx-binding --with-termlib \
                --disable-macros \
                "${flavor_args[@]}" || exit $?
            run_logged "$log_file" make -j "$JOBS" || exit $?
            run_logged "$log_file" make install || exit $?
        ) || return $?
    done

    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_tcp_wrappers_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local cflags="-$optimization -std=gnu17"

    # Clang 18 promotes implicit declarations in this 1997 K&R-style code to
    # errors.  Keep the historical source unchanged and retain them as
    # warnings, matching the behavior of the GCC builds.
    if [[ "$CC" == clang* ]]; then
        cflags+=" -Wno-error=implicit-function-declaration"
    fi

    printf '\nBUILD %-28s %-5s (%s, historical Makefile)\n' \
        "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir/lib" "$install_dir/include"
    cp -a -- "$source_path/." "$build_dir/"

    # tcp_wrappers has no configure or install target.  Reproduce the Debian
    # Linux settings but request only the static archive, avoiding its legacy
    # daemon and shared-library link steps.
    run_logged "$log_file" make -C "$build_dir" \
        "CC=$CC" "COPTS=$cflags" \
        RANLIB=ranlib ARFLAGS=rv AUX_OBJ=weak_symbols.o \
        NETGROUP=-DNETGROUP TLI= VSYSLOG= BUGS= \
        'EXTRA_CFLAGS=-DSYS_ERRLIST_DEFINED -DHAVE_STRERROR -DHAVE_WEAKSYMS -DINET6=1 -Dss_family=__ss_family -Dss_len=__ss_len' \
        config-check || return $?
    run_logged "$log_file" make -C "$build_dir" -j "$JOBS" \
        "CC=$CC" "COPTS=$cflags" \
        RANLIB=ranlib ARFLAGS=rv AUX_OBJ=weak_symbols.o \
        NETGROUP=-DNETGROUP TLI= VSYSLOG= BUGS= \
        'EXTRA_CFLAGS=-DSYS_ERRLIST_DEFINED -DHAVE_STRERROR -DHAVE_WEAKSYMS -DINET6=1 -Dss_family=__ss_family -Dss_len=__ss_len' \
        libwrap.a || return $?

    cp -- "$build_dir/libwrap.a" "$install_dir/lib/" || return $?
    cp -- "$build_dir/tcpd.h" "$install_dir/include/" || return $?
    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_glibc_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local prepared_source="$variant_dir/source"

    # Keep compatibility fixes outside the downloaded revision.  glibc 2.39
    # still calls the public fortified syslog alias internally; newer upstream
    # code calls __syslog, avoiding GCC 13's always_inline diagnostic.
    need_build_cmd rsync || return 1
    rm -rf -- "$prepared_source"
    mkdir -p -- "$prepared_source"
    rsync -a --exclude=.git/ "$source_path/" "$prepared_source/" || return $?
    if [[ -f "$prepared_source/misc/syslog.c" ]]; then
        sed -i 's/^      syslog (/      __syslog (/' \
            "$prepared_source/misc/syslog.c"
    fi
    if [[ "$library" == glibc-2.17 && -f "$prepared_source/configure" ]]; then
        # Its configure whitelist predates GCC 10+ and GNU Make 4.x.  The
        # actual feature checks remain active; only the obsolete version
        # whitelist is relaxed in this private source copy.
        sed -i \
            -e 's/    4\.\[3-9\]\.\* | 4\.\[1-9\]\[0-9\]\.\* | \[5-9\]\.\* )/    * )/' \
            -e 's/    3\.79\* | 3\.\[89\]\*)/    * )/' \
            "$prepared_source/configure"
        # GCC 13 may select a 64-bit immediate for this inline asm operand;
        # x86-64 cannot encode movq imm64 directly into TLS memory.
        sed -i \
            's/: IMM_MODE ((uint64_t) cast_to_integer (value)),/: "r" ((uint64_t) cast_to_integer (value)),/' \
            "$prepared_source/nptl/sysdeps/x86_64/tls.h"
    fi
    source_path="$prepared_source"

    printf '\nBUILD %-28s %-5s (%s, libraries only)\n' \
        "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env \
            "CC=$CC" "CXX=$CXX" \
            "CFLAGS=-$optimization -D_FORTIFY_SOURCE=0" \
            "$source_path/configure" \
            "--prefix=$install_dir" --disable-werror || exit $?
        # The default `all` target also links host-side C++ support tools.
        # That fails for old releases against a newer host libstdc++.
        run_logged "$log_file" make -j "$JOBS" lib || exit $?
    ) || return $?

    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_xz_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local flags="-$optimization -g -std=gnu17"
    local ldflags=""
    local prepared_source="$variant_dir/source"
    (( STATIC_EXECUTABLES )) && ldflags="-static"

    # xz is fetched from Git and its bootstrap writes generated files next to
    # the sources.  Work on a per-variant copy so the pinned checkout remains
    # immutable and a later fetch can verify it cleanly.
    need_build_cmd rsync || return 1
    rm -rf -- "$prepared_source"
    mkdir -p -- "$prepared_source"
    rsync -a --exclude=.git/ "$source_path/" "$prepared_source/" || return $?
    source_path="$prepared_source"

    if [[ ! -x "$source_path/configure" ]]; then
        (
            cd -- "$source_path"
            run_logged "$log_file" ./autogen.sh --no-po4a
        )
    fi

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env CC="$CC" CXX="$CXX" \
            "CFLAGS=$flags" "CXXFLAGS=$flags" "LDFLAGS=$ldflags" \
            "$source_path/configure" \
            "--prefix=$install_dir" \
            --enable-static --disable-shared --disable-nls || exit $?
        if (( STATIC_EXECUTABLES )) && [[ -x ./libtool ]]; then
            run_logged "$log_file" make -j "$JOBS" \
                "LDFLAGS=$ldflags -all-static" || exit $?
        else
            run_logged "$log_file" make -j "$JOBS" || exit $?
        fi
        if (( INSTALL )); then
            if (( STATIC_EXECUTABLES )) && [[ -x ./libtool ]]; then
                run_logged "$log_file" make install \
                    "LDFLAGS=$ldflags -all-static" || exit $?
            else
                run_logged "$log_file" make install || exit $?
            fi
        fi
    )
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_openssl_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local flags="-$optimization -g"
    local ldflags=""
    local -a disabled_features=(no-shared no-tests no-dso no-engine)
    local build_target=build_sw
    (( STATIC_EXECUTABLES )) && ldflags="-static"

    # OpenSSL 1.1 has no `no-module` configure option; it was introduced by
    # the provider/module architecture in OpenSSL 3.
    if [[ "$library" == openssl-1.* ]]; then
        build_target=build_libs
    else
        disabled_features+=(no-module)
    fi

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    cp -a -- "$source_path/." "$build_dir/"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env CC="$CC" CXX="$CXX" \
            "LDFLAGS=$ldflags" \
            perl Configure linux-x86_64 \
            "--prefix=$install_dir" "--openssldir=$install_dir/ssl" \
            "${disabled_features[@]}" \
            "$flags" || exit $?
        run_logged "$log_file" make -j "$JOBS" "$build_target" || exit $?
        if (( INSTALL )); then
            run_logged "$log_file" make install_sw || exit $?
        fi
    )
    if [[ -d "$install_dir/lib64" ]]; then
        mkdir -p -- "$install_dir/lib"
        find "$install_dir/lib64" -maxdepth 1 -type f -name '*.a' -exec \
            cp -f -- {} "$install_dir/lib/" \;
        if [[ -d "$install_dir/lib64/pkgconfig" ]]; then
            mkdir -p -- "$install_dir/lib/pkgconfig"
            find "$install_dir/lib64/pkgconfig" -maxdepth 1 -type f -exec \
                cp -f -- {} "$install_dir/lib/pkgconfig/" \;
        fi
    fi
    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_selinux_variant() {
    local library="$1"
    local optimization="$2"
    local source_path="$3"
    local variant_dir="$4"
    local build_dir="$5"
    local install_dir="$6"
    local log_file="$7"
    local pcre_prefix="$OUTPUT_DIR/$(compiler_id)/pcre2-10.47/$optimization/install"
    local selinux_root="$build_dir/libselinux"

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir/lib" "$install_dir/include"
    cp -a -- "$source_path/." "$build_dir/"

    # The Debian libselinux source package is already rooted at src/include,
    # whereas the current SELinux monorepo contains a libselinux/ subfolder.
    [[ -d "$selinux_root/src" ]] || selinux_root="$build_dir"

    run_logged "$log_file" make -C "$selinux_root/src" -j "$JOBS" \
        CC="$CC" \
        "CPPFLAGS=-I../../libsepol/include" \
        "CFLAGS=-$optimization -std=gnu17 -Wall -Wextra -Wno-error -DSHARED" \
        "LDFLAGS=-L$pcre_prefix/lib" \
        "PCRE_CFLAGS=-DUSE_PCRE2 -DPCRE2_CODE_UNIT_WIDTH=8 -I$pcre_prefix/include" \
        "PCRE_LDLIBS=$pcre_prefix/lib/libpcre2-8.a" \
        DISABLE_SETRANS=y DISABLE_RPM=y DISABLE_X11=y \
        libselinux.a || return $?

    cp -- "$selinux_root/src/libselinux.a" "$install_dir/lib/" || return $?
    cp -a -- "$selinux_root/include/selinux" "$install_dir/include/" || return $?
    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_path" "$build_dir" "$install_dir"
}

build_configure_variant() {
    local library="$1"
    local optimization="$2"
    local compiler="$3"
    local source_root="$4"
    local variant_dir="$5"
    local build_dir="$6"
    local install_dir="$7"
    local log_file="$8"
    local build_system="$9"
    local cflags="-$optimization -std=gnu17"
    local cxxflags="-$optimization -std=gnu++17"
    local cppflags=""
    local ldflags=""
    local pkg_config_path=""
    local dependencies
    local -a configure_args
    local -a build_env

    if [[ "$(basename -- "$CC")" == clang-* ]]; then
        # Old C releases rely on declarations that were implicit in the C89
        # era.  Clang 18 diagnoses them as errors by default even though the
        # same sources remain buildable with GCC.
        cflags+=" -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-function-pointer-types -Wno-error=deprecated-non-prototype"
    fi

    if [[ "$library" == glib-1.2.* || "$library" == gtk-1.2.* ]]; then
        # GLib 1.2's public header uses GNU89 `extern inline` semantics.  With
        # the C99 semantics selected by modern GCC/Clang, every translation
        # unit emits g_bit_* and linking fails with duplicate definitions.
        # Keep the historical semantics both while building GLib and while
        # GTK's configure probes include the installed GLib header.
        cflags+=" -fgnu89-inline"
    fi

    if [[ "$library" == libxcrypt-3.0-* ]]; then
        # This release enables -Werror internally, but predates the alias and
        # alignment diagnostics emitted by GCC 13 and Clang 18.
        cflags+=" -Wno-error -Wno-missing-attributes -Wno-nonnull-compare -Wno-pointer-bool-conversion -Wno-cast-align"
    fi

    if [[ ("$library" == gtk || "$library" == gtk-*) &&
          "$(basename -- "$CC")" == clang-18 ]]; then
        # GTK 2.24 predates Clang 18's stricter treatment of callbacks whose
        # declared function-pointer type is only ABI-compatible.
        cflags+=" -Wno-error=incompatible-function-pointer-types"
    fi

    if [[ "$library" == libxcb-1.10-1 ]]; then
        local proto_source="$SOURCE_DIR/xcb-proto-1.10"
        local proto_build="$variant_dir/xcb-proto-build"
        local proto_prefix="$variant_dir/xcb-proto-install"
        local proto_python
        [[ -x "$proto_source/configure" ]] || {
            printf 'dipendenza sorgente mancante: %s\n' "$proto_source" >&2
            return 1
        }
        rm -rf -- "$proto_build" "$proto_prefix"
        mkdir -p -- "$proto_build"
        (
            cd -- "$proto_build"
            run_logged "$log_file" "$proto_source/configure" \
                "--prefix=$proto_prefix" || exit $?
        ) || return $?
        proto_python="$proto_prefix/local/lib/python3.12/dist-packages"
        mkdir -p -- "$proto_prefix/share/xcb" "$proto_python/xcbgen" \
            "$proto_prefix/lib/pkgconfig"
        cp -- "$proto_source/src/"*.xml "$proto_source/src/xcb.xsd" \
            "$proto_prefix/share/xcb/" || return $?
        cp -- "$proto_source/xcbgen/"*.py "$proto_python/xcbgen/" || return $?
        cp -- "$proto_build/xcb-proto.pc" "$proto_prefix/lib/pkgconfig/" || return $?
        pkg_config_path="$proto_prefix/lib/pkgconfig"
    fi

    prepare_configure_source "$build_system" "$source_root" "$log_file" || return $?
    configure_arguments "$library" "$install_dir" "$compiler" "$optimization" configure_args
    dependencies="$(dependency_libraries_for "$library" "$compiler" "$optimization")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    if [[ "$library" == libxcb-1.10-1 ]]; then
        pkg_config_path="$proto_prefix/lib/pkgconfig${pkg_config_path:+:$pkg_config_path}"
    fi

    build_env=(
        "CC=$CC"
        "CXX=$CXX"
        "CFLAGS=$cflags"
        "CXXFLAGS=$cxxflags"
        "CPPFLAGS=$cppflags"
        "LDFLAGS=$ldflags"
        "PKG_CONFIG_PATH=$pkg_config_path"
    )
    if [[ "$library" == libxcb-0.9.92-* ]]; then
        # The Debian source archive already contains all generated protocol C
        # files.  Configure only needs a successful no-op XSLT command.
        build_env+=(
            "XSLTPROC=/usr/bin/true"
            "ac_cv_path_XSLTPROC=/usr/bin/true"
        )
    fi
    if [[ "$library" == gtk-1.2.* ]]; then
        local glib_variant="$OUTPUT_DIR/$compiler/glib-1.2.10-20/$optimization"
        local glib_source="$OUTPUT_DIR/$compiler/glib-1.2.10-20/$optimization/source"
        mkdir -p -- "$glib_variant/install/bin" \
            "$glib_variant/install/include/glib-1.2" \
            "$glib_variant/install/lib/glib/include"
        cp -f -- "$glib_source/glib-config" "$glib_variant/install/bin/" \
            || return $?
        cp -f -- "$glib_source/glib.h" \
            "$glib_variant/install/include/glib-1.2/" || return $?
        cp -f -- "$glib_source/glibconfig.h" \
            "$glib_variant/install/lib/glib/include/" || return $?
        build_env+=(
            "GLIB_CONFIG=$OUTPUT_DIR/$compiler/glib-1.2.10-20/$optimization/install/bin/glib-config"
        )
    fi
    if [[ "$library" == attr-2.0.* || "$library" == acl-2.0.* ]]; then
        # Their pre-AC_PREFIX configure scripts use uppercase PREFIX variables
        # and otherwise silently install below /usr.
        build_env+=("PREFIX=$install_dir" "ROOT_PREFIX=$install_dir")
    fi
    if [[ "$library" == libxcb-1.10-1 ]]; then
        build_env+=("PYTHONPATH=$proto_python")
    fi

    printf '\nBUILD %-28s %-5s (%s, %s)\n' \
        "$library" "$optimization" "$CC" "$build_system"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env "${build_env[@]}" \
            "$source_root/configure" "${configure_args[@]}" || exit $?
        if [[ ("$library" == attr-2.0.* || "$library" == acl-2.0.*) &&
              -f include/builddefs ]]; then
            # These releases discover the host's libtool executable instead
            # of generating a project wrapper.  Modern libtool needs an
            # explicit language tag for compilation.
            sed -i 's|^LIBTOOL[[:space:]]*=.*|LIBTOOL = /usr/bin/libtool --tag=CC|' \
                include/builddefs
            printf '\nLCFLAGS += -%s %s\n' "$optimization" "$cppflags" \
                >> include/builddefs
        fi
        if [[ "$library" == libxcb || "$library" == libxcb-* ]]; then
            # The static archives live entirely in src/.  Old Debian releases
            # otherwise descend into tests and try to regenerate them with
            # their exact historical Automake version.
            if [[ "$library" == libxcb-0.9.92-* ]]; then
                # Its single Make rule recreates every XML symlink.  Running
                # that rule in parallel races with itself and corrupts links.
                run_logged "$log_file" make -C src -j 1 || exit $?
            else
                run_logged "$log_file" make -C src -j "$JOBS" || exit $?
            fi
        elif [[ "$library" == gtk || "$library" == gtk-* ]]; then
            # The perf/example executables do not support the static-only
            # configuration; build the actual GTK library subtrees directly.
            run_logged "$log_file" make -C gdk -j "$JOBS" || exit $?
            if ! run_logged "$log_file" make -C gtk -j "$JOBS"; then
                find gtk -type f -name '*.a' -print -quit | grep -q . || exit 1
                printf 'WARN: GTK tools/tests failed after static archive creation\n' \
                    | tee -a "$log_file"
            fi
        elif (( STATIC_EXECUTABLES )) && [[ -x ./libtool ]]; then
            run_logged "$log_file" make -j "$JOBS" \
                "LDFLAGS=$ldflags -all-static" || exit $?
        else
            if ! run_logged "$log_file" make -j "$JOBS"; then
                # Several historical projects build their static library
                # before optional tools/tests that no longer compile on a
                # modern host.  The dataset needs archives, not those tools.
                find . -type f -name '*.a' -print -quit | grep -q . || exit 1
                printf 'WARN: utility/test build failed after static archive creation\n' \
                    | tee -a "$log_file"
            fi
        fi
        if (( INSTALL )) &&
           [[ "$library" != util-linux && "$library" != util-linux-* &&
              "$library" != libxcb && "$library" != libxcb-* &&
              "$library" != gtk && "$library" != gtk-* ]]; then
            if (( STATIC_EXECUTABLES )) && [[ -x ./libtool ]]; then
                run_logged "$log_file" make install \
                    "LDFLAGS=$ldflags -all-static" || exit $?
            else
                if ! run_logged "$log_file" make install; then
                    find . -type f -name '*.a' -print -quit | grep -q . || exit 1
                    printf 'WARN: install failed; collecting built static archives\n' \
                        | tee -a "$log_file"
                fi
            fi
        fi
    ) || return $?

    if [[ "$library" == glib-1.2.* ]]; then
        # Old GLib builds the library before legacy tests fail.  Recreate the
        # small public development layout GTK 1.2 expects from glib-config.
        mkdir -p -- "$install_dir/bin" "$install_dir/include/glib-1.2" \
            "$install_dir/lib/glib/include"
        cp -f -- "$source_root/glib-config" "$install_dir/bin/"
        cp -f -- "$source_root/glib.h" "$install_dir/include/glib-1.2/"
        cp -f -- "$source_root/glibconfig.h" \
            "$install_dir/lib/glib/include/"
    elif [[ "$library" == attr-2.0.* ]]; then
        mkdir -p -- "$install_dir/include/attr"
        cp -f -- "$source_root/include/"*.h "$install_dir/include/attr/" || return $?
    elif [[ "$library" == acl-2.0.* ]]; then
        mkdir -p -- "$install_dir/include/sys"
        cp -f -- "$source_root/include/acl.h" "$install_dir/include/sys/" || return $?
        cp -f -- "$source_root/include/"*.h "$install_dir/include/" || return $?
    fi

    if [[ "$library" == util-linux || "$library" == util-linux-* ||
          "$library" == libxcb || "$library" == libxcb-* ||
          "$library" == gtk || "$library" == gtk-* ]]; then
        # Installing util-linux also tries to chown its setuid programs.  The
        # dataset only needs archives, so collect them from the finished build.
        collect_build_archives "$build_dir" "$install_dir" || return $?
    fi
    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_root" "$build_dir" "$install_dir"
}

build_cmake_variant() {
    local library="$1"
    local optimization="$2"
    local compiler="$3"
    local source_root="$4"
    local variant_dir="$5"
    local build_dir="$6"
    local install_dir="$7"
    local log_file="$8"
    local cflags="-$optimization -std=gnu17"
    local cxxflags="-$optimization -std=gnu++17"
    local cppflags=""
    local ldflags=""
    local pkg_config_path=""
    local dependencies
    local -a cmake_args=(-DBUILD_SHARED_LIBS=OFF)

    if [[ "$library" == graphite2-0.9.4.dfsg-4 ]]; then
        local prepared_source="$variant_dir/source"
        need_build_cmd rsync || return 1
        rm -rf -- "$prepared_source"
        mkdir -p -- "$prepared_source"
        rsync -a --exclude=.git/ "$source_root/" "$prepared_source/" || return $?
        source_root="$prepared_source"
        apply_historical_compatibility_fixes "$library" "$source_root" || return $?
    fi

    case "$library" in
        brotli|brotli-*)
            cmake_args+=(
                -DBROTLI_DISABLE_TESTS=ON
                -DBROTLI_BUILD_TOOLS=OFF
            )
            ;;
        zstd|zstd-*)
            cmake_args+=(
                -DZSTD_BUILD_SHARED=OFF
                -DZSTD_BUILD_PROGRAMS=OFF
                -DZSTD_BUILD_TESTS=OFF
                -DZSTD_BUILD_CONTRIB=OFF
            )
            ;;
    esac

    need_build_cmd cmake || return 1
    dependencies="$(dependency_libraries_for "$library" "$compiler" "$optimization")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    printf '\nBUILD %-28s %-5s (%s, cmake)\n' \
        "$library" "$optimization" "$CC"
    run_logged "$log_file" env \
        "CC=$CC" "CXX=$CXX" \
        "CFLAGS=$cflags $cppflags" "CXXFLAGS=$cxxflags $cppflags" \
        "LDFLAGS=$ldflags" "PKG_CONFIG_PATH=$pkg_config_path" \
        cmake -S "$source_root" -B "$build_dir" \
            -DCMAKE_INSTALL_PREFIX="$install_dir" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_FLAGS_RELEASE= \
            -DCMAKE_CXX_FLAGS_RELEASE= \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DCMAKE_EXE_LINKER_FLAGS="$ldflags" \
            "${cmake_args[@]}" || return $?
    run_logged "$log_file" cmake --build "$build_dir" -- -j "$JOBS" || return $?
    if (( INSTALL )); then
        run_logged "$log_file" cmake --install "$build_dir" || return $?
    fi

    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_root" "$build_dir" "$install_dir"
}

build_meson_variant() {
    local library="$1"
    local optimization="$2"
    local compiler="$3"
    local source_root="$4"
    local variant_dir="$5"
    local build_dir="$6"
    local install_dir="$7"
    local log_file="$8"
    local cflags="-$optimization -std=gnu17"
    local cxxflags="-$optimization -std=gnu++17"
    local cppflags=""
    local ldflags=""
    local pkg_config_path=""
    local dependencies
    local -a meson_args=()

    case "$library" in
        at-spi2-core|at-spi2-core-*)
            meson_args+=(
                -Ddocs=false
                -Dintrospection=disabled
                -Duse_systemd=false
            )
            ;;
    esac

    need_build_cmd meson || return 1
    need_build_cmd ninja || return 1
    dependencies="$(dependency_libraries_for "$library" "$compiler" "$optimization")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    printf '\nBUILD %-28s %-5s (%s, meson)\n' \
        "$library" "$optimization" "$CC"
    run_logged "$log_file" env \
        "CC=$CC" "CXX=$CXX" \
        "CFLAGS=$cflags $cppflags" "CXXFLAGS=$cxxflags $cppflags" \
        "LDFLAGS=$ldflags" "PKG_CONFIG_PATH=$pkg_config_path" \
        meson setup "$build_dir" "$source_root" \
            --prefix "$install_dir" \
            --default-library=static \
            --buildtype=plain \
            "${meson_args[@]}" || return $?
    run_logged "$log_file" ninja -C "$build_dir" -j "$JOBS" || return $?
    if (( INSTALL )); then
        run_logged "$log_file" ninja -C "$build_dir" install || return $?
    fi

    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_root" "$build_dir" "$install_dir"
}

build_make_variant() {
    local library="$1"
    local optimization="$2"
    local compiler="$3"
    local source_root="$4"
    local variant_dir="$5"
    local build_dir="$6"
    local install_dir="$7"
    local log_file="$8"
    local cflags="-$optimization -std=gnu17"
    local cxxflags="-$optimization -std=gnu++17"
    local cppflags=""
    local ldflags=""
    local pkg_config_path=""
    local dependencies

    if [[ "$(basename -- "$CC")" == clang-* ]]; then
        cflags+=" -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-function-pointer-types -Wno-error=deprecated-non-prototype"
    fi

    dependencies="$(dependency_libraries_for "$library" "$compiler" "$optimization")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    printf '\nBUILD %-28s %-5s (%s, make)\n' \
        "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir"
    cp -a -- "$source_root/." "$build_dir/"
    apply_historical_compatibility_fixes "$library" "$build_dir" || return $?

    if [[ "$library" == libcap-1.10 ]]; then
        local -a cap_objects=(
            cap_alloc.o cap_proc.o cap_extint.o cap_flag.o cap_text.o cap_sys.o
        )
        mkdir -p -- "$install_dir/lib" "$install_dir/include/sys"
        run_logged "$log_file" make -C "$build_dir/libcap" -j "$JOBS" \
            "CC=$CC" \
            "CFLAGS=$cflags $cppflags -I$build_dir/libcap/include" \
            "${cap_objects[@]}" || return $?
        run_logged "$log_file" ar cr "$install_dir/lib/libcap.a" \
            "${cap_objects[@]/#/$build_dir/libcap/}" || return $?
        run_logged "$log_file" ranlib "$install_dir/lib/libcap.a" || return $?
        cp -- "$build_dir/libcap/include/sys/capability.h" \
            "$install_dir/include/sys/" || return $?
        ensure_static_archive "$library" "$install_dir" || return $?
        write_build_info "$library" "$optimization" "$variant_dir" \
            "$source_root" "$build_dir" "$install_dir"
        return 0
    fi

    (
        cd -- "$build_dir"
        run_logged "$log_file" env \
            "CC=$CC" "CXX=$CXX" \
            "CFLAGS=$cflags $cppflags" \
            "CXXFLAGS=$cxxflags $cppflags" \
            "PKG_CONFIG_PATH=$pkg_config_path" \
            make -j "$JOBS" "LDFLAGS=$ldflags" || exit $?
        if (( INSTALL )); then
            run_logged "$log_file" env \
                "PREFIX=$install_dir" "prefix=$install_dir" \
                make install "LDFLAGS=$ldflags" || exit $?
        fi
    ) || return $?

    ensure_static_archive "$library" "$install_dir" || return $?
    write_build_info "$library" "$optimization" "$variant_dir" \
        "$source_root" "$build_dir" "$install_dir"
}

build_variant() {
    local library="$1"
    local optimization="$2"
    local compiler="$3"
    local source_path="$SOURCE_DIR/$library"
    local source_root
    local build_system
    local variant_dir="$OUTPUT_DIR/$compiler/$library/$optimization"
    local build_dir="$variant_dir/build"
    local install_dir="$variant_dir/install"
    local log_file="$variant_dir/build.log"
    local build_status

    if [[ ("$library" == glibc || "$library" == glibc-*) && "$optimization" == O0 ]]; then
        printf 'SKIP  %-28s %-5s glibc richiede almeno -O2 nella pipeline\n' \
            "$library" "$optimization"
        return 0
    fi

    source_root="$(source_root_for_library "$source_path")"
    build_system="$(build_system_for_source "$library" "$source_root")"

    if skip_nonbuildable_source "$library"; then
        printf 'SKIP  %-28s %-5s sorgente non buildabile/placeholder\n' \
            "$library" "$optimization"
        return 0
    fi

    if (( CLEAN && ! DRY_RUN )); then
        rm -rf -- "$variant_dir"
    fi

    if (( SKIP_EXISTING )) && variant_is_complete "$variant_dir"; then
        printf 'SKIP  %-28s %-5s variante già presente\n' \
            "$library" "$optimization"
        return 0
    fi

    # A previous failed attempt is not a reusable build tree.  Reconfiguring
    # it causes misleading, optimization-dependent errors in old Autotools
    # projects, so recreate only that incomplete variant.
    if [[ -d "$variant_dir" ]] && ! variant_is_complete "$variant_dir"; then
        rm -rf -- "$variant_dir"
    fi

    if (( DRY_RUN )); then
        printf 'DRY   %-28s %-5s build_system=%s source=%s\n' \
            "$library" "$optimization" "$build_system" "$source_root"
        return 0
    fi

    mkdir -p -- "$build_dir" "$install_dir"
    : > "$log_file"

    if [[ "$build_system" == configure || "$build_system" == autoreconf ||
          "$build_system" == autogen ]]; then
        # Automake may refresh configure/aclocal files during `make`.  Keep
        # those generated changes in the external build tree, never in the
        # revision-pinned source checkout.
        local prepared_source="$variant_dir/source"
        need_build_cmd rsync || return 1
        rm -rf -- "$prepared_source"
        mkdir -p -- "$prepared_source"
        rsync -a --exclude=.git/ "$source_root/" "$prepared_source/" || return $?
        source_root="$prepared_source"
        if [[ -f "$source_root/config.status" ]]; then
            [[ ! -f "$source_root/Makefile" ]] || \
                make -C "$source_root" distclean >/dev/null 2>&1 || true
            rm -f -- "$source_root/config.status" "$source_root/config.log" \
                "$source_root/config.cache"
        fi
        apply_historical_compatibility_fixes "$library" "$source_root" || return $?
        if [[ "$library" == libxcrypt || "$library" == libxcrypt-* ]]; then
            # Perl 5.38 moved smartmatch/when to the general deprecation
            # category; these releases make every warning fatal.
            while IFS= read -r perl_source; do
                sed -i "/use warnings FATAL/a no warnings 'deprecated';" "$perl_source"
            done < <(grep -rl 'use warnings FATAL' "$source_root")
            build_system=configure
        elif [[ "$library" == libxcb || "$library" == libxcb-* ]]; then
            # xcbgen 1.16 no longer pre-populates namecount for enum names in
            # the way libxcb 1.14/1.15's generator expects.  Make the lookup
            # tolerant in this per-variant copy, leaving downloaded sources
            # untouched and preserving the generated C API.
            if [[ -f "$source_root/src/c_client.py" ]]; then
                sed -i 's/namecount\[tname\]/namecount.get(tname, 0)/g' \
                    "$source_root/src/c_client.py"
            fi
            build_system="$(build_system_for_source "$library" "$source_root")"
        else
            build_system="$(build_system_for_source "$library" "$source_root")"
        fi
        # Configure in the private source copy.  A number of releases from
        # the 1990s/2000s do not implement VPATH builds and read VERSION,
        # Makefile.in or headers relative to the working directory.
        build_dir="$source_root"
    fi

    case "$build_system" in
        bzip2)
            build_bzip2_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        glibc)
            build_glibc_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        ncurses)
            build_ncurses_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        openssl)
            build_openssl_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        selinux)
            build_selinux_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        tcp-wrappers)
            build_tcp_wrappers_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        xz)
            build_xz_variant "$library" "$optimization" "$source_path" \
                "$variant_dir" "$build_dir" "$install_dir" "$log_file"
            ;;
        configure|autogen|autoreconf)
            build_configure_variant "$library" "$optimization" "$compiler" \
                "$source_root" "$variant_dir" "$build_dir" "$install_dir" \
                "$log_file" "$build_system"
            ;;
        cmake)
            build_cmake_variant "$library" "$optimization" "$compiler" \
                "$source_root" "$variant_dir" "$build_dir" "$install_dir" \
                "$log_file"
            ;;
        meson)
            build_meson_variant "$library" "$optimization" "$compiler" \
                "$source_root" "$variant_dir" "$build_dir" "$install_dir" \
                "$log_file"
            ;;
        make)
            build_make_variant "$library" "$optimization" "$compiler" \
                "$source_root" "$variant_dir" "$build_dir" "$install_dir" \
                "$log_file"
            ;;
        *)
            printf 'nessun build system riconosciuto per %s (%s)\n' \
                "$library" "$source_root" >&2
            return 1
            ;;
    esac
    build_status=$?
    (( build_status == 0 )) || return "$build_status"
    verify_static_executables "$install_dir"
}

write_manifest() {
    local compiler="$1"
    local manifest="$OUTPUT_DIR/$compiler/manifest.tsv"
    local lib_dir variant_dir optimization library archive

    if (( DRY_RUN )); then
        printf '\nDRY manifest: %s\n' "$manifest"
        return 0
    fi

    {
        printf 'optimization\tlibrary\tarchive\n'
        while IFS= read -r lib_dir; do
            variant_dir="${lib_dir%/install/lib}"
            variant_dir="${variant_dir%/install/lib64}"
            optimization="${variant_dir##*/}"
            library="${variant_dir%/*}"
            library="${library##*/}"
            while IFS= read -r archive; do
                printf '%s\t%s\t%s\n' \
                    "$optimization" "$library" "$archive"
            done < <(find "$lib_dir" -maxdepth 1 -type f -name '*.a' -print | sort)
        done < <(
            find "$OUTPUT_DIR/$compiler" -mindepth 4 -maxdepth 4 \
                \( -path '*/install/lib' -o -path '*/install/lib64' \) \
                -type d -print 2>/dev/null | sort
        )
    } > "$manifest"
    printf '\nManifest: %s\n' "$manifest"
}

run_historical_mode() {
    shift # `historical`
    local action=all
    local historical_jobs="$JOBS"
    local use_covered=0
    local historical_dry_run=0
    local -a only=()
    local -a toolchains=(
        "gcc-11|g++-11"
        "gcc-13|g++-13"
        "clang-14|clang++-14"
        "clang-18|clang++-18"
    )

    if (($#)) && [[ "$1" =~ ^(all|fetch|build|list)$ ]]; then
        action="$1"
        shift
    fi
    while (($#)); do
        case "$1" in
            -j|--jobs)
                (($# >= 2)) || die "$1 richiede un valore"
                historical_jobs="$2"; shift 2 ;;
            --only|-l|--libraries)
                (($# >= 2)) || die "$1 richiede un valore"
                only+=("$2"); shift 2 ;;
            --use-covered)
                use_covered=1; shift ;;
            --dry-run)
                historical_dry_run=1; shift ;;
            -h|--help)
                cat <<'EOF'
Uso: build_libraries.sh historical [all|fetch|build|list] [opzioni]

Scarica e/o compila le tre versioni historical definite in
Dataset/manifests/source_manifest.json, senza costruire o modificare ELF.

  --only LIST       pacchetti separati da virgola; ripetibile
  -j, --jobs N      job paralleli
  --use-covered     usa build correnti equivalenti dove dichiarato
  --dry-run         mostra le operazioni senza scrivere
EOF
                return 0 ;;
            *)
                die "argomento historical sconosciuto: $1" ;;
        esac
    done

    [[ "$historical_jobs" =~ ^[1-9][0-9]*$ ]] ||
        die "--jobs deve essere positivo"

    local selection=all
    if ((${#only[@]})); then
        selection="$(IFS=,; printf '%s' "${only[*]}")"
    fi
    local -a fetch_args=(
        "$SCRIPT_DIR/fetch_dataset_sources.sh"
        --historical-only "$selection"
        --historical-jobs "$historical_jobs"
    )
    (( use_covered )) && fetch_args+=(--use-covered)

    if [[ "$action" == list ]]; then
        exec "${fetch_args[@]}" --list
    fi
    if [[ "$action" == all || "$action" == fetch ]]; then
        local -a download_args=("${fetch_args[@]}")
        (( historical_dry_run )) && download_args+=(--dry-run)
        "${download_args[@]}"
    fi
    [[ "$action" == all || "$action" == build ]] || return 0

    local -a name_args=("${fetch_args[@]}" --historical-build-names)
    mapfile -t historical_libraries < <("${name_args[@]}")
    ((${#historical_libraries[@]})) || die "nessuna versione historical selezionata"

    local csv
    csv="$(IFS=,; printf '%s' "${historical_libraries[*]}")"
    local -a failures=()
    local toolchain historical_cc historical_cxx
    for toolchain in "${toolchains[@]}"; do
        IFS='|' read -r historical_cc historical_cxx <<< "$toolchain"
        local -a selected_libraries=()
        local library
        for library in "${historical_libraries[@]}"; do
            [[ "$historical_cc" == clang-* && "$library" == glibc-* ]] && continue
            selected_libraries+=("$library")
        done
        csv="$(IFS=,; printf '%s' "${selected_libraries[*]}")"
        local -a build_args=(
            "$SCRIPT_DIR/build_libraries.sh"
            --libraries "$csv"
            --optimizations O0,O2,O3,Os
            --jobs "$historical_jobs"
            --cc "$historical_cc"
            --cxx "$historical_cxx"
            --output "$DATASET_DIR/builds/libraries"
            --skip-existing
            --continue-on-error
            --include-excluded
        )
        (( historical_dry_run )) && build_args+=(--dry-run)
        "${build_args[@]}" || failures+=("$historical_cc/$historical_cxx")
    done
    if ((${#failures[@]})); then
        printf 'Build historical con errori: %s\n' "${failures[*]}" >&2
        return 1
    fi
    printf 'Build historical completata; nessun ELF è stato modificato.\n'
}

if [[ "${1:-}" == historical ]]; then
    run_historical_mode "$@"
    exit $?
fi

while (($#)); do
    case "$1" in
        -l|--libraries)
            (($# >= 2)) || die "valore mancante per $1"
            split_csv "$2" LIBRARIES
            shift 2
            ;;
        --curated)
            LIBRARIES=("${CURATED_LIBRARIES[@]}")
            shift
            ;;
        -o|--optimizations)
            (($# >= 2)) || die "valore mancante per $1"
            split_csv "$2" OPTIMIZATIONS
            shift 2
            ;;
        -j|--jobs)
            (($# >= 2)) || die "valore mancante per $1"
            JOBS="$2"
            shift 2
            ;;
        --cc)
            (($# >= 2)) || die "valore mancante per $1"
            CC="$2"
            shift 2
            ;;
        --cxx)
            (($# >= 2)) || die "valore mancante per $1"
            CXX="$2"
            shift 2
            ;;
        --output)
            (($# >= 2)) || die "valore mancante per $1"
            OUTPUT_DIR="$(realpath -m -- "$2")"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --no-install)
            INSTALL=0
            shift
            ;;
        --static-executables)
            STATIC_EXECUTABLES=1
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --include-excluded)
            ALLOW_EXCLUDED=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --list-libraries)
            LIST_LIBRARIES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "opzione sconosciuta: $1"
            ;;
    esac
done

if ((${#LIBRARIES[@]} == 0)); then
    mapfile -t LIBRARIES < <(default_libraries)
fi

if (( LIST_LIBRARIES )); then
    printf '%s\n' "${LIBRARIES[@]}"
    exit 0
fi

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs deve essere un intero positivo"
command -v "$CC" >/dev/null 2>&1 || die "compilatore C non trovato: $CC"
command -v "$CXX" >/dev/null 2>&1 || die "compilatore C++ non trovato: $CXX"
command -v make >/dev/null 2>&1 || die "make non trovato"
command -v tee >/dev/null 2>&1 || die "tee non trovato"

normalize_optimizations
validate_libraries
mkdir -p -- "$OUTPUT_DIR"

COMPILER_ID="$(compiler_id)"
FAILURES=()

printf 'Sorgenti:      %s\n' "$SOURCE_DIR"
printf 'Output:        %s\n' "$OUTPUT_DIR/$COMPILER_ID"
printf 'Librerie:      %s\n' "${LIBRARIES[*]}"
printf 'Ottimizzazioni:%s\n' " ${OPTIMIZATIONS[*]}"
printf 'Job:           %s\n' "$JOBS"

for library in "${LIBRARIES[@]}"; do
    for optimization in "${OPTIMIZATIONS[@]}"; do
        if ! build_variant "$library" "$optimization" "$COMPILER_ID"; then
            FAILURES+=("$library/$optimization")
            printf 'FAIL  %-18s %-5s (log nella directory della variante)\n' \
                "$library" "$optimization" >&2
            if (( ! CONTINUE_ON_ERROR )); then
                exit 1
            fi
        fi
    done
done

write_manifest "$COMPILER_ID"

if ((${#FAILURES[@]})); then
    printf '\nBuild completata con errori: %s\n' "${FAILURES[*]}" >&2
    exit 1
fi

printf '\nBuild completata con successo.\n'
