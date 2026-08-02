#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$DATASET_DIR/sources/lib_sources"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/builds/lib_builds}"

CURATED_LIBRARIES=(
    zlib-1.3.1
    bzip2
    xz
    openssl
    openssl-3.5.0
    acl-2.3.2
    file-5.46
    glibc-2.41
    gmp-6.3.0
    libiconv-1.18
    mpfr-4.2.1
    ncurses-6.5
    pcre2-10.47
    readline-8.2
    selinux-3.7
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
    ncurses-6.3
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

discover_source_libraries() {
    [[ -d "$SOURCE_DIR" ]] || return 0
    find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        sort
}

is_excluded_library() {
    local library="$1"
    local excluded

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

configure_arguments() {
    local library="$1"
    local prefix="$2"
    local compiler="$3"
    local optimization="$4"
    local -n result="$5"
    local dependency_prefix

    result=("--prefix=$prefix")
    case "$library" in
        acl|acl-*|attr|libxcb)
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
        libiconv|libiconv-*)
            result+=(--enable-static --disable-shared)
            ;;
        gmp|gmp-*)
            result+=(--enable-static --disable-shared)
            ;;
        mpfr|mpfr-*)
            dependency_prefix="$OUTPUT_DIR/$compiler/gmp-6.3.0/$optimization/install"
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
    local install_dir library dependency
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

    while IFS= read -r install_dir; do
        library="${install_dir#"$OUTPUT_DIR/$compiler/"}"
        library="${library%%/*}"
        [[ "$library" != "$current_library" ]] || continue
        for dependency in $dependencies; do
            [[ "$library" == "$dependency" ]] || continue
            [[ -d "$install_dir/include" ]] && include_dirs+=("-I$install_dir/include")
            [[ -d "$install_dir/lib" ]] && lib_dirs+=("-L$install_dir/lib")
            [[ -d "$install_dir/lib/pkgconfig" ]] && pkg_dirs+=("$install_dir/lib/pkgconfig")
        done
    done < <(
        find "$OUTPUT_DIR/$compiler" -mindepth 3 -maxdepth 3 \
            -path "*/$optimization/install" -type d -print 2>/dev/null | sort
    )

    cppflags_ref="$(printf '%s ' "${include_dirs[@]}")"
    if (( STATIC_EXECUTABLES )); then
        lib_dirs+=("-static")
    fi
    ldflags_ref="$(printf '%s ' "${lib_dirs[@]}")"
    pkg_config_ref="$(join_by_colon "${pkg_dirs[@]}")"
}

dependency_libraries_for() {
    local library="$1"

    case "$library" in
        acl|acl-*)
            printf 'attr'
            ;;
        file-*)
            printf 'zlib-1.3.1'
            ;;
        freetype)
            printf 'zlib-1.3.1 bzip2 libpng brotli'
            ;;
        fontconfig)
            printf 'freetype libexpat'
            ;;
        glib)
            printf 'libffi pcre2-10.47 zlib-1.3.1'
            ;;
        harfbuzz)
            printf 'freetype glib'
            ;;
        libpng)
            printf 'zlib-1.3.1'
            ;;
        mpfr|mpfr-*)
            printf 'gmp-6.3.0'
            ;;
        selinux|selinux-*)
            printf 'pcre2-10.47'
            ;;
        pango)
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

variant_is_complete() {
    local variant_dir="$1"
    local install_dir="$variant_dir/install"

    has_static_archive "$install_dir" || return 1
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
        bzip2)
            printf 'bzip2'
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
        xz)
            printf 'xz'
            return
            ;;
    esac

    case "$library" in
        freetype)
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
    elif [[ -f "$source_root/configure.ac" || -f "$source_root/Makefile.am" ]]; then
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
    (( STATIC_EXECUTABLES )) && ldflags="-static"

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    cp -a -- "$source_path/." "$build_dir/"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env CC="$CC" CXX="$CXX" \
            "LDFLAGS=$ldflags" \
            perl Configure linux-x86_64 \
            "--prefix=$install_dir" "--openssldir=$install_dir/ssl" \
            no-shared no-tests no-dso no-module no-engine \
            "$flags" || exit $?
        run_logged "$log_file" make -j "$JOBS" build_sw || exit $?
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

    printf '\nBUILD %-18s %-5s (%s)\n' "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir/lib" "$install_dir/include"
    cp -a -- "$source_path/." "$build_dir/"

    run_logged "$log_file" make -C "$build_dir/libselinux/src" -j "$JOBS" \
        CC="$CC" \
        "CPPFLAGS=-I../../libsepol/include" \
        "CFLAGS=-$optimization -std=gnu17 -Wall -Wextra -Wno-error" \
        "LDFLAGS=-L$pcre_prefix/lib" \
        "PCRE_CFLAGS=-DUSE_PCRE2 -DPCRE2_CODE_UNIT_WIDTH=8 -I$pcre_prefix/include" \
        "PCRE_LDLIBS=$pcre_prefix/lib/libpcre2-8.a" \
        DISABLE_SETRANS=y DISABLE_RPM=y DISABLE_X11=y \
        libselinux.a || return $?

    cp -- "$build_dir/libselinux/src/libselinux.a" "$install_dir/lib/" || return $?
    cp -a -- "$build_dir/libselinux/include/selinux" "$install_dir/include/" || return $?
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

    prepare_configure_source "$build_system" "$source_root" "$log_file" || return $?
    configure_arguments "$library" "$install_dir" "$compiler" "$optimization" configure_args
    dependencies="$(dependency_libraries_for "$library")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    build_env=(
        "CC=$CC"
        "CXX=$CXX"
        "CFLAGS=$cflags"
        "CXXFLAGS=$cxxflags"
        "CPPFLAGS=$cppflags"
        "LDFLAGS=$ldflags"
        "PKG_CONFIG_PATH=$pkg_config_path"
    )

    printf '\nBUILD %-28s %-5s (%s, %s)\n' \
        "$library" "$optimization" "$CC" "$build_system"
    (
        cd -- "$build_dir"
        run_logged "$log_file" env "${build_env[@]}" \
            "$source_root/configure" "${configure_args[@]}" || exit $?
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
    ) || return $?

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

    need_build_cmd cmake || return 1
    dependencies="$(dependency_libraries_for "$library")"
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
            -DBUILD_SHARED_LIBS=OFF || return $?
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

    need_build_cmd meson || return 1
    need_build_cmd ninja || return 1
    dependencies="$(dependency_libraries_for "$library")"
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
            --buildtype=plain || return $?
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

    dependencies="$(dependency_libraries_for "$library")"
    dependency_flags "$compiler" "$optimization" "$library" "$dependencies" \
        cppflags ldflags pkg_config_path

    printf '\nBUILD %-28s %-5s (%s, make)\n' \
        "$library" "$optimization" "$CC"
    rm -rf -- "$build_dir"
    mkdir -p -- "$build_dir" "$install_dir"
    cp -a -- "$source_root/." "$build_dir/"
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

    if (( DRY_RUN )); then
        printf 'DRY   %-28s %-5s build_system=%s source=%s\n' \
            "$library" "$optimization" "$build_system" "$source_root"
        return 0
    fi

    mkdir -p -- "$build_dir" "$install_dir"
    : > "$log_file"

    if [[ "$library" == attr ]]; then
        # Automake may refresh configure/aclocal files during `make`.  Keep
        # those generated changes in the external build tree, never in the
        # revision-pinned source checkout.
        local prepared_source="$variant_dir/source"
        need_build_cmd rsync || return 1
        rm -rf -- "$prepared_source"
        mkdir -p -- "$prepared_source"
        rsync -a --exclude=.git/ "$source_root/" "$prepared_source/" || return $?
        source_root="$prepared_source"
        build_system="$(build_system_for_source "$library" "$source_root")"
    fi

    case "$build_system" in
        bzip2)
            build_bzip2_variant "$library" "$optimization" "$source_path" \
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
