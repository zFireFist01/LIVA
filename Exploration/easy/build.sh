#!/bin/bash
set -euo pipefail

OPT_FLAGS="-O3"
INCLUDE_FLAGS=(
  -Ilibraries/mylib
  -Ilibraries/mathops
  -Ilibraries/statistics
  -Ilibraries/scommessa
)
BINARY_INCLUDE_FLAGS=(
  -I../.build_include
  -I../libraries/mylib
  -I../libraries/mathops
  -I../libraries/statistics
  -I../libraries/scommessa
)

LIB_SOURCES=(
  "libraries/mylib/mylib.c"
  "libraries/mathops/mathops.c"
  "libraries/statistics/statistics.c"
  "libraries/scommessa/scommessa.c"
)

compile_libraries() {
  local cc="$1"
  local suffix="$2"
  local opt_flags="$3"

  echo "Compilo librerie${suffix:+ ($suffix)} con $cc ${opt_flags:-senza ottimizzazioni}"

  for src in "${LIB_SOURCES[@]}"; do
    local obj="${src%.c}${suffix}.o"
    local disass="${src%.c}${suffix}.disass.txt"
    "$cc" -c "$src" "${INCLUDE_FLAGS[@]}" $opt_flags -o "$obj"
    objdump -d "$obj" > "$disass"
  done

  ar rcs "libraries/libmylib${suffix}.a" \
    "libraries/mylib/mylib${suffix}.o" \
    "libraries/mathops/mathops${suffix}.o"
  ar rcs "libraries/libstats${suffix}.a" \
    "libraries/statistics/statistics${suffix}.o"
  ar rcs "libraries/libscommessa${suffix}.a" \
    "libraries/scommessa/scommessa${suffix}.o"
}

cleanup_optimized_libraries() {
  echo "Elimino solo gli archivi delle librerie ottimizzate, lasciando gli .o"
  rm -f \
    libraries/libmylib_opt.a \
    libraries/libstats_opt.a \
    libraries/libscommessa_opt.a
}

prepare_compat_headers() {
  mkdir -p .build_include/libraries
  ln -sf ../../libraries/mylib/mylib.h .build_include/libraries/mylib.h
  ln -sf ../../libraries/mathops/mathops.h .build_include/libraries/mathops.h
  ln -sf ../../libraries/statistics/statistics.h .build_include/libraries/statistics.h
  ln -sf ../../libraries/scommessa/scommessa.h .build_include/libraries/scommessa.h
}

build_binary() {
  local cc="$1"
  local out_dir="$2"

  if ! command -v "$cc" >/dev/null 2>&1; then
    echo "Salto $out_dir: compilatore '$cc' non trovato."
    return
  fi

  mkdir -p "$out_dir"

  (
    cd "$out_dir"

    "$cc" -static ../main.c "${BINARY_INCLUDE_FLAGS[@]}" \
      -L../libraries -lmylib -lstats -O0 -o calculator_static

    "$cc" -static ../main.c "${BINARY_INCLUDE_FLAGS[@]}" \
      -L../libraries -lmylib_opt -lstats_opt $OPT_FLAGS -o calculator_static_opt

    "$cc" -static ../main.c "${BINARY_INCLUDE_FLAGS[@]}" \
      -L../libraries -lmylib_opt -lstats_opt -O0 -o calculator_static_main_noopt_lib_opt

    "$cc" -static ../main.c "${BINARY_INCLUDE_FLAGS[@]}" \
      -L../libraries -lmylib -lstats $OPT_FLAGS -o calculator_static_main_opt_lib_noopt

    objdump -d calculator_static > disass.txt
    objdump -d calculator_static_opt > disass_opt.txt
    objdump -d calculator_static_main_noopt_lib_opt > disass_main_noopt_lib_opt.txt
    objdump -d calculator_static_main_opt_lib_noopt > disass_main_opt_lib_noopt.txt

    objdump -s -j .rodata calculator_static > rodata.txt
    objdump -s -j .rodata calculator_static_opt > rodata_opt.txt
    objdump -s -j .rodata calculator_static_main_noopt_lib_opt > rodata_main_noopt_lib_opt.txt
    objdump -s -j .rodata calculator_static_main_opt_lib_noopt > rodata_main_opt_lib_noopt.txt
  )
}

if ! command -v gcc >/dev/null 2>&1; then
  echo "Errore: gcc non trovato. Serve almeno gcc per creare le librerie."
  exit 1
fi

prepare_compat_headers
compile_libraries gcc "" "-O0"
compile_libraries gcc "_opt" "$OPT_FLAGS"

build_binary gcc-13 gcc13
build_binary gcc-11 gcc11
build_binary clang-18 clang18
build_binary clang-14 clang14

cleanup_optimized_libraries

echo "Build completata:"
echo "  - libraries/libmylib.a, libraries/libstats.a, libraries/libscommessa.a (senza ottimizzazioni)"
echo "  - gli archivi *_opt.a sono stati usati per il link e poi eliminati; gli *_opt.o restano disponibili"
echo "  - <compilatore>/calculator_static (main e librerie senza ottimizzazioni)"
echo "  - <compilatore>/calculator_static_opt (main e librerie con $OPT_FLAGS)"
echo "  - <compilatore>/calculator_static_main_noopt_lib_opt (main senza ottimizzazioni, librerie con $OPT_FLAGS)"
echo "  - <compilatore>/calculator_static_main_opt_lib_noopt (main con $OPT_FLAGS, librerie senza ottimizzazioni)"
