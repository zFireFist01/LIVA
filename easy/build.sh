#!/bin/bash
set -e



# Compila i file sorgente in oggetti
gcc -c mylib.c -o mylib.o
gcc -c mathops.c -o mathops.o

# Crea la libreria statica
ar rcs libmylib.a mylib.o mathops.o


# Crea le directory per i diversi compilatori
mkdir -p gcc13 gcc11 clang18 clang14

# # Compila il programma staticamente, con e senza ottimizzazione
# cd gcc13
# gcc-13 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt
# gcc-13 -static ../main.c -L.. -lmylib -o calculator_static
# cd ..

cd gcc11
gcc-11 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt
gcc-11 -static ../main.c -L.. -lmylib -o calculator_static
cd ..

# # Compila il programma staticamente con clang, con e senza ottimizzazione
# cd clang18
# clang-18 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt_clang
# clang-18 -static ../main.c -L.. -lmylib -o calculator_static_clang
# cd ..

cd clang14
clang-14 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt_clang
clang-14 -static ../main.c -L.. -lmylib -o calculator_static_clang
cd ..

echo "Build completata:"
echo "  → calculator_static_opt (con -O3)"
echo "  → calculator_static (senza ottimizzazioni)"
echo "  → calculator_static_opt_clang (con -O3)"
echo "  → calculator_static_clang (senza ottimizzazioni)"