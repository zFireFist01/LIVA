#!/bin/bash
set -e

# Compila i file sorgente in oggetti
gcc -c libraries/statistics.c -o libraries/statistics.o 
gcc -c libraries/mylib.c -o libraries/mylib.o
gcc -c libraries/mathops.c -o libraries/mathops.o

# Crea la libreria statica
ar rcs libmylib.a libraries/mylib.o libraries/mathops.o
ar rcs libstats.a libraries/statistics.o

# Crea le directory per i diversi compilatori
mkdir -p gcc13 gcc11 clang18 clang14

# # Compila il programma staticamente, con e senza ottimizzazione
# cd gcc13
# gcc-13 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt
# gcc-13 -static ../main.c -L.. -lmylib -o calculator_static
# objdump -d calculator_static_opt > disass_opt.txt
# objdump -d calculator_static > disass.txt
# cd ..

cd gcc11
gcc-11 -static ../main.c -Wl,--trace -L..  -lmylib -lstats -O3 -o calculator_static_opt
gcc-11 -static ../main.c -L..  -lmylib -lstats -o calculator_static

# Crea i file di disassemblaggio
objdump -d calculator_static_opt > disass_opt.txt
objdump -d calculator_static > disass.txt

# Crea il contenuto di rodata
objdump -s -j .rodata calculator_static > rodata.txt
objdump -s -j .rodata calculator_static_opt > rodata_opt.txt

cd ..

# # Compila il programma staticamente con clang, con e senza ottimizzazione
# cd clang18
# clang-18 -static ../main.c -L.. -lmylib -O3 -o calculator_static_opt
# clang-18 -static ../main.c -L.. -lmylib -o calculator_static

#objdump -d calculator_static_opt > disass_opt.txt
#objdump -d calculator_static > disass.txt

# cd ..

cd clang14
clang-14 -static ../main.c -L.. -lmylib -lstats -O3 -o calculator_static_opt
clang-14 -static ../main.c -L.. -lmylib -lstats -o calculator_static

# Crea i file di disassemblaggio
objdump -d calculator_static_opt > disass_opt.txt
objdump -d calculator_static > disass.txt

# Crea il contenuto di rodata
objdump -s -j .rodata calculator_static > rodata.txt
objdump -s -j .rodata calculator_static_opt > rodata_opt.txt

cd ..

# Pulisci i file temporanei
rm libmylib.a libstats.a
rm libraries/*.o

echo "Build completata:"
echo "  → calculator_static_opt (con -O3)"
echo "  → calculator_static (senza ottimizzazioni)"
echo "  → calculator_static_opt_clang (con -O3)"
echo "  → calculator_static_clang (senza ottimizzazioni)"
echo "Eliminate librerie temporanee."