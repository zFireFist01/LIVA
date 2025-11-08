#!/bin/bash
set -e

# Compila i file sorgente in oggetti
gcc -c mylib.c -o mylib.o
gcc -c mathops.c -o mathops.o

# Crea la libreria statica
ar rcs libmylib.a mylib.o mathops.o

# Compila il programma staticamente, con e senza ottimizzazione
gcc -static main.c -L. -lmylib -O3 -o calculator_static_opt
gcc -static main.c -L. -lmylib -o calculator_static

echo "Build completata:"
echo "  → calculator_static_opt (con -O3)"
echo "  → calculator_static (senza ottimizzazioni)"
