#include "mylib.h"
#include <stdio.h>

#define MIN 10
#define PAROLA "Difficile Analisi Binaria"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    printf("Devo sottrare!");
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}