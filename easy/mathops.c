#include "mathops.h"

int square(int x) {
    return x * x;
}

int cube(int x) {
    return x * x * x;
}

int power(int x, int n) {
    int result = 1;
    for (int i = 0; i < n; i++) {
        result *= x;
    }
    return result;
}

int isPrime(int x) {
    if (x <= 1) return 0;
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) return 0;
    }
    return 1;
}