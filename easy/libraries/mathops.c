#include "mathops.h"
#include "statistics.h"
#include <stdio.h>

int square(int x) {
    return x * x;
}

int cube(int x) {
    printf("Calculating cube of %d\n", x);
    return x * x * x;
}

int power(int x, int n) {
    int result = 1;
    printf("Calculating %d to the power of %d\n", x, n);
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

// Nuove funzioni che chiamano altre funzioni
int sumOfSquares(int a, int b) {
    return square(a) + square(b);
}

int powerOfCube(int x) {
    int c = cube(x);
    return power(c, 2);
}

int countPrimesInRange(int start, int end) {
    int count = 0;
    for (int i = start; i <= end; i++) {
        if (isPrime(i)) {
            count++;
        }
    }
    return count;
}

// Funzioni che usano statistics
int averageOfSquares(int arr[], int size) {
    int squared[size];
    for (int i = 0; i < size; i++) {
        squared[i] = square(arr[i]);
    }
    return average(squared, size);  // Chiama statistics
}

int maxPrimeInArray(int arr[], int size) {
    int max = findMax(arr, size);  // Chiama statistics
    while (max > 1) {
        if (isPrime(max)) {
            return max;
        }
        max--;
    }
    return -1;
}

int sumOfPrimeSquares(int arr[], int size) {
    int primeSquares[size];
    int count = 0;
    
    for (int i = 0; i < size; i++) {
        if (isPrime(arr[i])) {
            primeSquares[count++] = square(arr[i]);
        }
    }
    
    return sum(primeSquares, count);  // Chiama statistics
}