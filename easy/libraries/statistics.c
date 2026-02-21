#include "statistics.h"
#include "mathops.h"
#include <stdio.h>

int sum(int arr[], int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += arr[i];
    }
    return total;
}

int average(int arr[], int size) {
    if (size == 0) return 0;
    return sum(arr, size) / size;
}

int findMax(int arr[], int size) {
    if (size == 0) return 0;
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int findMin(int arr[], int size) {
    if (size == 0) return 0;
    int min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}


// int variance(int arr[], int size) {
//     if (size == 0) return 0;
//     int avg = average(arr, size);
//     int sumSquaredDiff = 0;
  
//     for (int i = 0; i < size; i++) {
//         int diff = arr[i] - avg;
//         sumSquaredDiff += square(diff);  // Chiama funzione da mathops
//     }
    
//     return sumSquaredDiff / size;
// }

// int sumOfCubes(int arr[], int size) {
//     int total = 0;
//     printf("Computing sum of cubes for %d elements\n", size);
//     for (int i = 0; i < size; i++) {
//         total += cube(arr[i]);  // Chiama funzione da mathops
//     }
//     return total;
// }