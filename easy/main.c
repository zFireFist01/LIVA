#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "mylib.h"
#include "mathops.h"

// New utility function prototypes
char* reverseString(char* str);
int countDigits(int number);
int isPalindrome(int num);
void printBinary(int num);


// New utility function implementations
char* reverseString(char* str) {
    int length = strlen(str);
    int start = 0;
    int end = length - 1;
    char temp;
    
    while (start < end) {
        temp = str[start];
        str[start] = str[end];
        str[end] = temp;
        start++;
        end--;
    }
    return str;
}

int countDigits(int number) {
    int count = 0;
    if (number == 0) return 1;
    while (number != 0) {
        count++;
        number /= 10;
    }
    return count;
}

int isPalindrome(int num) {
    int reversed = 0, original = num;
    while (num > 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return original == reversed;
}

void printBinary(int num) {
    if (num > 1) {
        printBinary(num / 2);
    }
    printf("%d", num % 2);
}

int main() {
    int a = 10, b = 5;
    char str[] = "Hello Binary Analysis";
    int testNum = 12321;
    int decimalNum = 42;

    // Calling functions from mylib.h and mathops.h
    
    printf("Calculator Demo using mylib.h and mathops.h\n");
    printf("------------------------------------------\n");
    printf("First number: %d\n", a);
    printf("Second number: %d\n", b);
    
    printf("\nBasic Operations:\n");
    printf("Addition: %d + %d = %d\n", a, b, add(a, b));
    printf("Subtraction: %d - %d = %d\n", a, b, subtract(a, b));
    printf("Multiplication: %d * %d = %d\n", a, b, multiply(a, b));
    printf("Division: %d / %d = %d\n", a, b, divide(a, b));

    printf("\nAdvanced Operations:\n");
    printf("Square of %d = %d\n", a, square(a));
    printf("Cube of %d = %d\n", a, cube(a));
    printf("Power: %d^%d = %d\n", a, b, power(a, b));
    printf("%d is prime? %s\n", a, isPrime(a) ? "Yes" : "No");
    printf("%d is prime? %s\n", 17, isPrime(17) ? "Yes" : "No");


    // Calling function implemented in this file

    printf("\n--- Testing Utility Functions ---\n");
    printf("Original string: %s\n", str);
    printf("Reversed string: %s\n", reverseString(str));
    printf("Number of digits in %d: %d\n", testNum, countDigits(testNum));
    printf("Is %d palindrome? %s\n", testNum, isPalindrome(testNum) ? "Yes" : "No");
    printf("Binary representation of %d: ", decimalNum);
    printBinary(decimalNum);
    printf("\n");

    return 0;
}