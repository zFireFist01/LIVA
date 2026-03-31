#include "scommessa.h"
#include <stdio.h>
#include "mathops.h"
#include "mylib.h"

int foo(int x) {
    if (x > 9){
        x = square(x);
    } else {
        x = add(x, x);
    }
    return x;
}