import os
import re
import difflib

FILE_A = "disass.txt"
FILE_B = "disass_opt.txt"
OUTDIR = "diff_out"

CUSTOM_FUNCTIONS = [
    # mylib.c
    "add", "subtract", "multiply", "divide",
    # mathops.c
    "square", "cube", "power", "isPrime", "sumOfSquares", 
    "powerOfCube", "countPrimesInRange", "averageOfSquares", 
    "maxPrimeInArray", "sumOfPrimeSquares",
    # statistics.c
    "sum", "average", "findMax", "findMin", "variance", "sumOfCubes",
    # utility functions
    "countDigits", "isPalindrome", "printBinary", "reverseString",
    # main
    "main"
]

FUNC_REGEX = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:")


def parse_functions(path):
    functions = {}
    current_name = None
    current_body = []

    with open(path, "r") as f:
        for line in f:
            m = FUNC_REGEX.match(line)
            if m:
                # Save previous function
                if current_name:
                    functions[current_name] = current_body
                current_name = m.group(1)
                current_body = [line]
            else:
                if current_name:
                    current_body.append(line)

    # Save last function if any
    if current_name:
        functions[current_name] = current_body

    return functions


def write_diff(name, bodyA, bodyB):
    diff = difflib.unified_diff(
        bodyA,
        bodyB,
        fromfile=f"{name} (FILE A)",
        tofile=f"{name} (FILE B)",
        lineterm=""
    )

    path = os.path.join(OUTDIR, f"{name}.diff.txt")
    with open(path, "w") as f:
        for line in diff:
            f.write(line + "\n")


def main():
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    print("[*] Parsing functions...")
    funcsA = parse_functions(FILE_A)
    funcsB = parse_functions(FILE_B)

    summary_lines = []

    print("[*] Generating diffs only for CUSTOM_FUNCTIONS...")
    for name in CUSTOM_FUNCTIONS:
        inA = name in funcsA
        inB = name in funcsB

        if not inA and not inB:
            summary_lines.append(f"{name}: NOT FOUND in either file")
            continue
        if not inA:
            summary_lines.append(f"{name}: ONLY in {FILE_B}")
            continue
        if not inB:
            summary_lines.append(f"{name}: ONLY in {FILE_A}")
            continue

        bodyA = funcsA[name]
        bodyB = funcsB[name]

        if bodyA == bodyB:
            summary_lines.append(f"{name}: IDENTICAL")
            continue

        write_diff(name, bodyA, bodyB)
        summary_lines.append(f"{name}: DIFFERENT → diff_out/{name}.diff.txt")

    with open(os.path.join(OUTDIR, "summary.txt"), "w") as sf:
        sf.write("\n".join(summary_lines))

    print("[✓] Done. Check diff_out/summary.txt and per-function diffs.")


if __name__ == "__main__":
    main()
