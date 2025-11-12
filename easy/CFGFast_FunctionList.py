import sys
import angr
import logging

# Disattiva spam di warning e dirty helper
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)

def main(binary_path):
    print(f"[+] Loading binary: {binary_path}")
    project = angr.Project(binary_path, auto_load_libs=False)

    print("[+] Building CFGFast (complete scan)...")
    cfg = project.analyses.CFGFast(
        normalize=True,
        force_complete_scan=True,
        force_smart_scan=False  # evita conflitto
    )

    print("[+] Functions discovered:")
    functions = sorted(project.kb.functions.values(), key=lambda f: f.addr)
    for f in functions:
        print(f"0x{f.addr:x}\t{f.name}")

    print(f"\n[+] Total functions found: {len(functions)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 list_functions.py <binary_path>")
        sys.exit(1)
    main(sys.argv[1])
