import sys
import angr
import logging

# Disattiva spam di warning e dirty helper
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)


def main(binary_path, output_path):
    project = angr.Project(binary_path, auto_load_libs=False)
    cfg = project.analyses.CFGFast()
    functions = cfg.kb.functions

    target_asm = [
        "endbr64",
        "push %rbp",
        "lea",  # pattern generico, evita offset
        "push %rbx",
        "sub %rax,%rbx",
        "sar $0x3,%rbx",
        "call *0x0(%rbp,%rbx,8)",
        "jmp"
    ]

    target_func = None
    for func in functions.values():
        asm_instructions = []
        for block in func.blocks:
            for insn in block.capstone.insns:
                asm_instructions.append(insn.mnemonic + ' ' + insn.op_str)

        if all(any(t in asm for asm in asm_instructions) for t in target_asm):
            target_func = func
            break

    if target_func is None:
        print("[-] Target function not found.")
        return

    print(f"[+] Found target function: {target_func.name} @ {hex(target_func.addr)}")

    valid_functions = [f for f in functions.values() if f.addr <= target_func.addr]

    with open(output_path, 'w') as f:
        for func in sorted(valid_functions, key=lambda f: f.addr):
            f.write(f"{hex(func.addr)} {func.name}\n")

    print(f"[+] Saved {len(valid_functions)} functions to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python angr_script.py <binary_path> <output_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
