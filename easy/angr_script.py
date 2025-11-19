import sys
import angr
import logging
import re

# Disattiva spam di warning e dirty helper
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)


def normalize_instruction(insn_str):
    """Normalize instruction removing addresses and offsets"""
    # Remove hex values
    insn_str = re.sub(r'0x[0-9a-f]+', '*', insn_str, flags=re.IGNORECASE)
    # Normalize whitespace
    insn_str = re.sub(r'\s+', ' ', insn_str)
    return insn_str.strip().lower()


def extract_mnemonic(insn_str):
    """Extract just the instruction mnemonic"""
    parts = insn_str.split()
    return parts[0] if parts else ""


def is_call_fini(asm_instructions):
    """
    Identifica call_fini cercando pattern caratteristici:
    - endbr64 all'inizio
    - due lea consecutive (caricamento array bounds)
    - sub + sar (calcolo size array)
    - loop con call indiretta e decremento
    - jmp finale (tail call a _fini)
    """
    normalized = [normalize_instruction(insn) for insn in asm_instructions]
    mnemonics = [extract_mnemonic(insn) for insn in normalized]
    
    # Pattern obbligatori
    has_endbr64 = 'endbr64' in mnemonics
    has_two_lea = sum(1 for m in mnemonics if m == 'lea') >= 2
    has_sub = 'sub' in mnemonics
    has_sar = 'sar' in mnemonics
    has_indirect_call = any('call' in insn and ('*' in insn or 'qword ptr' in insn or 'QWORD PTR' in insn) 
                            for insn in normalized)
    has_final_jmp = mnemonics and mnemonics[-1] == 'jmp'
    
    # Cerca pattern loop: call + sub/dec + jne
    has_loop = False
    for i in range(len(mnemonics) - 2):
        if (mnemonics[i] in ['call'] and 
            mnemonics[i+1] in ['sub', 'dec'] and 
            mnemonics[i+2] in ['jne', 'jnz']):
            has_loop = True
            break
    
    # Score basato sui pattern trovati
    score = sum([
        has_endbr64,
        has_two_lea,
        has_sub,
        has_sar,
        has_indirect_call,
        has_final_jmp,
        has_loop
    ])
    
    # Richiedi almeno 5/7 pattern
    return score >= 5


def main(binary_path, output_path):
    project = angr.Project(binary_path, auto_load_libs=False)
    cfg = project.analyses.CFGFast()
    functions = cfg.kb.functions

    target_func = None
    for func in functions.values():
        asm_instructions = []
        for block in func.blocks:
            for insn in block.capstone.insns:
                full_insn = f"{insn.mnemonic} {insn.op_str}"
                asm_instructions.append(full_insn)

        if is_call_fini(asm_instructions):
            target_func = func
            print(f"[+] Found candidate: {func.name} @ {hex(func.addr)}")
            # Opzionale: stampa le istruzioni per debug
            # for insn in asm_instructions[:20]:
            #     print(f"    {insn}")
            break

    if target_func is None:
        print("[-] Target function not found.")
        return

    print(f"[+] Target function: {target_func.name} @ {hex(target_func.addr)}")

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