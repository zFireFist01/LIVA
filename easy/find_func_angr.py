import angr
import sys

#Se manca il file binario come argomento, esci
if len(sys.argv) < 2:
    print("Usage: python find_func_angr.py <binary>")
    sys.exit(1)

proj = angr.Project(sys.argv[1], auto_load_libs=False)
cfg = proj.analyses.CFGFast()

# Trova la funzione per nome
func = cfg.kb.functions.function(name="call_fini")
print(func)


# Stampa il disassemblato in stile assembly
for block in func.blocks:
    print(block.disassembly)
