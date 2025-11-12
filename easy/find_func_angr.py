import angr

proj = angr.Project("calculator_static", auto_load_libs=False)
cfg = proj.analyses.CFGFast()

# Trova la funzione per nome
func = cfg.kb.functions.function(name="call_fini")
print(func)


# Stampa il disassemblato in stile assembly
for block in func.blocks:
    print(block.disassembly)
