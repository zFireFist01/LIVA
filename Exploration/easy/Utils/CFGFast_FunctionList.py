import angr
import logging
import sys

# Disattiva warning
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)

BINARY_PATH = sys.argv[1]

def print_call_tree(cfg, func_node, visited=None, prefix="", is_last=True):
    """Stampa ricorsivamente l'albero delle chiamate"""
    if visited is None:
        visited = set()
    
    # Ottieni indirizzo della funzione
    if hasattr(func_node, 'addr'):
        addr = func_node.addr
        name = func_node.name if func_node.name else hex(addr)
    elif isinstance(func_node, int):
        addr = func_node
        func = cfg.kb.functions.get(addr)
        name = func.name if func and func.name else hex(addr)
    else:
        return
    
    if addr in visited:
        return
    visited.add(addr)
    
    # Simboli per l'albero
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{name} (0x{addr:x})")
    
    # Ottieni successori nel call graph
    cg = cfg.functions.callgraph
    try:
        successors = list(cg.successors(func_node))
    except:
        # Se il nodo non esiste nel grafo, prova con l'indirizzo
        try:
            successors = list(cg.successors(addr))
        except:
            successors = []
    
    # Filtra funzioni di sistema/librerie
    filtered = []
    for succ in successors:
        if hasattr(succ, 'is_simprocedure') and succ.is_simprocedure:
            continue
        if hasattr(succ, 'name') and succ.name and succ.name.startswith('__'):
            continue
        filtered.append(succ)
    
    # Stampa ricorsivamente i figli
    for i, callee in enumerate(filtered):
        is_last_child = (i == len(filtered) - 1)
        extension = "    " if is_last else "│   "
        print_call_tree(cfg, callee, visited, prefix + extension, is_last_child)

def main():
    print(f"[+] Loading binary: {BINARY_PATH}")
    proj = angr.Project(BINARY_PATH, auto_load_libs=False)

    print("[+] Building CFG...")
    cfg = proj.analyses.CFGFast(normalize=True, force_complete_scan=True)

    # Trova il main
    main_func = None
    if 'main' in proj.kb.functions:
        main_func = proj.kb.functions['main']
    else:
        # Usa entry point
        entry_addr = proj.entry
        main_func = proj.kb.functions.function(entry_addr)
    
    if not main_func:
        print("[-] Could not find main function")
        return

    print(f"\n[+] Call tree from {main_func.name}:\n")
    print_call_tree(cfg, main_func)

if __name__ == "__main__":
    main()