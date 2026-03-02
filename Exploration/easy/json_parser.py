import os
import sys
import json
import re
from typing import Dict, Set, List, Any, Tuple


def load_cfg_file(path: str) -> List[dict]:
    """
    Carica il contenuto JSON del file.
    Supporta sia:
      - lista con un solo grafo: [ { ... } ]
      - oggetto singolo: { ... }
    Restituisce sempre una lista di grafi.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def build_cfg(graph: dict):
    """
    A partire da un oggetto di grafo (output di agfj),
    costruisce:
      - mappa addr -> blocco
      - successori per blocco
      - predecessori per blocco
    """
    blocks = graph.get("blocks", [])
    addr_to_block: Dict[int, dict] = {}
    succ: Dict[int, Set[int]] = {}
    pred: Dict[int, Set[int]] = {}

    for b in blocks:
        addr = b.get("addr")
        if addr is None:
            continue
        
        # Gestisci sia interi che stringhe hex
        if isinstance(addr, str):
            addr = int(addr, 16)
            
        addr_to_block[addr] = b
        succ[addr] = set()
        pred[addr] = set()

    for b in blocks:
        src = b.get("addr")
        if src is None:
            continue
            
        if isinstance(src, str):
            src = int(src, 16)

        for key in ("jump", "fail"):
            target = b.get(key)
            if target is not None:
                if isinstance(target, str):
                    target = int(target, 16)
                if target in addr_to_block:
                    succ[src].add(target)
                    pred[target].add(src)

    return addr_to_block, succ, pred


def extract_called_functions(graph: dict) -> Set[str]:
    """
    Estrae i nomi delle funzioni chiamate analizzando le istruzioni 'call'.
    """
    called_functions = set()
    blocks = graph.get("blocks", [])
    
    for block in blocks:
        ops = block.get("ops", [])
        for op in ops:
            disasm = op.get("disasm", "")
            
            # Pattern per istruzioni call
            # Esempi: "call sym.add", "call 0x401234", "call fcn.00401234"
            if disasm.startswith("call "):
                target = disasm[5:].strip()
                
                # Estrai il nome della funzione se presente
                # sym.add -> add
                # fcn.00401234 -> ignora (funzione anonima)
                if target.startswith("sym."):
                    func_name = target[4:]  # Rimuovi "sym."
                    called_functions.add(func_name)
                elif target.startswith("fcn."):
                    # Opzionale: gestisci funzioni anonime
                    pass
                # Ignora chiamate dirette a indirizzi numerici
    
    return called_functions


def find_function_file(func_name: str, cfg_dir: str) -> str:
    """
    Cerca il file JSON che contiene la funzione specificata.
    Cerca pattern: cfg_*_sym.<func_name>.json o cfg_*_<func_name>.json
    """
    if not os.path.isdir(cfg_dir):
        return None
    
    # Pattern possibili:
    # 1. cfg_<addr>_sym.<func_name>.json (più comune)
    # 2. cfg_<addr>_<func_name>.json
    patterns = [
        f"_sym.{func_name}.json",
        f"_{func_name}.json"
    ]
    
    for fname in os.listdir(cfg_dir):
        if not fname.endswith(".json"):
            continue
        
        # Verifica se il filename contiene uno dei pattern
        for pattern in patterns:
            if pattern in fname:
                return os.path.join(cfg_dir, fname)
    
    return None


def draw_block_ascii(block: dict, addr: int, succ_list: List[int], pred_list: List[int]) -> str:
    """
    Disegna un blocco in formato ASCII art.
    """
    lines = []
    width = 70
    
    # Header del blocco
    header = f" Block @ 0x{addr:x} "
    lines.append("╔" + "═" * width + "╗")
    lines.append("║" + header.center(width) + "║")
    lines.append("╠" + "═" * width + "╣")
    
    # Predecessori
    if pred_list:
        pred_str = ", ".join(f"0x{x:x}" for x in pred_list)
        lines.append("║" + f" ↓ From: {pred_str}".ljust(width) + "║")
        lines.append("╟" + "─" * width + "╢")
    
    # Istruzioni
    ops = block.get("ops", [])
    if ops:
        for op in ops:
            op_addr = op.get("addr")
            disasm = op.get("disasm", "")
            
            if isinstance(op_addr, str):
                op_addr = int(op_addr, 16)
            
            if op_addr is not None:
                instr = f" 0x{op_addr:x}: {disasm}"
            else:
                instr = f" ???: {disasm}"
            
            # Tronca se troppo lungo
            if len(instr) > width:
                instr = instr[:width-3] + "..."
            
            lines.append("║" + instr.ljust(width) + "║")
    else:
        lines.append("║" + " (no instructions)".ljust(width) + "║")
    
    # Successori
    if succ_list:
        lines.append("╟" + "─" * width + "╢")
        
        jump = block.get("jump")
        fail = block.get("fail")
        
        if isinstance(jump, str):
            jump = int(jump, 16)
        if isinstance(fail, str):
            fail = int(fail, 16)
        
        if jump is not None:
            lines.append("║" + f" ↓ Jump to: 0x{jump:x}".ljust(width) + "║")
        if fail is not None:
            lines.append("║" + f" ↓ Fail to: 0x{fail:x}".ljust(width) + "║")
    
    # Footer
    lines.append("╚" + "═" * width + "╝")
    
    return "\n".join(lines)


def print_graph_ascii(graph: dict, file_name: str, output_file=None, depth: int = 0) -> Set[str]:
    """
    Stampa un singolo grafo in formato ASCII art.
    Se output_file è specificato, scrive su file invece che su stdout.
    Restituisce l'insieme delle funzioni chiamate.
    """
    func_name = graph.get("name", "<unknown>")
    func_addr = graph.get("addr")
    
    if isinstance(func_addr, str):
        func_addr = int(func_addr, 16)
    
    nargs = graph.get("nargs")
    nlocals = graph.get("nlocals")
    ninstr = graph.get("ninstr")
    size = graph.get("size")
    stack = graph.get("stack")

    output = []
    indent = "  " * depth
    
    output.append(indent + "=" * 80)
    if depth > 0:
        output.append(indent + f"[CALLED FUNCTION - Depth {depth}]")
    output.append(indent + f"File: {file_name}")
    
    if func_addr is not None:
        output.append(indent + f"Function: {func_name} @ 0x{func_addr:x}")
    else:
        output.append(indent + f"Function: {func_name}")
    
    output.append(indent + f"  nargs={nargs} | nlocals={nlocals} | ninstr={ninstr} | size={size} | stack={stack}")
    output.append(indent + "=" * 80)
    output.append("")

    addr_to_block, succ, pred = build_cfg(graph)

    # Disegna ogni blocco in ordine di indirizzo
    for addr in sorted(addr_to_block.keys()):
        block = addr_to_block[addr]
        succ_list = sorted(list(succ.get(addr, [])))
        pred_list = sorted(list(pred.get(addr, [])))
        
        block_ascii = draw_block_ascii(block, addr, succ_list, pred_list)
        # Indenta ogni riga del blocco
        indented_block = "\n".join(indent + line for line in block_ascii.split("\n"))
        output.append(indented_block)
        output.append("")
    
    output.append(indent + "=" * 80)
    output.append("")
    
    result = "\n".join(output)
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(result)
    else:
        print(result)
    
    # Estrai le funzioni chiamate
    return extract_called_functions(graph)


def process_function_with_calls(graph: dict, file_name: str, cfg_dir: str, 
                                output_file=None, processed: Set[str] = None, 
                                depth: int = 0) -> None:
    """
    Processa una funzione e ricorsivamente le sue chiamate.
    """
    if processed is None:
        processed = set()
    
    func_name = graph.get("name", "<unknown>")
    
    # Evita loop infiniti
    if func_name in processed:
        return
    
    processed.add(func_name)
    
    # Stampa la funzione corrente
    called_funcs = print_graph_ascii(graph, file_name, output_file, depth)
    
    # Processa le funzioni chiamate
    for called_func in sorted(called_funcs):
        if called_func not in processed:
            func_file = find_function_file(called_func, cfg_dir)
            
            if func_file:
                print(f"{'  ' * (depth + 1)}[*] Processing called function: {called_func}")
                graphs = load_cfg_file(func_file)
                
                for g in graphs:
                    process_function_with_calls(
                        g, 
                        os.path.basename(func_file), 
                        cfg_dir, 
                        output_file, 
                        processed, 
                        depth + 1
                    )
            else:
                print(f"{'  ' * (depth + 1)}[-] Function {called_func} not found in {cfg_dir}")


def process_path(path: str, output_dir: str = None):
    """
    Se path è un file .json → stampa quel CFG con le funzioni chiamate.
    Se path è una directory → stampa tutti i .json dentro.
    Se output_dir è specificato, crea file ASCII nella directory.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determina la directory dei CFG per la ricerca delle funzioni
    if os.path.isfile(path):
        cfg_dir = os.path.dirname(path)
    else:
        cfg_dir = path
    
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            full_path = os.path.join(path, fname)
            graphs = load_cfg_file(full_path)
            
            if output_dir:
                output_file = os.path.join(output_dir, fname.replace(".json", ".txt"))
                open(output_file, 'w').close()
                
                for g in graphs:
                    print(f"[+] Processing: {fname}")
                    process_function_with_calls(g, fname, cfg_dir, output_file)
                print(f"[+] Created: {output_file}")
            else:
                for g in graphs:
                    process_function_with_calls(g, fname, cfg_dir)
    else:
        graphs = load_cfg_file(path)
        base_name = os.path.basename(path)
        
        if output_dir:
            output_file = os.path.join(output_dir, base_name.replace(".json", ".txt"))
            open(output_file, 'w').close()
            
            for g in graphs:
                print(f"[+] Processing: {base_name}")
                process_function_with_calls(g, base_name, cfg_dir, output_file)
            print(f"[+] Created: {output_file}")
        else:
            for g in graphs:
                process_function_with_calls(g, base_name, cfg_dir)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Uso: {sys.argv[0]} <file.json | directory> [output_directory]")
        sys.exit(1)

    path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None
    
    if not os.path.exists(path):
        print(f"Errore: path '{path}' inesistente")
        sys.exit(1)

    process_path(path, output_dir)


if __name__ == "__main__":
    main()