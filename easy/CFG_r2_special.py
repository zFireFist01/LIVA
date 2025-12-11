import os
import r2pipe
import json

BINARY_PATH = "./gcc11/calculator_static_opt"
OUTPUT_DIR = "./cfg_json"

# Definisci le funzioni delle tue librerie che vuoi analizzare
CUSTOM_FUNCTIONS = [
    # mylib.c
    "add", "subtract", "multiply", "divide",
    # mathops.c
    "square", "cube", "power", "isPrime", "sumOfSquares", 
    "powerOfCube", "countPrimesInRange", "averageOfSquares", 
    "maxPrimeInArray", "sumOfPrimeSquares",
    # statistics.c
    "sum", "average", "findMax", "findMin", "variance", "sumOfCubes",
    # utility functions (aggiungi queste)
    "countDigits", "isPalindrome", "printBinary", "reverseString",
    # main
    "main"
]

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def convert_addr_to_hex(obj):
    """Converte ricorsivamente tutti i campi 'addr' in formato esadecimale"""
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if key == "addr" and isinstance(value, int):
                new_obj[key] = hex(value)
            elif key in ["jump", "fail", "fcn_addr", "fcn_last"] and isinstance(value, int):
                new_obj[key] = hex(value)
            else:
                new_obj[key] = convert_addr_to_hex(value)
        return new_obj
    elif isinstance(obj, list):
        return [convert_addr_to_hex(item) for item in obj]
    else:
        return obj

def sort_cfg(cfg):
    if not cfg:
        return cfg

    graph = cfg[0]

    if "blocks" in graph:
        graph["blocks"] = sorted(graph["blocks"], key=lambda b: int(b.get("offset", b.get("addr", "0x0")), 16) if isinstance(b.get("offset", b.get("addr", "0x0")), str) else b.get("offset", b.get("addr", 0)))

    graph = dict(sorted(graph.items(), key=lambda x: x[0]))

    return [graph]

def is_custom_function(name):
    """Verifica se la funzione è una delle nostre funzioni custom"""
    for func in CUSTOM_FUNCTIONS:
        if func in name:
            return True
    return False

def main():
    ensure_dir(OUTPUT_DIR)

    r2 = r2pipe.open(BINARY_PATH)
    r2.cmd("aaa")

    raw = r2.cmd("aflj")
    functions = json.loads(raw) if raw.strip() else []

    print(f"[*] Trovate {len(functions)} funzioni totali")
    
    filtered_count = 0
    for f in functions:
        addr = f.get("offset", f.get("addr"))
        if addr is None:
            print("[-] Funzione senza offset, skip:", f)
            continue

        name = f.get("name", f"fcn_{addr:x}")
        
        # Filtra solo le funzioni custom
        if not is_custom_function(name):
            continue
            
        filtered_count += 1
        name_clean = name.replace("/", "_")

        print(f"[+] Generazione CFG JSON per {name_clean} @ {addr:#x}")

        raw_cfg = r2.cmd(f"agfj @{addr}")
        if not raw_cfg.strip():
            print(f"    [-] Nessun CFG trovato")
            continue

        try:
            cfg = json.loads(raw_cfg)
        except json.JSONDecodeError:
            print(f"    [-] JSON non valido per {name_clean}")
            continue

        cfg_sorted = sort_cfg(cfg)
        
        # Converti tutti gli indirizzi in formato esadecimale
        cfg_hex = convert_addr_to_hex(cfg_sorted)

        out_file = os.path.join(OUTPUT_DIR, f"cfg_{addr:x}_{name_clean}.json")

        with open(out_file, "w") as fp:
            json.dump(cfg_hex, fp, indent=4)

        print(f"    [+] Salvato in {out_file}")

    print(f"\n[*] Elaborate {filtered_count} funzioni custom su {len(functions)} totali")
    r2.quit()


if __name__ == "__main__":
    main()