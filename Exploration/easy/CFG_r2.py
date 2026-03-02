import os
import r2pipe
import json

BINARY_PATH = "./libraries/statistics.o"
OUTPUT_DIR = "./cfg_json_lib"

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def sort_cfg(cfg):
    if not cfg:
        return cfg

    graph = cfg[0]

    if "blocks" in graph:
        graph["blocks"] = sorted(graph["blocks"], key=lambda b: b.get("offset", 0))

    graph = dict(sorted(graph.items(), key=lambda x: x[0]))

    return [graph]

def main():
    ensure_dir(OUTPUT_DIR)

    r2 = r2pipe.open(BINARY_PATH)
    r2.cmd("aaa")

    raw = r2.cmd("aflj")
    functions = json.loads(raw) if raw.strip() else []

    for f in functions:
        # gestisci entrambi i campi: offset o addr
        addr = f.get("offset", f.get("addr"))
        if addr is None:
            print("[-] Funzione senza offset, skip:", f)
            continue

        name = f.get("name", f"fcn_{addr:x}").replace("/", "_")

        print(f"[+] Generazione CFG JSON per {name} @ {addr:#x}")

        raw_cfg = r2.cmd(f"agfj @{addr}")
        if not raw_cfg.strip():
            print(f"    [-] Nessun CFG trovato")
            continue

        try:
            cfg = json.loads(raw_cfg)
        except json.JSONDecodeError:
            print(f"    [-] JSON non valido per {name}")
            continue

        cfg_sorted = sort_cfg(cfg)

        out_file = os.path.join(OUTPUT_DIR, f"cfg_{addr:x}_{name}.json")

        with open(out_file, "w") as fp:
            json.dump(cfg_sorted, fp, indent=4)

        print(f"    [+] Salvato in {out_file}")

    r2.quit()


if __name__ == "__main__":
    main()
