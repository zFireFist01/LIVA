#!/usr/bin/env python3
# filepath: /home/paologennaro/Desktop/Tesi/FunctionMatching/script_log_batch.py
"""
Processa tutti i file .log il cui nome base è presente sia in exp_dataset
che in True_Lib_Exp, e salva l'output in una nuova cartella 'output_reports'.
"""

import os
import re
import sys
import io


# ── Percorsi ──────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LOG_DIR        = os.path.join(BASE_DIR, "log_tesi_rizzi", "output")
EXP_DATASET    = os.path.join(os.path.dirname(BASE_DIR),
                              "Exploration", "libseeker_repo", "exp_dataset")
TRUE_LIB_EXP   = os.path.join(os.path.dirname(BASE_DIR),
                              "Exploration", "libseeker_repo", "True_Lib_Exp")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output_reports")


# ── Parsing ───────────────────────────────────────────────────────────────
def parse_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_library = None
    libraries = {}

    for line in lines:
        lib_match = re.match(
            r'\t(/home/palm/exp1/libraries/\S+)\s+\.\.\.\s+(NO|YES)', line)
        if lib_match:
            lib_path = lib_match.group(1)
            lib_status = lib_match.group(2)
            current_library = lib_path
            libraries[current_library] = {
                'status': lib_status,
                'matched_cus': [],
                'unmatched_cus': []
            }
            continue

        if current_library:
            cu_match = re.match(
                r'\t\t(\S+\.o\S*)\s+\.\.\..*\[\s*(\d+\.\d+)\]\[\s*(\d+)\]',
                line)
            if cu_match:
                cu_name = cu_match.group(1)
                score = float(cu_match.group(2))
                matched = int(cu_match.group(3))
                if matched > 0:
                    libraries[current_library]['matched_cus'].append(
                        (cu_name, score, matched))
                else:
                    libraries[current_library]['unmatched_cus'].append(
                        (cu_name, score))

    return libraries


# ── Generazione report ────────────────────────────────────────────────────
def generate_report(libraries):
    """Restituisce la stessa stringa che prima veniva stampata a terminale."""
    buf = io.StringIO()

    matched_libs = {k: v for k, v in libraries.items() if v['status'] == 'YES'}
    unmatched_libs = {k: v for k, v in libraries.items() if v['status'] == 'NO'}

    # --- LIBRERIE IDENTIFICATE ---
    buf.write("=" * 80 + "\n")
    buf.write("LIBRERIE IDENTIFICATE (YES)\n")
    buf.write("=" * 80 + "\n")

    for lib, data in sorted(matched_libs.items()):
        lib_name = lib.split('/')[-1]
        n_matched = len(data['matched_cus'])
        n_unmatched = len(data['unmatched_cus'])
        total = n_matched + n_unmatched
        buf.write(f"\n✅ {lib_name}  ({n_matched}/{total} CU matchate)\n")
        buf.write("-" * 40 + "\n")
        for cu_name, score, count in data['matched_cus']:
            buf.write(f"   ✅ {cu_name:<45} (score: {score:.2f}, matched: {count})\n")
        for cu_name, score in data['unmatched_cus']:
            buf.write(f"   ❌ {cu_name:<45} (best score: {score:.2f})\n")

    # --- LIBRERIE NON IDENTIFICATE ---
    buf.write("\n" + "=" * 80 + "\n")
    buf.write("LIBRERIE NON IDENTIFICATE (NO)\n")
    buf.write("=" * 80 + "\n")

    for lib, data in sorted(unmatched_libs.items()):
        lib_name = lib.split('/')[-1]
        n_unmatched = len(data['unmatched_cus'])
        buf.write(f"\n❌ {lib_name}  ({n_unmatched} CU non matchate)\n")
        buf.write("-" * 40 + "\n")
        for cu_name, score in data['unmatched_cus']:
            buf.write(f"   ❌ {cu_name:<45} (best score: {score:.2f})\n")

    # --- RIEPILOGO ---
    buf.write("\n" + "=" * 80 + "\n")
    buf.write("RIEPILOGO\n")
    buf.write("=" * 80 + "\n")
    total_libs = len(libraries)
    total_matched_libs = len(matched_libs)
    total_unmatched_libs = len(unmatched_libs)
    total_matched_cus = sum(len(v['matched_cus']) for v in libraries.values())
    total_unmatched_cus = sum(len(v['unmatched_cus']) for v in libraries.values())
    total_cus = total_matched_cus + total_unmatched_cus
    buf.write(f"Librerie totali:           {total_libs}\n")
    buf.write(f"  - Identificate (YES):    {total_matched_libs}\n")
    buf.write(f"  - Non identificate (NO): {total_unmatched_libs}\n")
    buf.write(f"Compilation unit totali:   {total_cus}\n")
    buf.write(f"  - Matchate:              {total_matched_cus}\n")
    buf.write(f"  - Non matchate:          {total_unmatched_cus}\n")

    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # Nomi base presenti in True_Lib_Exp (es. "b2sum.gcc.O0")
    true_lib_names = set()
    for fname in os.listdir(TRUE_LIB_EXP):
        if fname.endswith('.txt'):
            true_lib_names.add(fname[:-4])          # rimuove ".txt"

    # Nomi base presenti in exp_dataset
    exp_dataset_names = set(os.listdir(EXP_DATASET))

    # Intersezione
    common = sorted(true_lib_names & exp_dataset_names)

    if not common:
        print("Nessun file in comune tra exp_dataset e True_Lib_Exp.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed = 0
    skipped = []
    for name in common:
        log_file = os.path.join(LOG_DIR, name + ".log")
        if not os.path.isfile(log_file):
            skipped.append(name)
            continue

        libraries = parse_log(log_file)
        report = generate_report(libraries)

        out_path = os.path.join(OUTPUT_DIR, name + ".txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)

        processed += 1
        print(f"  [✓] {name} → {out_path}")

    print(f"\nProcessati: {processed}/{len(common)}")
    if skipped:
        print(f"Saltati (log mancante): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
