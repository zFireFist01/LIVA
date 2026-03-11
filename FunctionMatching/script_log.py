#!/usr/bin/env python3
# filepath: /home/paologennaro/Desktop/Tesi/filter_unmatched.py

import sys
import re

def parse_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_library = None
    # Each library: { 'status': YES/NO, 'matched_cus': [...], 'unmatched_cus': [...] }
    libraries = {}
    
    for line in lines:
        # Match library line
        lib_match = re.match(r'\t(/home/palm/exp1/libraries/\S+)\s+\.\.\.\s+(NO|YES)', line)
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
        
        # Match compilation unit line
        if current_library:
            cu_match = re.match(r'\t\t(\S+\.o\S*)\s+\.\.\..*\[\s*(\d+\.\d+)\]\[\s*(\d+)\]', line)
            if cu_match:
                cu_name = cu_match.group(1)
                score = float(cu_match.group(2))
                matched = int(cu_match.group(3))
                if matched > 0:
                    libraries[current_library]['matched_cus'].append((cu_name, score, matched))
                else:
                    libraries[current_library]['unmatched_cus'].append((cu_name, score))
    
    return libraries

def main():
    if len(sys.argv) < 2:
        log_file = "/home/paologennaro/Desktop/Tesi/FunctionMatching/log_tesi_rizzi/output/b2sum.gcc.O0.log"
    else:
        log_file = sys.argv[1]
    
    libraries = parse_log(log_file)
    
    # Separate YES and NO libraries
    matched_libs = {k: v for k, v in libraries.items() if v['status'] == 'YES'}
    unmatched_libs = {k: v for k, v in libraries.items() if v['status'] == 'NO'}
    
    # --- IDENTIFIED LIBRARIES ---
    print("=" * 80)
    print("LIBRERIE IDENTIFICATE (YES)")
    print("=" * 80)
    
    for lib, data in sorted(matched_libs.items()):
        lib_name = lib.split('/')[-1]
        n_matched = len(data['matched_cus'])
        n_unmatched = len(data['unmatched_cus'])
        total = n_matched + n_unmatched
        print(f"\n✅ {lib_name}  ({n_matched}/{total} CU matchate)")
        print("-" * 40)
        for cu_name, score, count in data['matched_cus']:
            print(f"   ✅ {cu_name:<45} (score: {score:.2f}, matched: {count})")
        for cu_name, score in data['unmatched_cus']:
            print(f"   ❌ {cu_name:<45} (best score: {score:.2f})")
    
    # --- UNIDENTIFIED LIBRARIES ---
    print("\n" + "=" * 80)
    print("LIBRERIE NON IDENTIFICATE (NO)")
    print("=" * 80)
    
    for lib, data in sorted(unmatched_libs.items()):
        lib_name = lib.split('/')[-1]
        n_unmatched = len(data['unmatched_cus'])
        print(f"\n❌ {lib_name}  ({n_unmatched} CU non matchate)")
        print("-" * 40)
        for cu_name, score in data['unmatched_cus']:
            print(f"   ❌ {cu_name:<45} (best score: {score:.2f})")
    
    # --- SUMMARY ---
    print("\n" + "=" * 80)
    print("RIEPILOGO")
    print("=" * 80)
    total_libs = len(libraries)
    total_matched_libs = len(matched_libs)
    total_unmatched_libs = len(unmatched_libs)
    total_matched_cus = sum(len(v['matched_cus']) for v in libraries.values())
    total_unmatched_cus = sum(len(v['unmatched_cus']) for v in libraries.values())
    total_cus = total_matched_cus + total_unmatched_cus
    print(f"Librerie totali:           {total_libs}")
    print(f"  - Identificate (YES):    {total_matched_libs}")
    print(f"  - Non identificate (NO): {total_unmatched_libs}")
    print(f"Compilation unit totali:   {total_cus}")
    print(f"  - Matchate:              {total_matched_cus}")
    print(f"  - Non matchate:          {total_unmatched_cus}")

if __name__ == "__main__":
    main()