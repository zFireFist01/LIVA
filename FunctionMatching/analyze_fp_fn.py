#!/usr/bin/env python3
"""
Analisi False Positive / False Negative tra ground truth e predizioni.

Ground truth (.txt): elenco di librerie realmente incluse nel binario, con le
                     compilation unit (.o) e i simboli corrispondenti.
Predizioni (.log):   output del sistema di function matching (tesi Rizzi), con
                     la classificazione YES/NO per ogni libreria e i dettagli
                     delle corrispondenze funzione-per-funzione.

Categorie prodotte:
  - TP  (True Positive):    predetto YES, effettivamente presente
  - FP  (False Positive):   predetto YES, ma NON presente
  - FN  (False Negative):   predetto NO, ma presente
  - TN  (True Negative):    predetto NO, effettivamente assente
  - TP wrong reason:        libreria correttamente YES, ma le CU matchate
                            non corrispondono a quelle reali del GT
  - Librerie GT non valutate: presenti nel GT ma non nel set del log
"""

import re
import os
import sys
import argparse
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

def extract_base_lib_name(full_name: str) -> str:
    """
    Estrae il nome base da un nome di libreria con versione.
    Es: 'libc.a.2.40' -> 'libc', 'libX11.a.11-1.8.7' -> 'libX11'
    """
    m = re.match(r'^(.+?)\.a\.', full_name)
    if m:
        return m.group(1)
    return full_name


def parse_ground_truth(filepath: str) -> dict:
    """
    Parsa il file di ground truth.
    Ritorna un dict:
      { base_lib_name: {
            'versions': { version_name: {
                'cu_count': int,
                'cus': { cu_name: { 'matched': int, 'total': int, 'symbols': [str] } }
            }},
            'all_cus': set(),     # unione di tutte le CU (su tutte le versioni)
            'all_symbols': set()  # unione di tutti i simboli
        }
      }
    """
    gt = {}
    current_lib = None
    current_version = None
    current_cu = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # LIBRARY: libX11.a.11-1.8.7
            m = re.match(r'^LIBRARY:\s+(.+)', line)
            if m:
                full_name = m.group(1).strip()
                base_name = extract_base_lib_name(full_name)
                current_version = full_name
                current_cu = None

                if base_name not in gt:
                    gt[base_name] = {
                        'versions': {},
                        'all_cus': set(),
                        'all_symbols': set()
                    }
                gt[base_name]['versions'][full_name] = {
                    'cu_count': 0,
                    'cus': {}
                }
                current_lib = base_name
                continue

            if current_lib is None:
                continue

            # Included compilation units: N
            m = re.match(r'^\s+Included compilation units:\s+(\d+)', line)
            if m:
                gt[current_lib]['versions'][current_version]['cu_count'] = int(m.group(1))
                continue

            # CU line:  obj_file.o  (X/Y symbols matched)
            m = re.match(r'^\s{4}(\S+)\s+\((\d+)/(\d+)\s+symbols?\s+matched\)', line)
            if m:
                cu_name = m.group(1)
                matched = int(m.group(2))
                total = int(m.group(3))
                current_cu = cu_name
                gt[current_lib]['versions'][current_version]['cus'][cu_name] = {
                    'matched': matched,
                    'total': total,
                    'symbols': []
                }
                gt[current_lib]['all_cus'].add(normalize_cu_name(cu_name))
                continue

            # Symbol line:  - symbol_name
            m = re.match(r'^\s{6}-\s+(.+)', line)
            if m and current_cu:
                sym = m.group(1).strip()
                gt[current_lib]['versions'][current_version]['cus'][current_cu]['symbols'].append(sym)
                gt[current_lib]['all_symbols'].add(sym)
                continue

    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING LOG (PREDIZIONI)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_cu_name(name: str) -> str:
    """Normalizza il nome di una compilation unit per il confronto.
    Rimuove suffissi come .oS, .os e lascia solo il basename senza estensione."""
    # Rimuovi estensione .o, .oS, .os, .c.o, etc.
    name = re.sub(r'\.(oS|os|o)$', '', name)
    return name


def parse_log(filepath: str) -> dict:
    """
    Parsa il file di log delle predizioni.
    Ritorna un dict:
      { lib_base_name: {
            'prediction': 'YES' | 'NO',
            'has_warning': bool,
            'score': float,
            'matched_cu_count': int,
            'total_cu_count': int,
            'total_matched_funcs': int,
            'cus': {
                cu_name: {
                    'is_matched': bool,   # ha [W]
                    'score': float,
                    'matched_count': int,
                    'total_count': int,
                    'matches': [ (binary_func, lib_func, score) ]
                }
            }
        }
      }
    """
    log = {}
    current_lib = None
    current_cu = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # Library line: /home/palm/exp1/libraries/<name> ... YES/NO [W] [score] [m/t] [n] [time]
            m = re.match(
                r'^\t(/home/palm/exp1/libraries/(\S+))\s+\.{3}\s+'
                r'(YES|NO)\s*(\[W\])?\s*'
                r'\[([^\]]*)\]\s*'        # score
                r'\[\s*(\d+)/\s*(\d+)\]\s*'  # matched/total CU
                r'\[\s*(\d+)\]\s*'         # total matched funcs
                r'\[([^\]]*)\]',           # time
                line
            )
            if m:
                lib_name = m.group(2)
                current_lib = lib_name
                current_cu = None
                log[lib_name] = {
                    'prediction': m.group(3),
                    'has_warning': m.group(4) is not None,
                    'score': float(m.group(5)),
                    'matched_cu_count': int(m.group(6)),
                    'total_cu_count': int(m.group(7)),
                    'total_matched_funcs': int(m.group(8)),
                    'cus': {}
                }
                continue

            if current_lib is None:
                continue

            # CU line: \t\tcu_name.o  ... [W] [score][matched][total]
            # oppure: \t\tcu_name.o  ...     [score][matched][total]
            m = re.match(
                r'^\t\t(\S+)\s+\.{3}\s+(\[W\])?\s*'
                r'\[([^\]]*)\]\[\s*(\d+)\]\[\s*(\d+)\]',
                line
            )
            if m:
                cu_name = m.group(1)
                current_cu = cu_name
                log[current_lib]['cus'][cu_name] = {
                    'is_matched': m.group(2) is not None,
                    'score': float(m.group(3)),
                    'matched_count': int(m.group(4)),
                    'total_count': int(m.group(5)),
                    'matches': []
                }
                continue

            # Function match line: \t\t\tbinary_func ---> lib_func ... [score]
            m = re.match(
                r'^\t\t\t(.+?)\s+--->\s+(.+?)\s+\.{3}\s+\[([^\]]+)\]',
                line
            )
            if m and current_cu:
                bin_func = m.group(1).strip()
                lib_func = m.group(2).strip()
                try:
                    score = float(m.group(3))
                except ValueError:
                    score = 0.0
                log[current_lib]['cus'][current_cu]['matches'].append(
                    (bin_func, lib_func, score)
                )
                continue

    return log


# ═══════════════════════════════════════════════════════════════════════════════
# CONFRONTO E ANALISI
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(gt: dict, log: dict) -> dict:
    """
    Confronta ground truth e predizioni.
    Ritorna un dict con le categorie TP, FP, FN, TN, TP_wrong_reason,
    gt_not_in_log.
    """
    gt_libs = set(gt.keys())
    log_libs = set(log.keys())

    results = {
        'TP': [],         # True Positive (corretto YES)
        'FP': [],         # False Positive (predetto YES ma non presente)
        'FN': [],         # False Negative (predetto NO ma presente)
        'TN': [],         # True Negative (corretto NO)
        'TP_wrong_reason': [],  # YES corretto ma CU sbagliate
        'TP_right_reason': [],  # YES corretto con CU corrette
        'gt_not_in_log': [],    # Librerie GT non valutate dal sistema
    }

    # Librerie nel GT ma non nel log
    gt_only = gt_libs - log_libs
    for lib in sorted(gt_only):
        results['gt_not_in_log'].append({
            'lib': lib,
            'gt_cus': gt[lib]['all_cus'],
            'gt_symbols': gt[lib]['all_symbols'],
            'versions': list(gt[lib]['versions'].keys())
        })

    # Analisi per ogni libreria nel log
    for lib_name in sorted(log_libs):
        pred = log[lib_name]['prediction']
        is_in_gt = lib_name in gt_libs

        entry = {
            'lib': lib_name,
            'prediction': pred,
            'in_gt': is_in_gt,
            'log_score': log[lib_name]['score'],
            'log_matched_cus': log[lib_name]['matched_cu_count'],
            'log_total_cus': log[lib_name]['total_cu_count'],
            'log_total_matched_funcs': log[lib_name]['total_matched_funcs'],
            'has_warning': log[lib_name]['has_warning'],
        }

        if pred == 'YES' and is_in_gt:
            # True Positive - analizziamo anche se la ragione è corretta
            entry.update(_analyze_tp_reason(gt[lib_name], log[lib_name]))
            results['TP'].append(entry)

            if entry['wrong_cus'] or entry['missing_cus']:
                results['TP_wrong_reason'].append(entry)
            else:
                results['TP_right_reason'].append(entry)

        elif pred == 'YES' and not is_in_gt:
            # False Positive
            entry['matched_cus_detail'] = _get_matched_cus_from_log(log[lib_name])
            results['FP'].append(entry)

        elif pred == 'NO' and is_in_gt:
            # False Negative
            entry['gt_cus'] = sorted(gt[lib_name]['all_cus'])
            entry['gt_symbols_count'] = len(gt[lib_name]['all_symbols'])
            entry['versions'] = list(gt[lib_name]['versions'].keys())
            # Mostra anche i migliori match (score più alti) dal log
            entry['best_matches'] = _get_best_matches_from_log(log[lib_name])
            results['FN'].append(entry)

        else:
            # True Negative
            # Controlla se ci sono match alti che avrebbero potuto ingannare
            entry['best_matches'] = _get_best_matches_from_log(log[lib_name], top_n=3)
            results['TN'].append(entry)

    return results


def _analyze_tp_reason(gt_entry: dict, log_entry: dict) -> dict:
    """Analizza se un TP ha la ragione corretta (CU giuste)."""
    # CU matchate nel log (quelle con [W])
    log_matched_cus = set()
    log_matched_cus_detail = {}
    for cu_name, cu_data in log_entry['cus'].items():
        if cu_data['is_matched']:
            norm_name = normalize_cu_name(cu_name)
            log_matched_cus.add(norm_name)
            log_matched_cus_detail[norm_name] = {
                'original_name': cu_name,
                'score': cu_data['score'],
                'matches': cu_data['matches']
            }

    # CU nel ground truth (unione di tutte le versioni)
    gt_cus = gt_entry['all_cus']

    # CU matchate correttamente (presenti sia nel log che nel GT)
    correct_cus = log_matched_cus & gt_cus
    # CU matchate nel log ma non nel GT (match errati)
    wrong_cus = log_matched_cus - gt_cus
    # CU nel GT ma non matchate nel log (mancate)
    missing_cus = gt_cus - log_matched_cus

    # Per le CU matchate, verifica anche se le funzioni sono corrette
    correct_func_matches = []
    wrong_func_matches = []
    gt_symbols = gt_entry['all_symbols']

    for cu_norm in correct_cus:
        if cu_norm in log_matched_cus_detail:
            detail = log_matched_cus_detail[cu_norm]
            for bin_func, lib_func, score in detail['matches']:
                # Un match è corretto se la funzione binaria è un simbolo del GT
                if bin_func in gt_symbols:
                    correct_func_matches.append((bin_func, lib_func, score, cu_norm))
                else:
                    wrong_func_matches.append((bin_func, lib_func, score, cu_norm))

    return {
        'log_matched_cus': sorted(log_matched_cus),
        'gt_cus': sorted(gt_cus),
        'correct_cus': sorted(correct_cus),
        'wrong_cus': sorted(wrong_cus),
        'missing_cus': sorted(missing_cus),
        'correct_func_matches': correct_func_matches,
        'wrong_func_matches': wrong_func_matches,
        'log_matched_cus_detail': log_matched_cus_detail,
    }


def _get_matched_cus_from_log(log_entry: dict) -> list:
    """Ritorna le CU matchate (con [W]) per un entry del log."""
    result = []
    for cu_name, cu_data in log_entry['cus'].items():
        if cu_data['is_matched']:
            result.append({
                'cu': cu_name,
                'score': cu_data['score'],
                'matched_count': cu_data['matched_count'],
                'matches': cu_data['matches']
            })
    return result


def _get_best_matches_from_log(log_entry: dict, top_n: int = 5) -> list:
    """Ritorna i migliori match (score più alti) da tutte le CU."""
    all_matches = []
    for cu_name, cu_data in log_entry['cus'].items():
        for bin_func, lib_func, score in cu_data['matches']:
            all_matches.append({
                'cu': cu_name,
                'bin_func': bin_func,
                'lib_func': lib_func,
                'score': score
            })
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    return all_matches[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT / REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(results: dict, output_path: str = None):
    """Genera il report in formato testo."""
    lines = []

    def w(s=''):
        lines.append(s)

    w('=' * 100)
    w('REPORT ANALISI FALSE POSITIVE / FALSE NEGATIVE')
    w('=' * 100)
    w()

    # ── Riepilogo ──
    w('─' * 100)
    w('RIEPILOGO')
    w('─' * 100)
    tp_count = len(results['TP'])
    fp_count = len(results['FP'])
    fn_count = len(results['FN'])
    tn_count = len(results['TN'])
    tp_wrong = len(results['TP_wrong_reason'])
    tp_right = len(results['TP_right_reason'])
    gt_not_eval = len(results['gt_not_in_log'])

    total_pred = tp_count + fp_count + fn_count + tn_count
    w(f'  Librerie valutate:           {total_pred}')
    w(f'  True Positive (TP):          {tp_count}')
    w(f'    - di cui ragione corretta: {tp_right}')
    w(f'    - di cui ragione errata:   {tp_wrong}')
    w(f'  False Positive (FP):         {fp_count}')
    w(f'  False Negative (FN):         {fn_count}')
    w(f'  True Negative (TN):          {tn_count}')
    w(f'  Librerie GT non valutate:    {gt_not_eval}')
    w()

    if total_pred > 0:
        accuracy = (tp_count + tn_count) / total_pred
        w(f'  Accuracy (lib-level):        {accuracy:.4f}')
    if tp_count + fp_count > 0:
        precision = tp_count / (tp_count + fp_count)
        w(f'  Precision:                   {precision:.4f}')
    if tp_count + fn_count > 0:
        recall = tp_count / (tp_count + fn_count)
        w(f'  Recall:                      {recall:.4f}')
    if tp_count + fp_count > 0 and tp_count + fn_count > 0:
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            w(f'  F1 Score:                    {f1:.4f}')
    w()

    # ── FALSE POSITIVE ──
    w('=' * 100)
    w(f'FALSE POSITIVE ({fp_count})')
    w('Librerie predette YES ma NON presenti nel ground truth')
    w('=' * 100)
    for entry in results['FP']:
        w()
        w(f'  📛 {entry["lib"]}')
        w(f'     Prediction: YES | Score: {entry["log_score"]:.2f} | '
          f'CU matchate: {entry["log_matched_cus"]}/{entry["log_total_cus"]} | '
          f'Funzioni matchate: {entry["log_total_matched_funcs"]} | '
          f'Warning: {entry["has_warning"]}')
        if entry['matched_cus_detail']:
            w(f'     CU matchate dal sistema:')
            for cu_info in entry['matched_cus_detail']:
                w(f'       - {cu_info["cu"]}  (score: {cu_info["score"]:.2f}, '
                  f'matched: {cu_info["matched_count"]})')
                for bin_f, lib_f, sc in cu_info['matches']:
                    marker = ' ✓' if sc >= 0.95 else ''
                    w(f'           {bin_f:50s} ---> {lib_f:50s} [{sc:.2f}]{marker}')
    w()

    # ── FALSE NEGATIVE ──
    w('=' * 100)
    w(f'FALSE NEGATIVE ({fn_count})')
    w('Librerie predette NO ma presenti nel ground truth')
    w('=' * 100)
    for entry in results['FN']:
        w()
        w(f'  ❌ {entry["lib"]}')
        w(f'     Prediction: NO | Score: {entry["log_score"]:.2f} | '
          f'CU nel GT: {len(entry["gt_cus"])} | '
          f'Simboli GT: {entry["gt_symbols_count"]}')
        w(f'     Versioni GT: {", ".join(entry["versions"])}')
        w(f'     CU presenti nel GT: {", ".join(entry["gt_cus"][:20])}')
        if len(entry["gt_cus"]) > 20:
            w(f'       ... e altre {len(entry["gt_cus"]) - 20}')
        if entry['best_matches']:
            w(f'     Migliori match trovati dal sistema (ma non sufficienti):')
            for m in entry['best_matches']:
                w(f'       {m["cu"]:40s} | {m["bin_func"]:40s} ---> {m["lib_func"]:40s} [{m["score"]:.2f}]')
    w()

    # ── TRUE POSITIVE ──
    w('=' * 100)
    w(f'TRUE POSITIVE ({tp_count})')
    w('Librerie correttamente predette YES')
    w('=' * 100)

    # Prima quelli con ragione errata
    if results['TP_wrong_reason']:
        w()
        w(f'  --- TP CON RAGIONE ERRATA/PARZIALE ({tp_wrong}) ---')
        w(f'  (CU matchate dal sistema non corrispondono completamente al GT)')
        for entry in results['TP_wrong_reason']:
            w()
            w(f'  ⚠️  {entry["lib"]}')
            w(f'     Score: {entry["log_score"]:.2f} | '
              f'CU matchate (log): {entry["log_matched_cus"]} | '
              f'CU matchate (GT): {len(entry["gt_cus"])}')
            w(f'     CU corrette: {len(entry["correct_cus"])}  |  '
              f'CU errate (nel log ma non GT): {len(entry["wrong_cus"])}  |  '
              f'CU mancanti (nel GT ma non log): {len(entry["missing_cus"])}')

            if entry['wrong_cus']:
                w(f'     CU matchate erroneamente:')
                for cu in entry['wrong_cus'][:15]:
                    if cu in entry['log_matched_cus_detail']:
                        detail = entry['log_matched_cus_detail'][cu]
                        w(f'       - {detail["original_name"]:40s} (score: {detail["score"]:.2f})')
                        for bf, lf, sc in detail['matches'][:3]:
                            w(f'           {bf:50s} ---> {lf:50s} [{sc:.2f}]')
                    else:
                        w(f'       - {cu}')
                if len(entry['wrong_cus']) > 15:
                    w(f'       ... e altre {len(entry["wrong_cus"]) - 15}')

            if entry['missing_cus']:
                w(f'     CU nel GT ma non matchate dal sistema:')
                for cu in entry['missing_cus'][:20]:
                    w(f'       - {cu}')
                if len(entry['missing_cus']) > 20:
                    w(f'       ... e altre {len(entry["missing_cus"]) - 20}')

            if entry['wrong_func_matches']:
                w(f'     Match funzionali errati (funzione binaria non nel GT):')
                for bf, lf, sc, cu in entry['wrong_func_matches'][:10]:
                    w(f'       [{cu}] {bf:50s} ---> {lf:50s} [{sc:.2f}]')
                if len(entry['wrong_func_matches']) > 10:
                    w(f'       ... e altri {len(entry["wrong_func_matches"]) - 10}')

    # Poi quelli con ragione corretta (più breve)
    if results['TP_right_reason']:
        w()
        w(f'  --- TP CON RAGIONE CORRETTA ({tp_right}) ---')
        for entry in results['TP_right_reason']:
            w(f'  ✅ {entry["lib"]:40s}  score: {entry["log_score"]:.2f}  '
              f'CU matchate: {len(entry["correct_cus"])}/{len(entry["gt_cus"])}')
    w()

    # ── TRUE NEGATIVE (solo riepilogo) ──
    w('=' * 100)
    w(f'TRUE NEGATIVE ({tn_count})')
    w('Librerie correttamente predette NO')
    w('=' * 100)
    # Mostra i TN con match più alti (potenziali problemi)
    tn_with_high_scores = []
    for entry in results['TN']:
        if entry['best_matches'] and entry['best_matches'][0]['score'] >= 0.90:
            tn_with_high_scores.append(entry)

    if tn_with_high_scores:
        w()
        w(f'  TN con match alti (score >= 0.90, potenziali futuri FP):')
        for entry in tn_with_high_scores:
            top = entry['best_matches'][0]
            w(f'  ⚡ {entry["lib"]:40s}  miglior score: {top["score"]:.2f}  '
              f'({top["bin_func"]} ---> {top["lib_func"]})')
    w()
    w(f'  Tutti i TN ({tn_count}):')
    for entry in results['TN']:
        top_score = entry['best_matches'][0]['score'] if entry['best_matches'] else 0.0
        w(f'    {entry["lib"]:50s}  miglior score: {top_score:.2f}')
    w()

    # ── LIBRERIE GT NON VALUTATE ──
    if results['gt_not_in_log']:
        w('=' * 100)
        w(f'LIBRERIE GT NON VALUTATE ({gt_not_eval})')
        w('Librerie presenti nel ground truth ma non nel set di librerie del log')
        w('=' * 100)
        for entry in results['gt_not_in_log']:
            w(f'  ❓ {entry["lib"]:40s}  CU: {len(entry["gt_cus"])}  '
              f'Simboli: {len(entry["gt_symbols"])}  '
              f'Versioni: {", ".join(entry["versions"])}')
        w()

    report = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f'Report salvato in: {output_path}')
    else:
        print(report)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH: processa tutti i file disponibili
# ═══════════════════════════════════════════════════════════════════════════════

def find_matching_files(gt_dir: str, log_dir: str) -> list:
    """Trova coppie (gt_file, log_file) con lo stesso nome base."""
    gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                for f in os.listdir(gt_dir) if f.endswith('.txt')}
    log_files = {os.path.splitext(f)[0]: os.path.join(log_dir, f)
                 for f in os.listdir(log_dir) if f.endswith('.log')}

    pairs = []
    for name in sorted(set(gt_files.keys()) & set(log_files.keys())):
        pairs.append((gt_files[name], log_files[name], name))

    return pairs


def batch_analyze(gt_dir: str, log_dir: str, output_dir: str):
    """Analizza tutti i file corrispondenti e genera report."""
    pairs = find_matching_files(gt_dir, log_dir)
    if not pairs:
        print('Nessuna coppia GT/log trovata.')
        return

    os.makedirs(output_dir, exist_ok=True)

    # Riepilogo globale
    global_stats = {
        'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
        'TP_wrong': 0, 'TP_right': 0, 'gt_not_eval': 0
    }

    summary_lines = []
    summary_lines.append('=' * 120)
    summary_lines.append('RIEPILOGO GLOBALE ANALISI FP/FN')
    summary_lines.append('=' * 120)
    summary_lines.append('')
    summary_lines.append(f'{"Binary":<30s} {"TP":>4s} {"FP":>4s} {"FN":>4s} {"TN":>4s} '
                         f'{"TP_ok":>5s} {"TP_wr":>5s} {"GT_na":>5s} '
                         f'{"Prec":>6s} {"Rec":>6s} {"F1":>6s}')
    summary_lines.append('-' * 120)

    for gt_path, log_path, name in pairs:
        print(f'Analisi: {name} ...')
        gt = parse_ground_truth(gt_path)
        log = parse_log(log_path)
        results = analyze(gt, log)

        # Report singolo
        out_file = os.path.join(output_dir, f'{name}.analysis.txt')
        generate_report(results, out_file)

        # Statistiche
        tp = len(results['TP'])
        fp = len(results['FP'])
        fn = len(results['FN'])
        tn = len(results['TN'])
        tp_w = len(results['TP_wrong_reason'])
        tp_r = len(results['TP_right_reason'])
        gt_na = len(results['gt_not_in_log'])

        global_stats['TP'] += tp
        global_stats['FP'] += fp
        global_stats['FN'] += fn
        global_stats['TN'] += tn
        global_stats['TP_wrong'] += tp_w
        global_stats['TP_right'] += tp_r
        global_stats['gt_not_eval'] += gt_na

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        summary_lines.append(
            f'{name:<30s} {tp:>4d} {fp:>4d} {fn:>4d} {tn:>4d} '
            f'{tp_r:>5d} {tp_w:>5d} {gt_na:>5d} '
            f'{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}'
        )

    # Totali globali
    summary_lines.append('-' * 120)
    tp = global_stats['TP']
    fp = global_stats['FP']
    fn = global_stats['FN']
    tn = global_stats['TN']
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    summary_lines.append(
        f'{"TOTALE":<30s} {tp:>4d} {fp:>4d} {fn:>4d} {tn:>4d} '
        f'{global_stats["TP_right"]:>5d} {global_stats["TP_wrong"]:>5d} '
        f'{global_stats["gt_not_eval"]:>5d} '
        f'{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}'
    )
    summary_lines.append('')

    summary = '\n'.join(summary_lines)
    summary_path = os.path.join(output_dir, '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print()
    print(summary)
    print(f'\nRiepilogo salvato in: {summary_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Analisi FP/FN tra ground truth e predizioni function matching'
    )
    parser.add_argument('--gt', type=str, help='File ground truth (.txt) o directory')
    parser.add_argument('--log', type=str, help='File log predizioni (.log) o directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='File o directory di output per il report')
    parser.add_argument('--batch', action='store_true',
                        help='Modalità batch: processa tutti i file nelle directory')

    args = parser.parse_args()

    # Default paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_gt_dir = os.path.join(base_dir, '..', 'Exploration', 'libseeker_repo', 'True_Lib_Exp')
    default_log_dir = os.path.join(base_dir, 'log_tesi_rizzi', 'output')
    default_output_dir = os.path.join(base_dir, 'analysis_output')

    if args.batch:
        gt_dir = args.gt or default_gt_dir
        log_dir = args.log or default_log_dir
        output_dir = args.output or default_output_dir
        batch_analyze(gt_dir, log_dir, output_dir)
    else:
        gt_file = args.gt or os.path.join(default_gt_dir, 'b2sum.gcc.O0.txt')
        log_file = args.log or os.path.join(default_log_dir, 'b2sum.gcc.O0.log')

        if not os.path.isfile(gt_file):
            print(f'File GT non trovato: {gt_file}')
            sys.exit(1)
        if not os.path.isfile(log_file):
            print(f'File log non trovato: {log_file}')
            sys.exit(1)

        print(f'Ground truth: {gt_file}')
        print(f'Predizioni:   {log_file}')
        print()

        gt = parse_ground_truth(gt_file)
        log = parse_log(log_file)
        results = analyze(gt, log)
        generate_report(results, args.output)


if __name__ == '__main__':
    main()
