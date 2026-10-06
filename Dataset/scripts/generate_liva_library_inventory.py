#!/usr/bin/env python3
"""Generate the human-readable LIVA static-library inventory."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


OPTIMIZATION_ORDER = {"O0": 0, "O2": 1, "O3": 2, "Os": 3}
ROLE_LABELS = {
    "current": "corrente",
    "minor-alternative": "alternativa-minor",
    "major-alternative": "alternativa-major",
}
ROLE_ORDER = {"current": 0, "minor-alternative": 1, "major-alternative": 2}


def project_versions(source_manifest: dict) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for package in source_manifest["historical_packages"]:
        package_name = package["name"]
        for version in package["versions"]:
            label = version["label"]
            covered_by = version.get("covered_by")
            if covered_by:
                result[package_name][covered_by] = label
            for value in (version.get("version"), version.get("label")):
                if value:
                    value = value.split(":")[-1]
                    result[package_name][f"{package_name}-{value}"] = label
    return result


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    matrix_path = repository / "Dataset/manifests/library_matrix.tsv"
    summary_path = repository / "Dataset/manifests/library_matrix_summary.json"
    sources_path = repository / "Dataset/manifests/source_manifest.json"
    output_path = repository / "librerie_LIVA.txt"

    with matrix_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    version_map = project_versions(sources)

    archive_order = list(dict.fromkeys(row["archive"] for row in rows))
    selected = [row for row in rows if row["status"] == "selected"]
    selected_by_archive: dict[str, list[dict]] = defaultdict(list)
    missing_by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        destination = selected_by_archive if row["status"] == "selected" else missing_by_archive
        destination[row["archive"]].append(row)

    used_names = [name for name in archive_order if selected_by_archive[name]]
    unused_names = [name for name in archive_order if not selected_by_archive[name]]
    toolchains = list(dict.fromkeys(row["toolchain"] for row in selected))
    optimizations = sorted(
        {row["optimization"] for row in selected},
        key=lambda item: OPTIMIZATION_ORDER[item],
    )

    lines = [
        "LIVA - LIBRERIE STATICHE, VERSIONI, COMPILATORI E OTTIMIZZAZIONI",
        "",
        "Fonte: Dataset/manifests/library_matrix.tsv dopo il pruning.",
        f"Nomi di archivio richiesti: {summary['requested_archive_names']}",
        f"Nomi di archivio effettivamente usati: {len(used_names)}",
        f"Varianti effettivamente usate: {len(selected)}",
        f"Combinazioni non disponibili: {summary['missing_coordinates']}",
        f"Compilatori presenti: {', '.join(toolchains)}",
        f"Ottimizzazioni presenti: {', '.join(optimizations)}",
        "",
        "Formato:",
        "nome [pacchetto; numero varianti]",
        "  ruolo | versione | directory progetto: compilatore={ottimizzazioni}; ...",
        "",
        "Sono elencate soltanto le combinazioni realmente presenti e utilizzabili da LIVA.",
        "",
    ]

    for archive_name in used_names:
        archive_rows = selected_by_archive[archive_name]
        package = archive_rows[0]["package"]
        grouped: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for row in archive_rows:
            parts = Path(row["path"]).parts
            project = parts[1]
            version = version_map.get(package, {}).get(project, project)
            grouped[(row["role"], version, project)][row["toolchain"]].add(
                row["optimization"]
            )

        lines.append(f"{archive_name} [pacchetto={package}; varianti={len(archive_rows)}]")
        for (role, version, project), compiler_opts in sorted(
            grouped.items(),
            key=lambda item: (ROLE_ORDER[item[0][0]], item[0][1], item[0][2]),
        ):
            compiler_parts = []
            for compiler in toolchains:
                if compiler not in compiler_opts:
                    continue
                opts = sorted(compiler_opts[compiler], key=lambda item: OPTIMIZATION_ORDER[item])
                compiler_parts.append(f"{compiler}={{{','.join(opts)}}}")
            lines.append(
                f"  {ROLE_LABELS[role]} | versione={version} | progetto={project}: "
                + "; ".join(compiler_parts)
            )
        lines.append("")

    lines.extend(
        [
            "ARCHIVI NON USATI (NESSUNA VARIANTE DISPONIBILE)",
            "",
        ]
    )
    for archive_name in unused_names:
        archive_rows = missing_by_archive[archive_name]
        package = archive_rows[0]["package"]
        reasons = Counter(row["reason"] for row in archive_rows)
        reason_text = ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        lines.append(f"{archive_name} [pacchetto={package}]: {reason_text}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Creato {output_path}: {len(used_names)} archivi usati, "
        f"{len(selected)} varianti, {len(unused_names)} archivi non disponibili."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
