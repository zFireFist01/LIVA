#!/usr/bin/env python3
import argparse
import csv
import re
import signal
import subprocess
import sys
from pathlib import Path


SECTION_RE = re.compile(
    r"^\s*\[\s*\d+\]\s+"
    r"(?P<name>\S+)\s+"
    r"(?P<type>\S+)\s+"
    r"(?P<addr>[0-9a-fA-F]+)\s+"
    r"(?P<off>[0-9a-fA-F]+)\s+"
    r"(?P<size>[0-9a-fA-F]+)\s+"
)
FILE_RE = re.compile(r"^File:\s+.*\((?P<object>[^()]+)\)\s*$")


def run_readelf(path: Path) -> str:
    try:
        result = subprocess.run(
            ["readelf", "-S", "--wide", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("Errore: readelf non trovato nel PATH.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        return ""
    return result.stdout


def rodata_sections(path: Path):
    current_object = path.name

    for line in run_readelf(path).splitlines():
        file_match = FILE_RE.match(line)
        if file_match:
            current_object = file_match.group("object")
            continue

        section_match = SECTION_RE.match(line)
        if not section_match:
            continue

        name = section_match.group("name")
        if name != ".rodata" and not name.startswith(".rodata."):
            continue

        size = int(section_match.group("size"), 16)
        if size == 0:
            continue

        yield {
            "library": path.name,
            "object": current_object,
            "section": name,
            "size": size,
        }


def library_rows(libraries):
    for lib in libraries:
        sections = list(rodata_sections(lib))
        exact_sections = [s for s in sections if s["section"] == ".rodata"]
        prefix_sections = sections

        yield {
            "library": lib.name,
            "exact_rodata_bytes": sum(s["size"] for s in exact_sections),
            "exact_rodata_sections": len(exact_sections),
            "all_rodata_bytes": sum(s["size"] for s in prefix_sections),
            "all_rodata_sections": len(prefix_sections),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calcola la dimensione di .rodata e delle sottosezioni .rodata.* "
            "per le librerie statiche in all_libs."
        )
    )
    parser.add_argument(
        "library_dir",
        nargs="?",
        default="Exploration/libseeker_repo/build_lib/all_libs",
        help="Directory che contiene le librerie da analizzare.",
    )
    parser.add_argument(
        "--per-section",
        action="store_true",
        help="Stampa una riga per ogni sezione .rodata/.rodata.* trovata.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Scrive il TSV in un file invece che su stdout.",
    )
    return parser.parse_args()


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    args = parse_args()
    library_dir = Path(args.library_dir)

    if not library_dir.is_dir():
        print(f"Errore: directory non trovata: {library_dir}", file=sys.stderr)
        return 1

    libraries = sorted(path for path in library_dir.iterdir() if path.is_file())
    output = open(args.output, "w", newline="") if args.output else sys.stdout

    try:
        if args.per_section:
            fieldnames = ["library", "object", "section", "size"]
            rows = (section for lib in libraries for section in rodata_sections(lib))
        else:
            fieldnames = [
                "library",
                "exact_rodata_bytes",
                "exact_rodata_sections",
                "all_rodata_bytes",
                "all_rodata_sections",
            ]
            rows = library_rows(libraries)

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
