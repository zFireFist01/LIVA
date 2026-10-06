#!/usr/bin/env python3
"""Safe extraction helpers for normal static archives.

GNU ``ar x`` overwrites members that share the same name.  That is rare in the
LIVA matrix, but losing one member would corrupt compilation-unit ground truth,
so duplicate occurrences are extracted individually with ``ar xN``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class ExtractedArchiveMember:
    name: str
    occurrence: int
    path: Path


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def extract_archive_members(
    archive: Path | str, destination: Path | str
) -> list[ExtractedArchiveMember]:
    """Extract every member, including duplicate-name occurrences."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    names = [name for name in _run(["ar", "t", archive.as_posix()], cwd=destination).splitlines() if name]
    counts = Counter(names)
    _run(["ar", "x", archive.as_posix()], cwd=destination)

    occurrences: defaultdict[str, int] = defaultdict(int)
    members: list[ExtractedArchiveMember] = []
    duplicate_root = destination / ".duplicate-members"
    for index, name in enumerate(names):
        occurrences[name] += 1
        occurrence = occurrences[name]
        extracted_name = Path(name).name
        if counts[name] == 1:
            member_path = destination / extracted_name
        else:
            if occurrence == 1:
                # ``ar x`` left only the last duplicate under the plain name;
                # every occurrence is re-extracted below, so discard that copy.
                (destination / extracted_name).unlink(missing_ok=True)
            occurrence_dir = duplicate_root / f"{index:06d}"
            occurrence_dir.mkdir(parents=True, exist_ok=True)
            _run(
                ["ar", "xN", str(occurrence), archive.as_posix(), name],
                cwd=occurrence_dir,
            )
            source = occurrence_dir / extracted_name
            member_path = destination / f".member-{index:06d}-{extracted_name}"
            shutil.move(source, member_path)
        if not member_path.is_file():
            raise RuntimeError(f"archive member was not extracted: {archive}!/{name}")
        members.append(ExtractedArchiveMember(name, occurrence, member_path))

    shutil.rmtree(duplicate_root, ignore_errors=True)
    return members
