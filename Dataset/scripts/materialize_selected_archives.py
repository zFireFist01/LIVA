#!/usr/bin/env python3
"""Make every selected LIVA .a path a self-contained GNU archive."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


AR_MAGIC = b"!<arch>\n"
THIN_MAGIC = b"!<thin>\n"
ELF_MAGIC = b"\x7fELF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None, stdin: str | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def normal_archive_from_members(
    destination: Path,
    members: list[Path],
    *,
    cwd: Path,
) -> Path:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.materializing")
    temporary.unlink(missing_ok=True)
    try:
        if not members:
            run(["ar", "qc", temporary.as_posix()], cwd=cwd)
        for start in range(0, len(members), 100):
            chunk = members[start:start + 100]
            run(
                ["ar", "qc", temporary.as_posix(), *(path.as_posix() for path in chunk)],
                cwd=cwd,
            )
        run(["ar", "s", temporary.as_posix()])
        if temporary.open("rb").read(8) != AR_MAGIC:
            raise RuntimeError("materialized file is not a normal GNU archive")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def thin_source(path: Path) -> Path | None:
    variant_dir = path.parents[2]
    candidates = [
        candidate
        for candidate in (variant_dir / "build").rglob(path.name)
        if candidate.is_file() and candidate.open("rb").read(8) == THIN_MAGIC
    ]
    return min(candidates, key=lambda item: len(item.parts)) if candidates else None


def thin_members(source: Path) -> list[Path]:
    names = [
        line
        for line in run(["ar", "t", source.name], cwd=source.parent).splitlines()
        if line
    ]
    return [Path(name) for name in names]


def materialize_thin(path: Path) -> tuple[str, int]:
    source = thin_source(path)
    if source is None:
        raise RuntimeError(f"original thin archive not found for {path}")
    members = thin_members(source)
    if not all((source.parent / member).is_file() for member in members):
        raise RuntimeError(f"thin archive has missing members: {source}")
    normal_archive_from_members(path, members, cwd=source.parent)
    actual_count = len(run(["ar", "t", path.as_posix()]).splitlines())
    if actual_count != len(members):
        raise RuntimeError(f"member count changed for {path}")
    return "thin", len(members)


def materialize_linker_script(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    variant_dir = path.parents[2]
    referenced: list[Path] = []
    group = re.search(r"\b(?:GROUP|INPUT)\s*\((.*?)\)", text, flags=re.DOTALL)
    if group is None:
        raise RuntimeError(f"no GROUP/INPUT expression in {path}")
    for token in group.group(1).split():
        if not token.endswith(".a"):
            continue
        candidate = Path(token)
        if candidate.is_file():
            referenced.append(candidate)
            continue
        # The builds were produced inside /workspace and later moved to this
        # checkout.  Linker scripts keep the old absolute prefix, so resolve
        # installed siblings by basename before using build-tree fallbacks.
        installed_sibling = path.parent / candidate.name
        if installed_sibling.is_file():
            referenced.append(installed_sibling)
            continue
        if re.fullmatch(r"libm-[^/]+\.a", candidate.name):
            fallback = variant_dir / "build/math/libm.a"
            if fallback.is_file():
                referenced.append(fallback)
                continue
        raise RuntimeError(f"unresolved linker-script archive {token} in {path}")
    if not referenced:
        raise RuntimeError(f"no archives referenced by {path}")

    temporary = path.with_name(f".{path.name}.{os.getpid()}.materializing")
    temporary.unlink(missing_ok=True)
    script = [f"CREATE {temporary.as_posix()}"]
    script.extend(f"ADDLIB {item.resolve().as_posix()}" for item in referenced)
    script.extend(("SAVE", "END", ""))
    try:
        run(["ar", "-M"], stdin="\n".join(script))
        if temporary.open("rb").read(8) != AR_MAGIC:
            raise RuntimeError("MRI output is not a normal GNU archive")
        member_count = len(run(["ar", "t", temporary.as_posix()]).splitlines())
        os.replace(temporary, path)
        return "linker_script", member_count
    finally:
        temporary.unlink(missing_ok=True)


def materialize_elf(path: Path) -> tuple[str, int]:
    normal_archive_from_members(path, [Path(path.name)], cwd=path.parent)
    return "relocatable_elf", 1


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    matrix = repository / "Dataset/manifests/library_matrix.tsv"
    build_root = repository / "Dataset/builds/libraries"
    report = repository / "Dataset/manifests/materialized_archives.jsonl"
    selected: list[Path] = []
    with matrix.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["status"] == "selected":
                selected.append(build_root / row["path"])

    existing_records = []
    if report.is_file():
        existing_records = [
            json.loads(line)
            for line in report.read_text(encoding="utf-8").splitlines()
            if line
        ]
    recorded_paths = {record["path"] for record in existing_records}
    records = list(existing_records)

    def record_transformation(record: dict) -> None:
        relative = record["path"]
        if relative in recorded_paths:
            return
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            stream.flush()
        recorded_paths.add(relative)
        records.append(record)

    for path in selected:
        magic = path.open("rb").read(8)
        if magic == AR_MAGIC:
            # Recover provenance if a previous interrupted run already
            # materialized this archive before appending its report.
            source = thin_source(path)
            if source is not None:
                record_transformation(
                    {
                        "path": path.relative_to(repository).as_posix(),
                        "source_kind": "thin",
                        "old_sha256": sha256(source),
                        "new_sha256": sha256(path),
                        "members": len(thin_members(source)),
                    }
                )
            continue
        old_sha256 = sha256(path)
        if magic == THIN_MAGIC:
            source_kind, members = materialize_thin(path)
        elif magic.startswith(ELF_MAGIC):
            source_kind, members = materialize_elf(path)
        else:
            source_kind, members = materialize_linker_script(path)
        record_transformation(
            {
                "path": path.relative_to(repository).as_posix(),
                "source_kind": source_kind,
                "old_sha256": old_sha256,
                "new_sha256": sha256(path),
                "members": members,
            }
        )
        print(f"[{len(records)}] {source_kind}: {path} ({members} members)")

    remaining = sum(path.open("rb").read(8) != AR_MAGIC for path in selected)
    print(
        f"Materialized={len(records)}, selected={len(selected)}, "
        f"remaining_non_ar={remaining}, report={report}"
    )
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
