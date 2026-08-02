#!/usr/bin/env python3
"""Build the independent Toybox/musl matrix used by Dataset B.

The script deliberately stops at build artifacts.  It does not assemble a
dataset directory and it does not emit the final ground truth.  Every final
link is instead routed through ``linker_map_cc.py`` so the ground-truth step
can consume an exact linker map and the corresponding link-command JSON.

Default matrix (121 programs x 10 cells = 1210 static ELF files) uses
GCC 11/13 and Clang 14/18. O1 is excluded for programs and musl; every musl
optimization is selected deterministically from O0/O2/O3/Os with seed
20260731.

Toybox 0.8.14 is built from an isolated source copy for each cell.  musl
1.2.6 is configured, built, and installed separately for every compiler
family/library-optimization pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = DATASET_DIR.parent

TOYBOX_VERSION = "0.8.14"
MUSL_VERSION = "1.2.6"
EXPECTED_PROGRAM_COUNT = 121
OPTIMIZATION_FLAGS = {
    "O0": "-O0",
    "O2": "-O2",
    "O3": "-O3",
    "Os": "-Os",
}
LINUX_UAPI_METADATA_CACHE: dict[str, Any] | None = None

# Toybox deliberately omits these generated single-command make targets
# because they collide with top-level Makefile targets.  The inventory only
# contains ``help`` today; keep ``install`` here for correct custom manifests.
DIRECT_SINGLE_SCRIPT_TARGETS = frozenset({"help", "install"})

# ``scripts/single.sh wget`` enables every WGET_* option, including the
# optional LibTLS backend, even though upstream declares it ``default n``.
# Dataset B intentionally has one controlled library (musl), so build wget's
# HTTP-only configuration instead of silently linking an untracked host TLS
# library.
WGET_STANDALONE_CONFIGURATION = {
    "mode": "internal_http_only",
    "enabled_symbols": ["WGET"],
    "disabled_symbols": ["WGET_LIBTLS", "TOYBOX_LIBCRYPTO"],
}

# Do not let a developer's shell silently change the compiler, target,
# optimization, sysroot, output paths, or instrumentation of the dataset.
SANITIZED_BUILD_ENVIRONMENT = frozenset(
    {
        "AR",
        "ASAN",
        "CC",
        "CPP",
        "CFLAGS",
        "CPPFLAGS",
        "CROSS_COMPILE",
        "CXX",
        "CXXFLAGS",
        "DESTDIR",
        "EXTRA_DRIVER_FLAGS",
        "GENDIR",
        "HOSTCC",
        "KCONFIG_CONFIG",
        "LD",
        "LDFLAGS",
        "LDOPTIMIZE",
        "LIBS",
        "LINK_MAP_ROOT",
        "MAKEFLAGS",
        "MFLAGS",
        "NM",
        "NOSTRIP",
        "OPTIMIZE",
        "OUTNAME",
        "PREFIX",
        "RANLIB",
        "REAL_COMPILER",
        "REAL_CXX",
        "STRIP",
        "SYSROOT",
        "TOOLCHAIN_PREFIX",
        "UNSTRIPPED",
        "V",
        "VERSION",
    }
)


@dataclass(frozen=True)
class MatrixCell:
    """One explicit compiler/program/musl optimization cell."""

    compiler: str
    program_optimization: str
    musl_optimization: str

    @property
    def compiler_family(self) -> str:
        return "gcc" if self.compiler.startswith("gcc") else "clang"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.compiler}-prog-{self.program_optimization}"
            f"-musl-{self.musl_optimization}"
        )


# Keep this readable and explicit: it is part of the dataset definition.
# musl optimizations are the deterministic outputs of seed 20260731.
MATRIX: tuple[MatrixCell, ...] = (
    MatrixCell("gcc-11", "O0", "O3"),
    MatrixCell("gcc-13", "O2", "O2"),
    MatrixCell("gcc-11", "O3", "O0"),
    MatrixCell("gcc-13", "Os", "O3"),
    MatrixCell("gcc-11", "O2", "O2"),
    MatrixCell("clang-14", "O0", "O2"),
    MatrixCell("clang-18", "O2", "Os"),
    MatrixCell("clang-14", "O3", "O0"),
    MatrixCell("clang-18", "Os", "O2"),
    MatrixCell("clang-14", "O2", "O2"),
)

DEFAULT_PROGRAM_LIST = DATASET_DIR / "manifests/unseen_programs.txt"
DEFAULT_SOURCE_ROOT = DATASET_DIR / "sources"
DEFAULT_BUILD_ROOT = (
    REPOSITORY_ROOT.parent
    / f"{REPOSITORY_ROOT.name}_artifacts"
    / "builds"
    / "toybox_unseen_balanced"
)
DEFAULT_LINKER_WRAPPER = SCRIPT_DIR / "linker_map_cc.py"

TOYBOX_SOURCE_URL = (
    "https://landley.net/toybox/downloads/toybox-0.8.14.tar.gz"
)
TOYBOX_SOURCE_SHA256 = (
    "827e4cdfd69f5da973e00e2a59b30b3c9857fb7fae74c362fd0b4f96be7929b0"
)
MUSL_SOURCE_URL = "https://git.musl-libc.org/git/musl"
MUSL_SOURCE_REVISION = "9fa28ece75d8a2191de7c5bb53bed224c5947417"
MUSL_RELEASE_URL = "https://musl.libc.org/releases/musl-1.2.6.tar.gz"


class BuildError(RuntimeError):
    """A build input or command failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("deve essere un intero positivo") from error
    if result < 1:
        raise argparse.ArgumentTypeError("deve essere maggiore di zero")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN write JSON {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def shell_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


class CommandRunner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        visible_env: Iterable[str] = (),
    ) -> None:
        argv = [str(part) for part in command]
        selected_env = []
        if env is not None:
            for key in visible_env:
                if key in env:
                    selected_env.append(f"{key}={shlex.quote(env[key])}")
        env_prefix = (" ".join(selected_env) + " ") if selected_env else ""
        print(f"+ (cd {shlex.quote(str(cwd))} && {env_prefix}{shell_command(argv)})")
        if self.dry_run:
            return

        if not cwd.is_dir():
            raise BuildError(f"directory di lavoro inesistente: {cwd}")
        if log_path is None:
            result = subprocess.run(argv, cwd=cwd, env=env, check=False)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {shell_command(argv)}\n")
                log.flush()
                result = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=env,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        if result.returncode:
            suffix = f" (log: {log_path})" if log_path else ""
            raise BuildError(
                f"comando terminato con codice {result.returncode}: "
                f"{shell_command(argv)}{suffix}"
            )


def command_version(command: str, *, dry_run: bool) -> dict[str, str]:
    resolved = shutil.which(command)
    if resolved is None and Path(command).is_file():
        resolved = str(Path(command).resolve())
    if resolved is None:
        if dry_run:
            return {
                "command": command,
                "path": command,
                "version": "unavailable during dry-run",
            }
        raise BuildError(f"compilatore non trovato: {command}")
    try:
        result = subprocess.run(
            [resolved, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        first_line = next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            "unknown",
        )
    except OSError as error:
        if not dry_run:
            raise BuildError(f"impossibile eseguire {resolved}: {error}") from error
        first_line = "unavailable during dry-run"
    return {"command": command, "path": resolved, "version": first_line}


def source_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def normalize_duplicate_archive_members(archive: Path) -> list[dict[str, Any]]:
    """Rename duplicate ar members so every linked compilation unit is unique.

    musl's libc.a intentionally contains two objects named free.lo,
    realloc.lo and clone.lo.  GNU ld prints only the member name in a linker
    map, making the two occurrences ambiguous.  Renaming duplicate members in
    the installed copy preserves their object bytes and symbols while making
    the per-ELF ground truth occurrence-exact.
    """
    listing = subprocess.run(
        ["ar", "t", str(archive)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    counts: dict[str, int] = {}
    for name in listing:
        counts[name] = counts.get(name, 0) + 1
    duplicates = {name: count for name, count in counts.items() if count > 1}
    if not duplicates:
        return []

    normalized: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="normalize-ar-", dir=archive.parent
    ) as temporary:
        work = Path(temporary)
        additions: list[Path] = []
        for name, count in sorted(duplicates.items()):
            if Path(name).name != name or name in {".", ".."}:
                raise BuildError(f"nome membro ar non sicuro: {name!r}")
            suffix = Path(name).suffix
            stem = name[: -len(suffix)] if suffix else name
            for occurrence in range(1, count + 1):
                subprocess.run(
                    ["ar", "xN", str(occurrence), str(archive), name],
                    cwd=work,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                extracted = work / name
                renamed = work / f"{stem}__occ{occurrence}{suffix}"
                extracted.replace(renamed)
                additions.append(renamed)
                normalized.append({
                    "original_name": name,
                    "occurrence": occurrence,
                    "normalized_name": renamed.name,
                    "object_sha256": sha256(renamed),
                })
            for _ in range(count):
                subprocess.run(
                    ["ar", "dN", "1", str(archive), name],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        subprocess.run(
            ["ar", "q", str(archive), *(str(path) for path in additions)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["ranlib", str(archive)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return normalized


def load_programs(path: Path) -> list[str]:
    if not path.is_file():
        raise BuildError(f"inventario programmi inesistente: {path}")
    programs: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw_line.split("#", 1)[0].strip()
        if not value:
            continue
        if (
            value in {".", ".."}
            or "/" in value
            or "\\" in value
            or any(character.isspace() for character in value)
            or "\0" in value
        ):
            raise BuildError(
                f"nome programma non sicuro a {path}:{line_number}: {value!r}"
            )
        if value in seen:
            raise BuildError(
                f"programma duplicato a {path}:{line_number}: {value!r}"
            )
        seen.add(value)
        programs.append(value)
    if not programs:
        raise BuildError(f"inventario programmi vuoto: {path}")
    return programs


def resolve_source(
    explicit: Path | None,
    source_root: Path,
    candidates: Sequence[Path],
    *,
    dry_run: bool,
) -> Path:
    if explicit is not None:
        result = explicit.expanduser().resolve(strict=False)
        if not result.is_dir() and not dry_run:
            raise BuildError(f"directory sorgente inesistente: {result}")
        return result

    expanded = [(source_root / candidate).resolve(strict=False) for candidate in candidates]
    for candidate in expanded:
        if candidate.is_dir():
            return candidate
    if dry_run:
        print(
            "DRY-RUN warning: sorgente non trovato; uso il percorso atteso "
            f"{expanded[0]}"
        )
        return expanded[0]
    searched = "\n  - ".join(str(path) for path in expanded)
    raise BuildError(f"sorgente non trovato; percorsi verificati:\n  - {searched}")


def verify_sources(toybox_source: Path, musl_source: Path, *, dry_run: bool) -> None:
    if dry_run and (not toybox_source.is_dir() or not musl_source.is_dir()):
        return

    toys_header = toybox_source / "toys.h"
    if not toys_header.is_file():
        raise BuildError(f"sorgente Toybox non valido (manca toys.h): {toybox_source}")
    match = re.search(
        r'#define\s+TOYBOX_VERSION\s+"([^"]+)"',
        toys_header.read_text(encoding="utf-8", errors="replace"),
    )
    if match is None or match.group(1) != TOYBOX_VERSION:
        actual = match.group(1) if match else "unknown"
        raise BuildError(
            f"versione Toybox errata: attesa {TOYBOX_VERSION}, trovata {actual}"
        )

    version_file = musl_source / "VERSION"
    if not version_file.is_file():
        raise BuildError(f"sorgente musl non valido (manca VERSION): {musl_source}")
    actual_musl = version_file.read_text(encoding="utf-8").strip()
    if actual_musl != MUSL_VERSION:
        raise BuildError(
            f"versione musl errata: attesa {MUSL_VERSION}, trovata {actual_musl}"
        )
    revision = source_revision(musl_source)
    if revision is not None and revision != MUSL_SOURCE_REVISION:
        raise BuildError(
            f"revision musl errata: attesa {MUSL_SOURCE_REVISION}, trovata {revision}"
        )


def ensure_safe_build_root(
    build_root: Path,
    *,
    toybox_source: Path,
    musl_source: Path,
) -> Path:
    result = build_root.expanduser().resolve(strict=False)
    forbidden_exact = {
        Path("/").resolve(),
        Path.home().resolve(),
        REPOSITORY_ROOT.resolve(),
        DATASET_DIR.resolve(),
        toybox_source.resolve(strict=False),
        musl_source.resolve(strict=False),
    }
    if result in forbidden_exact:
        raise BuildError(f"--build-root non sicura: {result}")
    for source in (toybox_source, musl_source):
        source_resolved = source.resolve(strict=False)
        if result.is_relative_to(source_resolved) or source_resolved.is_relative_to(result):
            raise BuildError(
                "--build-root e sorgenti non possono contenersi a vicenda: "
                f"{result}, {source_resolved}"
            )
    return result


def assert_removable(target: Path, build_root: Path) -> Path:
    resolved_target = target.resolve(strict=False)
    resolved_root = build_root.resolve(strict=False)
    if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
        raise BuildError(
            f"rimozione rifiutata fuori dalla root di build: {resolved_target}"
        )
    relative = resolved_target.relative_to(resolved_root)
    if len(relative.parts) < 2:
        raise BuildError(f"rimozione troppo ampia rifiutata: {resolved_target}")
    return resolved_target


def safe_remove_tree(target: Path, build_root: Path, *, dry_run: bool) -> None:
    resolved = assert_removable(target, build_root)
    if not resolved.exists() and not resolved.is_symlink():
        return
    print(f"CLEAN {resolved}")
    if dry_run:
        return
    if resolved.is_symlink() or resolved.is_file():
        resolved.unlink()
    else:
        shutil.rmtree(resolved)


def safe_unlink(target: Path, cell_root: Path, *, dry_run: bool) -> None:
    resolved = target.resolve(strict=False)
    root = cell_root.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved == root:
        raise BuildError(f"rimozione file rifiutata: {resolved}")
    if not resolved.exists() and not resolved.is_symlink():
        return
    print(f"REMOVE {resolved}")
    if dry_run:
        return
    if resolved.is_dir() and not resolved.is_symlink():
        raise BuildError(f"atteso file, trovata directory: {resolved}")
    resolved.unlink()


def musl_paths(build_root: Path, cell: MatrixCell) -> dict[str, Path]:
    root = build_root / "musl" / cell.compiler / cell.musl_optimization
    prefix = root / "install"
    wrapper_name = "musl-gcc" if cell.compiler_family == "gcc" else "musl-clang"
    return {
        "root": root,
        "build": root / "build",
        "prefix": prefix,
        "archive": prefix / "lib/libc.a",
        "driver": prefix / "bin" / wrapper_name,
        "info": root / "build-info.json",
        "log": root / "build.log",
    }


def cell_paths(build_root: Path, cell: MatrixCell) -> dict[str, Path]:
    root = build_root / "toybox" / cell.cell_id
    artifacts = root / "artifacts"
    return {
        "root": root,
        "source": root / "source",
        "artifacts": artifacts,
        "binaries": artifacts / "binaries",
        "unstripped": artifacts / "unstripped",
        "map_work": artifacts / "map-work",
        "maps": artifacts / "maps",
        "links": artifacts / "link-json",
        "binary_info": artifacts / "binary-info",
        "failures": artifacts / "failures",
        "info": root / "build-info.json",
        "log": root / "build.log",
    }


def clean_requested(
    scopes: Sequence[str],
    selected_cells: Sequence[MatrixCell],
    build_root: Path,
    *,
    dry_run: bool,
) -> None:
    if not scopes:
        return
    effective = set(scopes)
    clean_all = "all" in effective
    clean_musl = clean_all or bool(effective & {"selected", "musl"})
    clean_toybox = clean_all or bool(effective & {"selected", "toybox"})
    cells = MATRIX if clean_all else selected_cells

    if clean_musl:
        targets = {
            musl_paths(build_root, cell)["root"].resolve(strict=False)
            for cell in cells
        }
        for target in sorted(targets, key=str):
            safe_remove_tree(target, build_root, dry_run=dry_run)
    if clean_toybox:
        for cell in cells:
            safe_remove_tree(
                cell_paths(build_root, cell)["root"],
                build_root,
                dry_run=dry_run,
            )


def base_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in SANITIZED_BUILD_ENVIRONMENT:
        env.pop(key, None)
    env.update({"LANG": "C", "LC_ALL": "C"})
    return env


def linux_uapi_sources() -> tuple[tuple[str, Path], ...]:
    """Return the Linux UAPI trees needed by low-level Toybox applets."""

    asm_candidates = sorted(Path("/usr/include").glob("*-linux-gnu/asm"))
    mappings = [
        ("linux", Path("/usr/include/linux")),
        ("asm-generic", Path("/usr/include/asm-generic")),
    ]
    if asm_candidates:
        mappings.append(("asm", asm_candidates[0]))
    missing = [str(source) for _, source in mappings if not source.is_dir()]
    if not asm_candidates:
        missing.append("/usr/include/<target>-linux-gnu/asm")
    if missing:
        raise BuildError(
            "header UAPI Linux non disponibili; installare linux-libc-dev: "
            + ", ".join(missing)
        )
    return tuple(mappings)


def linux_uapi_metadata() -> dict[str, Any]:
    """Fingerprint the exact kernel headers copied into every musl sysroot."""

    global LINUX_UAPI_METADATA_CACHE
    if LINUX_UAPI_METADATA_CACHE is not None:
        return dict(LINUX_UAPI_METADATA_CACHE)

    digest = hashlib.sha256()
    file_count = 0
    sources = linux_uapi_sources()
    for destination, source_root in sources:
        for path in sorted(source_root.rglob("*"), key=lambda value: str(value)):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = Path(destination) / path.relative_to(source_root)
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(os.readlink(path).encode("utf-8"))
            else:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            digest.update(b"\0")
            file_count += 1

    LINUX_UAPI_METADATA_CACHE = {
        "provider": "linux-libc-dev",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "source_directories": {
            destination: str(source.resolve()) for destination, source in sources
        },
    }
    return dict(LINUX_UAPI_METADATA_CACHE)


def install_linux_uapi_headers(prefix: Path, *, dry_run: bool) -> dict[str, Any]:
    metadata = linux_uapi_metadata()
    include_root = prefix / "include"
    if dry_run:
        print(f"DRY UAPI Linux -> {include_root}")
        return metadata
    include_root.mkdir(parents=True, exist_ok=True)
    for destination, source in linux_uapi_sources():
        shutil.copytree(
            source,
            include_root / destination,
            dirs_exist_ok=True,
            symlinks=True,
        )
    return metadata


def build_musl(
    *,
    cell: MatrixCell,
    compiler: dict[str, str],
    musl_source: Path,
    build_root: Path,
    jobs: int,
    skip_existing: bool,
    runner: CommandRunner,
) -> dict[str, Any]:
    paths = musl_paths(build_root, cell)
    archive = paths["archive"]
    driver = paths["driver"]
    existing_info = json_read(paths["info"])
    linux_uapi = linux_uapi_metadata()
    complete = bool(
        archive.is_file()
        and driver.is_file()
        and (paths["prefix"] / "include/linux/fs.h").is_file()
        and (paths["prefix"] / "include/asm/unistd.h").is_file()
    )
    current_revision = source_revision(musl_source) if musl_source.is_dir() else None
    reusable = bool(
        complete
        and skip_existing
        and existing_info is not None
        and existing_info.get("schema_version") == 3
        and existing_info.get("compiler") == compiler
        and existing_info.get("optimization") == cell.musl_optimization
        and existing_info.get("cflags") == OPTIMIZATION_FLAGS[cell.musl_optimization]
        and isinstance(existing_info.get("source"), dict)
        and existing_info["source"].get("version") == MUSL_VERSION
        and existing_info["source"].get("git_revision") == current_revision
        and existing_info.get("linux_uapi") == linux_uapi
        and existing_info.get("archive_sha256") == sha256(archive)
    )
    normalized_members: list[dict[str, Any]] = []
    if reusable:
        print(
            f"SKIP musl {cell.compiler}/{cell.musl_optimization}: "
            f"{archive}"
        )
        # A dry run is strictly read-only, including when it inspects a cache
        # created before duplicate archive members were normalized.
        if runner.dry_run:
            return existing_info
        normalized_members = normalize_duplicate_archive_members(archive)
        if normalized_members:
            existing_info["normalized_duplicate_members"] = normalized_members
            existing_info["archive_sha256"] = sha256(archive)
            json_write(paths["info"], existing_info, dry_run=False)
        return existing_info
    # A non-reusable tree must never be configured or compiled incrementally.
    # In particular, an interrupted build may already contain config.mak and
    # object files produced by a different compiler even though libc.a is not
    # complete yet.  Reusing those files would make the recorded fingerprint
    # incorrect.  With --skip-existing only an exact fingerprint match reaches
    # the early return above; without it, rebuilding is deliberately fresh.
    if paths["root"].exists() and not runner.dry_run:
        reason = "fingerprint mismatch" if skip_existing else "fresh rebuild requested"
        print(
            f"INVALIDATE musl {cell.compiler}/{cell.musl_optimization}: "
            f"{reason}"
        )
        safe_remove_tree(paths["root"], build_root, dry_run=False)

    env = base_environment()
    env.update(
        {
            "CC": compiler["path"],
            "CFLAGS": OPTIMIZATION_FLAGS[cell.musl_optimization],
        }
    )
    configure = [
        str((musl_source / "configure").resolve(strict=False)),
        f"--prefix={paths['prefix']}",
        "--disable-shared",
        # "--enable-wrapper" asks configure to detect and require the wrapper
        # matching CC (musl-gcc for GCC, musl-clang for Clang).
        "--enable-wrapper",
    ]

    if not runner.dry_run:
        paths["build"].mkdir(parents=True, exist_ok=True)
    config_mak = paths["build"] / "config.mak"
    if not config_mak.is_file() or runner.dry_run:
        runner.run(
            configure,
            cwd=paths["build"],
            env=env,
            log_path=paths["log"],
            visible_env=("CC", "CFLAGS"),
        )
    else:
        print(f"REUSE configurazione musl {config_mak}")

    runner.run(
        ["make", f"-j{jobs}"],
        cwd=paths["build"],
        env=env,
        log_path=paths["log"],
        visible_env=("CC", "CFLAGS"),
    )
    runner.run(
        ["make", "install"],
        cwd=paths["build"],
        env=env,
        log_path=paths["log"],
        visible_env=("CC", "CFLAGS"),
    )
    linux_uapi = install_linux_uapi_headers(
        paths["prefix"], dry_run=runner.dry_run
    )

    if not runner.dry_run:
        if not archive.is_file():
            raise BuildError(f"libc.a musl non prodotta: {archive}")
        if not driver.is_file():
            raise BuildError(
                "wrapper toolchain musl non prodotto; verificare "
                f"--enable-wrapper e il compiler host: {driver}"
            )
        normalized_members = normalize_duplicate_archive_members(archive)

    payload: dict[str, Any] = {
        "schema_version": 3,
        "kind": "musl_build",
        "status": "planned" if runner.dry_run else "complete",
        "generated_at": utc_now(),
        "compiler": compiler,
        "optimization": cell.musl_optimization,
        "cflags": OPTIMIZATION_FLAGS[cell.musl_optimization],
        "source": {
            "project": "musl",
            "version": MUSL_VERSION,
            "path": str(musl_source.resolve(strict=False)),
            "url": MUSL_SOURCE_URL,
            "upstream_release_url": MUSL_RELEASE_URL,
            "pinned_revision": MUSL_SOURCE_REVISION,
            "git_revision": source_revision(musl_source)
            if musl_source.is_dir()
            else None,
        },
        "build_directory": str(paths["build"].resolve(strict=False)),
        "install_prefix": str(paths["prefix"].resolve(strict=False)),
        "toolchain_driver": str(driver.resolve(strict=False)),
        "archive": str(archive.resolve(strict=False)),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
        "normalized_duplicate_members": normalized_members,
        "linux_uapi": linux_uapi,
        "configure_command": configure,
    }
    json_write(paths["info"], payload, dry_run=runner.dry_run)
    return payload


def copy_toybox_source(
    source: Path,
    destination: Path,
    *,
    runner: CommandRunner,
) -> None:
    if destination.is_dir():
        print(f"REUSE copia Toybox isolata {destination}")
        return
    print(f"COPY {source} -> {destination}")
    if runner.dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".github",
            "__pycache__",
            "*.pyc",
        ),
    )


def toybox_cell_fingerprint(
    *,
    cell: MatrixCell,
    compiler: dict[str, str],
    toybox_source: Path,
    musl_info: dict[str, Any],
    linker_wrapper: Path,
) -> dict[str, Any]:
    musl_source = (
        musl_info.get("source")
        if isinstance(musl_info.get("source"), dict)
        else {}
    )
    optimize_flags = (
        f"{OPTIMIZATION_FLAGS[cell.program_optimization]} "
        "-ffunction-sections -fdata-sections "
        "-fno-asynchronous-unwind-tables -fno-strict-aliasing"
    )
    return {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "compiler": compiler,
        "program_optimization": cell.program_optimization,
        "program_optimize_flags": optimize_flags,
        "musl_optimization": cell.musl_optimization,
        "musl": {
            "schema_version": musl_info.get("schema_version"),
            "compiler": musl_info.get("compiler"),
            "cflags": musl_info.get("cflags"),
            "archive_sha256": musl_info.get("archive_sha256"),
            "linux_uapi_sha256": (
                musl_info.get("linux_uapi", {}).get("sha256")
                if isinstance(musl_info.get("linux_uapi"), dict)
                else None
            ),
            "source_version": musl_source.get("version"),
            "source_url": musl_source.get("url"),
            "source_revision": musl_source.get("git_revision"),
        },
        "toybox": {
            "version": TOYBOX_VERSION,
            "url": TOYBOX_SOURCE_URL,
            "sha256": TOYBOX_SOURCE_SHA256,
            "git_revision": source_revision(toybox_source)
            if toybox_source.is_dir()
            else None,
        },
        "linker_wrapper_sha256": (
            sha256(linker_wrapper) if linker_wrapper.is_file() else None
        ),
    }


def standalone_build_targets(
    toybox_source: Path, programs: Sequence[str]
) -> dict[str, str]:
    """Resolve aliases which Toybox's single.sh cannot enable directly.

    Most OLDTOY aliases have their own hidden Kconfig symbol and therefore
    work as a normal ``make <alias>`` target.  A small number (for example
    ``halt``/``poweroff`` -> ``reboot`` and ``nc`` -> ``netcat``) only share
    the implementation's symbol.  Toybox generates make targets for them,
    but ``scripts/single.sh`` cannot create a valid configuration for those
    target names.  Build the implementation target and preserve the result
    under the requested argv[0] name instead.
    """

    sources = sorted(toybox_source.glob("toys/*/*.c"))
    if not sources:
        return {program: program for program in programs}
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sources
    )
    configs = {
        value.lower()
        for value in re.findall(
            r"^config\s+([A-Za-z0-9_]+)\s*$", text, flags=re.MULTILINE
        )
    }
    aliases = {
        alias: target
        for alias, target in re.findall(
            r"OLDTOY\(\s*([^,\s]+)\s*,\s*([^,\s]+)", text
        )
    }

    result: dict[str, str] = {}
    for program in programs:
        config_name = program.replace("-", "_").lower()
        target = program
        visited = {program}
        while config_name not in configs and target in aliases:
            target = aliases[target]
            if target in visited:
                raise BuildError(f"ciclo OLDTOY rilevato per {program}")
            visited.add(target)
            config_name = target.replace("-", "_").lower()
        result[program] = target
    return result


def standalone_configuration(program: str) -> dict[str, Any] | None:
    if program == "wget":
        return dict(WGET_STANDALONE_CONFIGURATION)
    return None


def rewrite_kconfig(
    path: Path,
    *,
    enabled: Iterable[str],
    disabled: Iterable[str],
) -> None:
    """Rewrite symbols in a generated Kconfig file, requiring every symbol."""

    text = path.read_text(encoding="utf-8")
    for symbol in enabled:
        pattern = rf"^(?:# CONFIG_{re.escape(symbol)} is not set|CONFIG_{re.escape(symbol)}=.*)$"
        text, count = re.subn(
            pattern,
            f"CONFIG_{symbol}=y",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise BuildError(f"simbolo Kconfig non trovato: {symbol} in {path}")
    for symbol in disabled:
        pattern = rf"^(?:# CONFIG_{re.escape(symbol)} is not set|CONFIG_{re.escape(symbol)}=.*)$"
        text, count = re.subn(
            pattern,
            f"# CONFIG_{symbol} is not set",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise BuildError(f"simbolo Kconfig non trovato: {symbol} in {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_http_only_wget(
    *,
    source: Path,
    paths: dict[str, Path],
    program_env: dict[str, str],
    runner: CommandRunner,
    log_path: Path,
    visible_env: Sequence[str],
) -> None:
    """Mirror Toybox single.sh while keeping optional TLS backends disabled."""

    config = source / ".config"
    global_dependencies: list[str] = []
    if not runner.dry_run:
        if not config.is_file():
            raise BuildError(f"configurazione Toybox mancante: {config}")
        global_dependencies = [
            match.group(1)
            for match in re.finditer(
                r"^CONFIG_(TOYBOX_[A-Za-z0-9_]+)=y$",
                config.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if match.group(1) != "TOYBOX"
        ]

    single_env = dict(program_env)
    single_env["KCONFIG_CONFIG"] = ".singleconfig"
    runner.run(
        ["make", "allnoconfig"],
        cwd=source,
        env=single_env,
        log_path=log_path,
        visible_env=(*visible_env, "KCONFIG_CONFIG"),
    )
    if not runner.dry_run:
        rewrite_kconfig(
            source / ".singleconfig",
            enabled=("WGET", *global_dependencies),
            disabled=("WGET_LIBTLS", "TOYBOX_LIBCRYPTO", "TOYBOX"),
        )
    single_env["OUTNAME"] = str(
        (paths["binaries"] / "wget").resolve(strict=False)
    )
    runner.run(
        ["scripts/make.sh"],
        cwd=source,
        env=single_env,
        log_path=log_path,
        visible_env=(*visible_env, "KCONFIG_CONFIG", "OUTNAME"),
    )


def toybox_environment(
    *,
    cell: MatrixCell,
    compiler: dict[str, str],
    musl_driver: Path,
    linker_wrapper: Path,
    paths: dict[str, Path],
    jobs: int,
    map_root: Path,
) -> dict[str, str]:
    env = base_environment()
    env.update(
        {
            # linker_map_cc.py is compiler-driver compatible and forwards all
            # compile/preprocess probes without generating a map for them.
            "CC": str(linker_wrapper.resolve(strict=False)),
            "HOSTCC": compiler["path"],
            "REAL_COMPILER": str(musl_driver.resolve(strict=False)),
            "EXTRA_DRIVER_FLAGS": "-static",
            "LINK_MAP_ROOT": str(map_root.resolve(strict=False)),
            "CFLAGS": "-g",
            "OPTIMIZE": (
                f"{OPTIMIZATION_FLAGS[cell.program_optimization]} "
                "-ffunction-sections -fdata-sections "
                "-fno-asynchronous-unwind-tables -fno-strict-aliasing"
            ),
            "LDFLAGS": "-static",
            "LDOPTIMIZE": "-Wl,--gc-sections -Wl,--as-needed",
            "CPUS": str(jobs),
            "PREFIX": f"{paths['binaries'].resolve(strict=False)}/",
            "UNSTRIPPED": str(paths["unstripped"].resolve(strict=False)),
            "NOSTRIP": "1",
            "KCONFIG_CONFIG": ".config",
            "GENDIR": "generated",
        }
    )
    return env


def is_elf(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def inspect_static_elf(path: Path) -> dict[str, Any]:
    if not is_elf(path):
        raise BuildError(f"output non ELF: {path}")
    result: dict[str, Any] = {"elf": True, "static": None}
    readelf = shutil.which("readelf")
    if readelf is None:
        result["verification"] = "ELF magic only; readelf unavailable"
        return result
    completed = subprocess.run(
        [readelf, "-l", "-d", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise BuildError(f"readelf non riesce a leggere {path}")
    has_interpreter = "INTERP" in completed.stdout
    has_needed = "(NEEDED)" in completed.stdout
    result.update(
        {
            "static": not has_interpreter and not has_needed,
            "has_interp_segment": has_interpreter,
            "has_needed_entries": has_needed,
            "verification": "readelf -l -d",
        }
    )
    if not result["static"]:
        raise BuildError(f"output ELF non statico: {path}")
    return result


def link_json_matches(path: Path, expected_output: Path) -> bool:
    payload = json_read(path)
    if payload is None:
        return False
    output = payload.get("output")
    if output:
        try:
            if Path(str(output)).resolve(strict=False) == expected_output.resolve(
                strict=False
            ):
                return True
        except OSError:
            pass
    command = payload.get("command")
    if isinstance(command, list):
        argv = [str(value) for value in command]
        for index, argument in enumerate(argv):
            candidate: str | None = None
            if argument == "-o" and index + 1 < len(argv):
                candidate = argv[index + 1]
            elif argument.startswith("-o") and len(argument) > 2:
                candidate = argument[2:]
            if candidate is not None:
                candidate_path = Path(candidate)
                if not candidate_path.is_absolute():
                    cwd = payload.get("cwd")
                    candidate_path = Path(str(cwd or path.parent)) / candidate_path
                if candidate_path.resolve(strict=False) == expected_output.resolve(
                    strict=False
                ):
                    return True
    return False


def find_link_json(expected_output: Path, search_roots: Sequence[Path]) -> Path:
    direct_candidates = (
        expected_output.with_name(f"{expected_output.name}.link.json"),
        expected_output.with_suffix(expected_output.suffix + ".link.json"),
    )
    for candidate in direct_candidates:
        if candidate.is_file() and link_json_matches(candidate, expected_output):
            return candidate.resolve()

    matches: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            for candidate in root.rglob("*.link.json"):
                if link_json_matches(candidate, expected_output):
                    matches.append(candidate.resolve())
    if not matches:
        raise BuildError(f"link-command JSON non trovato per {expected_output}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def resolve_map_from_json(link_json: Path) -> Path | None:
    payload = json_read(link_json)
    if payload is None:
        return None
    for key in ("map", "map_file", "linker_map"):
        value = payload.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = Path(str(payload.get("cwd") or link_json.parent)) / candidate
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_linker_map(
    *,
    program: str,
    expected_output: Path,
    link_json: Path,
    map_root: Path,
) -> Path:
    from_json = resolve_map_from_json(link_json)
    if from_json is not None:
        return from_json
    direct = expected_output.with_name(f"{expected_output.name}.map")
    if direct.is_file():
        return direct.resolve()
    candidates = []
    if map_root.is_dir():
        candidates = [path.resolve() for path in map_root.rglob("*.map")]
    preferred = [
        path
        for path in candidates
        if path.name in {f"{program}.map", f"{expected_output.name}.map"}
        or path.stem == program
    ]
    usable = preferred or candidates
    if not usable:
        raise BuildError(f"linker map non trovata per {expected_output}")
    return max(usable, key=lambda path: path.stat().st_mtime_ns)


def canonicalize_link_artifacts(
    *,
    program: str,
    unstripped: Path,
    paths: dict[str, Path],
    map_root: Path,
) -> tuple[Path, Path]:
    link_json = find_link_json(
        unstripped,
        (unstripped.parent, map_root, paths["artifacts"]),
    )
    linker_map = find_linker_map(
        program=program,
        expected_output=unstripped,
        link_json=link_json,
        map_root=map_root,
    )
    canonical_map = paths["maps"] / f"{program}.map"
    canonical_link = paths["links"] / f"{program}.link.json"
    canonical_map.parent.mkdir(parents=True, exist_ok=True)
    canonical_link.parent.mkdir(parents=True, exist_ok=True)
    if linker_map.resolve() != canonical_map.resolve(strict=False):
        shutil.copy2(linker_map, canonical_map)
    if link_json.resolve() != canonical_link.resolve(strict=False):
        shutil.copy2(link_json, canonical_link)
    return canonical_map.resolve(), canonical_link.resolve()


def existing_binary_complete(
    *,
    program: str,
    paths: dict[str, Path],
    cell: MatrixCell,
    compiler: dict[str, str],
    musl_info: dict[str, Any],
    cell_fingerprint: dict[str, Any],
) -> bool:
    binary = paths["binaries"] / program
    info = paths["binary_info"] / f"{program}.json"
    payload = json_read(info)
    if not is_elf(binary) or payload is None:
        return False
    try:
        if inspect_static_elf(binary).get("static") is False:
            return False
    except BuildError:
        return False
    linker_map_value = payload.get("linker_map")
    link_json_value = payload.get("link_command_json")
    linker_map = Path(str(linker_map_value)) if linker_map_value else None
    link_json = Path(str(link_json_value)) if link_json_value else None
    musl = payload.get("musl") if isinstance(payload.get("musl"), dict) else {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    checks = (
        payload.get("schema_version") == 2,
        payload.get("program") == program,
        payload.get("program_optimization") == cell.program_optimization,
        payload.get("musl_optimization") == cell.musl_optimization,
        payload.get("compiler") == compiler,
        payload.get("cell_fingerprint") == cell_fingerprint,
        payload.get("standalone_configuration")
        == standalone_configuration(program),
        source.get("version") == TOYBOX_VERSION,
        source.get("url") == TOYBOX_SOURCE_URL,
        source.get("sha256") == TOYBOX_SOURCE_SHA256,
        source.get("git_revision") == cell_fingerprint["toybox"]["git_revision"],
        musl.get("version") == MUSL_VERSION,
        musl.get("archive_sha256") == musl_info.get("archive_sha256"),
        payload.get("binary_sha256") == sha256(binary),
        linker_map is not None and linker_map.is_file(),
        link_json is not None and link_json.is_file(),
    )
    if not all(checks):
        return False
    return bool(
        payload.get("linker_map_sha256") == sha256(linker_map)
        and payload.get("link_command_json_sha256") == sha256(link_json)
    )


def reset_program_artifacts(
    *,
    program: str,
    paths: dict[str, Path],
    build_root: Path,
    dry_run: bool,
) -> None:
    for candidate in (
        paths["binaries"] / program,
        paths["unstripped"] / program,
        (paths["unstripped"] / program).with_name(f"{program}.link.json"),
        paths["maps"] / f"{program}.map",
        paths["links"] / f"{program}.link.json",
        paths["binary_info"] / f"{program}.json",
        paths["failures"] / f"{program}.json",
    ):
        safe_unlink(candidate, paths["root"], dry_run=dry_run)
    safe_remove_tree(
        paths["map_work"] / program,
        build_root,
        dry_run=dry_run,
    )


def binary_metadata(
    *,
    program: str,
    standalone_build_target: str,
    cell: MatrixCell,
    compiler: dict[str, str],
    toybox_source: Path,
    toybox_tree: Path,
    musl_info: dict[str, Any],
    binary: Path,
    unstripped: Path,
    link_driver_output: Path,
    linker_map: Path,
    link_json: Path,
    elf_check: dict[str, Any],
    optimize_flags: str,
    cell_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    archive_value = musl_info.get("archive")
    archive = Path(str(archive_value)) if archive_value else None
    return {
        "schema_version": 2,
        "kind": "toybox_binary_build",
        "status": "complete",
        "generated_at": utc_now(),
        "dataset_profile": "unseen",
        "cell_id": cell.cell_id,
        "cell_fingerprint": cell_fingerprint,
        "program": program,
        "standalone_build_target": standalone_build_target,
        "standalone_configuration": standalone_configuration(program),
        "compiler": compiler,
        "program_optimization": cell.program_optimization,
        "program_optimize_flags": optimize_flags,
        "compiler_and_flags": {
            "compiler": compiler,
            "CFLAGS": "-g",
            "OPTIMIZE": optimize_flags,
            "LDFLAGS": "-static",
            "LDOPTIMIZE": "-Wl,--gc-sections -Wl,--as-needed",
            "EXTRA_DRIVER_FLAGS": "-static",
        },
        "musl_optimization": cell.musl_optimization,
        "link_type": "static",
        "elf_verification": elf_check,
        "source": {
            "project": "Toybox",
            "version": TOYBOX_VERSION,
            "path": str(toybox_source.resolve(strict=False)),
            "isolated_build_tree": str(toybox_tree.resolve(strict=False)),
            "url": TOYBOX_SOURCE_URL,
            "sha256": TOYBOX_SOURCE_SHA256,
            "git_revision": source_revision(toybox_source),
        },
        "binary": str(binary.resolve()),
        "binary_sha256": sha256(binary),
        "binary_size": binary.stat().st_size,
        "unstripped_link_output": str(unstripped.resolve()),
        "unstripped_link_output_sha256": sha256(unstripped),
        "link_driver_output": str(link_driver_output.resolve()),
        "linker_map": str(linker_map.resolve()),
        "linker_map_sha256": sha256(linker_map),
        "link_command_json": str(link_json.resolve()),
        "link_command_json_sha256": sha256(link_json),
        "musl": {
            "version": MUSL_VERSION,
            "optimization": cell.musl_optimization,
            "source": musl_info.get("source"),
            "install_prefix": musl_info.get("install_prefix"),
            "archive": str(archive.resolve()) if archive else None,
            "archive_sha256": sha256(archive)
            if archive is not None and archive.is_file()
            else musl_info.get("archive_sha256"),
            "toolchain_driver": musl_info.get("toolchain_driver"),
        },
    }


def build_cell(
    *,
    cell: MatrixCell,
    programs: Sequence[str],
    compiler: dict[str, str],
    toybox_source: Path,
    musl_info: dict[str, Any],
    build_root: Path,
    linker_wrapper: Path,
    jobs: int,
    skip_existing: bool,
    continue_on_error: bool,
    runner: CommandRunner,
) -> dict[str, Any]:
    paths = cell_paths(build_root, cell)
    musl_driver = Path(str(musl_info["toolchain_driver"]))
    cell_fingerprint = toybox_cell_fingerprint(
        cell=cell,
        compiler=compiler,
        toybox_source=toybox_source,
        musl_info=musl_info,
        linker_wrapper=linker_wrapper,
    )
    existing_cell_info = json_read(paths["info"])
    reusable_cell_tree = bool(
        skip_existing
        and existing_cell_info is not None
        and existing_cell_info.get("cell_fingerprint") == cell_fingerprint
        and paths["source"].is_dir()
    )
    if paths["root"].exists() and not reusable_cell_tree:
        reason = "fingerprint mismatch" if skip_existing else "fresh rebuild requested"
        print(f"INVALIDATE Toybox cell {cell.cell_id}: {reason}")
        if not runner.dry_run:
            safe_remove_tree(paths["root"], build_root, dry_run=False)
    copy_toybox_source(toybox_source, paths["source"], runner=runner)
    if not runner.dry_run:
        for key in (
            "binaries",
            "unstripped",
            "map_work",
            "maps",
            "links",
            "binary_info",
            "failures",
        ):
            paths[key].mkdir(parents=True, exist_ok=True)

    optimize_flags = (
        f"{OPTIMIZATION_FLAGS[cell.program_optimization]} "
        "-ffunction-sections -fdata-sections "
        "-fno-asynchronous-unwind-tables -fno-strict-aliasing"
    )
    base_info: dict[str, Any] = {
        "schema_version": 2,
        "kind": "toybox_matrix_cell",
        "status": "running" if not runner.dry_run else "planned",
        "generated_at": utc_now(),
        "dataset_profile": "unseen",
        "cell_id": cell.cell_id,
        "cell_fingerprint": cell_fingerprint,
        "compiler": compiler,
        "program_optimization": cell.program_optimization,
        "program_optimize_flags": optimize_flags,
        "compiler_and_flags": {
            "compiler": compiler,
            "CFLAGS": "-g",
            "OPTIMIZE": optimize_flags,
            "LDFLAGS": "-static",
            "LDOPTIMIZE": "-Wl,--gc-sections -Wl,--as-needed",
            "EXTRA_DRIVER_FLAGS": "-static",
        },
        "musl_optimization": cell.musl_optimization,
        "source": {
            "project": "Toybox",
            "version": TOYBOX_VERSION,
            "path": str(toybox_source.resolve(strict=False)),
            "isolated_build_tree": str(paths["source"].resolve(strict=False)),
            "url": TOYBOX_SOURCE_URL,
            "sha256": TOYBOX_SOURCE_SHA256,
            "git_revision": source_revision(toybox_source)
            if toybox_source.is_dir()
            else None,
        },
        "musl": musl_info,
        "libc_archive": musl_info.get("archive"),
        "linker_wrapper": str(linker_wrapper.resolve(strict=False)),
        "programs_requested": list(programs),
        "standalone_build_targets": standalone_build_targets(
            toybox_source, programs
        ),
        "programs_completed": [],
        "programs_skipped": [],
        "programs_failed": [],
        "artifacts": {
            key: str(paths[key].resolve(strict=False))
            for key in ("binaries", "maps", "links", "binary_info")
        },
    }
    json_write(paths["info"], base_info, dry_run=runner.dry_run)

    defconfig_map_root = paths["map_work"] / "_defconfig"
    env = toybox_environment(
        cell=cell,
        compiler=compiler,
        musl_driver=musl_driver,
        linker_wrapper=linker_wrapper,
        paths=paths,
        jobs=jobs,
        map_root=defconfig_map_root,
    )
    visible = (
        "CC",
        "HOSTCC",
        "REAL_COMPILER",
        "EXTRA_DRIVER_FLAGS",
        "LINK_MAP_ROOT",
        "CFLAGS",
        "OPTIMIZE",
        "LDFLAGS",
        "CPUS",
        "PREFIX",
        "UNSTRIPPED",
        "NOSTRIP",
    )
    try:
        runner.run(
            ["make", f"-j{jobs}", "defconfig"],
            cwd=paths["source"],
            env=env,
            log_path=paths["log"],
            visible_env=visible,
        )
    except BuildError as error:
        base_info["status"] = "failed"
        base_info["error"] = str(error)
        json_write(paths["info"], base_info, dry_run=runner.dry_run)
        raise

    build_targets = base_info["standalone_build_targets"]
    for program in programs:
        build_target = str(build_targets[program])
        if skip_existing and existing_binary_complete(
            program=program,
            paths=paths,
            cell=cell,
            compiler=compiler,
            musl_info=musl_info,
            cell_fingerprint=cell_fingerprint,
        ):
            print(f"SKIP {cell.cell_id}/{program}: artefatti completi")
            base_info["programs_skipped"].append(program)
            continue

        try:
            reset_program_artifacts(
                program=program,
                paths=paths,
                build_root=build_root,
                dry_run=runner.dry_run,
            )
            map_root = paths["map_work"] / program
            program_env = dict(env)
            program_env["LINK_MAP_ROOT"] = str(map_root.resolve(strict=False))
            standalone_command = (
                ["scripts/single.sh", build_target]
                if build_target in DIRECT_SINGLE_SCRIPT_TARGETS
                else ["make", f"-j{jobs}", "--", build_target]
            )
            if program == "wget" and build_target == "wget":
                build_http_only_wget(
                    source=paths["source"],
                    paths=paths,
                    program_env=program_env,
                    runner=runner,
                    log_path=paths["log"],
                    visible_env=visible,
                )
            else:
                runner.run(
                    standalone_command,
                    cwd=paths["source"],
                    env=program_env,
                    log_path=paths["log"],
                    visible_env=visible,
                )
            if runner.dry_run:
                base_info["programs_completed"].append(program)
                continue

            binary = paths["binaries"] / program
            unstripped = paths["unstripped"] / program
            built_binary = paths["binaries"] / build_target
            link_driver_output = paths["unstripped"] / build_target
            if not link_driver_output.is_file():
                raise BuildError(
                    f"output di link Toybox mancante: {link_driver_output}"
                )
            if build_target != program:
                if not built_binary.is_file():
                    raise BuildError(
                        f"binario alias Toybox mancante: {built_binary}"
                    )
                shutil.copy2(built_binary, binary)
                shutil.copy2(link_driver_output, unstripped)
            elf_check = inspect_static_elf(binary)
            if sha256(binary) != sha256(unstripped):
                raise BuildError(
                    "il binario finale non coincide con l'output non strippato "
                    f"nonostante NOSTRIP=1: {binary}"
                )
            linker_map, link_json = canonicalize_link_artifacts(
                program=program,
                unstripped=link_driver_output,
                paths=paths,
                map_root=map_root,
            )
            metadata = binary_metadata(
                program=program,
                standalone_build_target=build_target,
                cell=cell,
                compiler=compiler,
                toybox_source=toybox_source,
                toybox_tree=paths["source"],
                musl_info=musl_info,
                binary=binary,
                unstripped=unstripped,
                link_driver_output=link_driver_output,
                linker_map=linker_map,
                link_json=link_json,
                elf_check=elf_check,
                optimize_flags=optimize_flags,
                cell_fingerprint=cell_fingerprint,
            )
            json_write(
                paths["binary_info"] / f"{program}.json",
                metadata,
                dry_run=False,
            )
            base_info["programs_completed"].append(program)
        except Exception as error:
            failure = {
                "schema_version": 1,
                "kind": "toybox_binary_failure",
                "generated_at": utc_now(),
                "cell_id": cell.cell_id,
                "program": program,
                "standalone_build_target": build_target,
                "compiler": compiler,
                "program_optimization": cell.program_optimization,
                "musl_optimization": cell.musl_optimization,
                "error": str(error),
            }
            json_write(
                paths["failures"] / f"{program}.json",
                failure,
                dry_run=runner.dry_run,
            )
            base_info["programs_failed"].append(
                {"program": program, "error": str(error)}
            )
            json_write(paths["info"], base_info, dry_run=runner.dry_run)
            if not continue_on_error:
                raise
            print(f"ERROR {cell.cell_id}/{program}: {error}", file=sys.stderr)

    if base_info["programs_failed"]:
        base_info["status"] = "partial"
    else:
        base_info["status"] = "planned" if runner.dry_run else "complete"
    base_info["finished_at"] = utc_now()
    base_info["summary"] = {
        "requested": len(programs),
        "completed": len(base_info["programs_completed"]),
        "skipped": len(base_info["programs_skipped"]),
        "failed": len(base_info["programs_failed"]),
    }
    json_write(paths["info"], base_info, dry_run=runner.dry_run)
    return base_info


def select_cells(values: Sequence[str] | None) -> list[MatrixCell]:
    if not values:
        return list(MATRIX)
    requested: list[str] = []
    for value in values:
        requested.extend(item.strip() for item in value.split(",") if item.strip())
    by_id = {cell.cell_id: cell for cell in MATRIX}
    unknown = [value for value in requested if value not in by_id]
    if unknown:
        choices = ", ".join(by_id)
        raise BuildError(
            f"celle sconosciute: {', '.join(unknown)}; valori validi: {choices}"
        )
    result = []
    seen = set()
    for value in requested:
        if value not in seen:
            result.append(by_id[value])
            seen.add(value)
    return result


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compila il Dataset B: comandi standalone Toybox 0.8.14, "
            "link statico con musl 1.2.6 e linker map esatta."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--program-list",
        "--programs-file",
        dest="program_list",
        type=Path,
        default=DEFAULT_PROGRAM_LIST,
        help="inventario dei 121 comandi Toybox unseen",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="root sotto cui cercare automaticamente i due sorgenti",
    )
    parser.add_argument(
        "--toybox-source",
        "--toybox-src",
        dest="toybox_source",
        type=Path,
        help="directory sorgente Toybox 0.8.14 esplicita",
    )
    parser.add_argument(
        "--musl-source",
        "--musl-src",
        dest="musl_source",
        type=Path,
        help="directory sorgente musl 1.2.6 esplicita",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=DEFAULT_BUILD_ROOT,
        help="root degli artefatti intermedi (fuori dalla repository per default)",
    )
    parser.add_argument(
        "--linker-wrapper",
        type=Path,
        default=DEFAULT_LINKER_WRAPPER,
        help="wrapper compiler-like che produce .map e .link.json",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=max(1, os.cpu_count() or 1),
        help="parallelismo delle build musl/Toybox",
    )
    parser.add_argument(
        "--limit-programs",
        type=positive_int,
        help="usa solo i primi N programmi (smoke test)",
    )
    parser.add_argument(
        "--cell",
        "--only-cell",
        action="append",
        help=(
            "limita a una cella della matrice; ripetibile o separata da virgole "
            "(il default usa tutte e 10)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="riusa musl e binari completi già presenti",
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const="selected",
        action="append",
        choices=("selected", "musl", "toybox", "all"),
        default=[],
        help=(
            "pulisce solo directory note: selected (celle richieste, default), "
            "musl, toybox o all; ripetibile"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mostra copie, pulizie e comandi senza scrivere nulla",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continua con programmi/celle successivi e termina non-zero alla fine",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        source_root = args.source_root.expanduser().resolve(strict=False)
        toybox_source = resolve_source(
            args.toybox_source,
            source_root,
            (
                Path("unseen_sources") / f"toybox-{TOYBOX_VERSION}",
                Path(f"toybox-{TOYBOX_VERSION}"),
                Path("elf_sources") / f"toybox-{TOYBOX_VERSION}",
                Path("program_sources") / f"toybox-{TOYBOX_VERSION}",
            ),
            dry_run=args.dry_run,
        )
        musl_source = resolve_source(
            args.musl_source,
            source_root,
            (
                Path("unseen_sources") / f"musl-{MUSL_VERSION}",
                Path(f"musl-{MUSL_VERSION}"),
                Path("lib_sources") / f"musl-{MUSL_VERSION}",
                Path("library_sources") / f"musl-{MUSL_VERSION}",
            ),
            dry_run=args.dry_run,
        )
        verify_sources(toybox_source, musl_source, dry_run=args.dry_run)
        build_root = ensure_safe_build_root(
            args.build_root,
            toybox_source=toybox_source,
            musl_source=musl_source,
        )
        linker_wrapper = args.linker_wrapper.expanduser().resolve(strict=False)
        if not linker_wrapper.is_file() and not args.dry_run:
            raise BuildError(f"wrapper linker inesistente: {linker_wrapper}")
        if (
            linker_wrapper.is_file()
            and not os.access(linker_wrapper, os.X_OK)
            and not args.dry_run
        ):
            raise BuildError(f"wrapper linker non eseguibile: {linker_wrapper}")

        program_list = args.program_list.expanduser().resolve(strict=False)
        programs = load_programs(program_list)
        if program_list == DEFAULT_PROGRAM_LIST.resolve(strict=False) and len(programs) != EXPECTED_PROGRAM_COUNT:
            raise BuildError(
                f"inventario unseen inatteso: {len(programs)} programmi; "
                f"attesi {EXPECTED_PROGRAM_COUNT}"
            )
        original_program_count = len(programs)
        if args.limit_programs is not None:
            programs = programs[: args.limit_programs]
        selected_cells = select_cells(args.cell)

        compilers = {
            command: command_version(command, dry_run=args.dry_run)
            for command in sorted({cell.compiler for cell in selected_cells})
        }
        clean_requested(
            args.clean,
            selected_cells,
            build_root,
            dry_run=args.dry_run,
        )

        print(
            f"PLAN programmi={len(programs)}/{original_program_count} "
            f"celle={len(selected_cells)} ELF_attesi={len(programs) * len(selected_cells)}"
        )
        print(f"PROGRAM_LIST {program_list}")
        print(f"TOYBOX_SOURCE {toybox_source}")
        print(f"MUSL_SOURCE {musl_source}")
        print(f"BUILD_ROOT {build_root}")
        for cell in selected_cells:
            print(
                f"CELL {cell.cell_id}: compiler={cell.compiler} "
                f"program_opt={cell.program_optimization} "
                f"musl_opt={cell.musl_optimization}"
            )

        runner = CommandRunner(dry_run=args.dry_run)
        musl_cache: dict[tuple[str, str], dict[str, Any]] = {}
        cell_results = []
        failures: list[dict[str, str]] = []
        for cell in selected_cells:
            key = (cell.compiler, cell.musl_optimization)
            try:
                if key not in musl_cache:
                    musl_cache[key] = build_musl(
                        cell=cell,
                        compiler=compilers[cell.compiler],
                        musl_source=musl_source,
                        build_root=build_root,
                        jobs=args.jobs,
                        skip_existing=args.skip_existing,
                        runner=runner,
                    )
                result = build_cell(
                    cell=cell,
                    programs=programs,
                    compiler=compilers[cell.compiler],
                    toybox_source=toybox_source,
                    musl_info=musl_cache[key],
                    build_root=build_root,
                    linker_wrapper=linker_wrapper,
                    jobs=args.jobs,
                    skip_existing=args.skip_existing,
                    continue_on_error=args.continue_on_error,
                    runner=runner,
                )
                cell_results.append(result)
                for failed in result.get("programs_failed", []):
                    failures.append(
                        {
                            "cell_id": cell.cell_id,
                            "program": str(failed.get("program")),
                            "error": str(failed.get("error")),
                        }
                    )
            except Exception as error:
                failures.append({"cell_id": cell.cell_id, "error": str(error)})
                print(f"ERROR {cell.cell_id}: {error}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

        summary = {
            "schema_version": 1,
            "kind": "toybox_dataset_build_summary",
            "status": "planned"
            if args.dry_run
            else ("partial" if failures else "complete"),
            "generated_at": utc_now(),
            "dataset_profile": "unseen",
            "program_list": str(program_list),
            "program_count": len(programs),
            "matrix_cell_count": len(selected_cells),
            "expected_binary_count": len(programs) * len(selected_cells),
            "toybox_source": str(toybox_source),
            "musl_source": str(musl_source),
            "build_root": str(build_root),
            "matrix": [
                {
                    "cell_id": cell.cell_id,
                    "compiler": cell.compiler,
                    "compiler_family": cell.compiler_family,
                    "program_optimization": cell.program_optimization,
                    "musl_optimization": cell.musl_optimization,
                }
                for cell in selected_cells
            ],
            "failures": failures,
            "cells_completed": len(cell_results),
        }
        json_write(
            build_root / "build-summary.json",
            summary,
            dry_run=args.dry_run,
        )
        if failures:
            print(f"BUILD PARTIAL: {len(failures)} errori", file=sys.stderr)
            return 1
        print(
            "DRY-RUN completato" if args.dry_run else "Build Toybox completata"
        )
        return 0
    except (BuildError, OSError, subprocess.SubprocessError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
