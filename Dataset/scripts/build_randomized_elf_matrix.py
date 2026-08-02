#!/usr/bin/env python3
"""Build static ELF programs with reproducibly randomized libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from enrich_cu_ground_truth import (
    enrich_archives,
    inter_cu_ground_truth_summary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
REPO_DIR = DATASET_DIR.parent
DEFAULT_ARTIFACT_ROOT = REPO_DIR.parent / f"{REPO_DIR.name}_artifacts"
SOURCE_ROOT = DATASET_DIR / "sources/elf_sources"
LIB_ROOT = DATASET_DIR / "builds/lib_builds/gcc-16.1.1"
OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "builds/libseeker_balanced"
GROUND_TRUTH_ROOT = (
    DEFAULT_ARTIFACT_ROOT / "ground_truth/legacy-libseeker-primary"
)
OPTIMIZATIONS = ("O0", "O2", "O3", "Os")
GLIBC_OPTIMIZATIONS = ("O2", "O3", "Os")
LINKER_MAP_MEMBER_RE = re.compile(r"^(?P<archive>.+?\.a)\((?P<member>[^()]+)\)")
FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class LibraryDefinition:
    source: str
    archives: tuple[str, ...]


@dataclass
class LinkContext:
    env: dict[str, str]
    archive_metadata: list[dict[str, str]]
    map_file: Path
    library_prefixes: dict[str, Path]


LIBRARIES = {
    "glibc": LibraryDefinition("glibc-2.41", ("libc.a",)),
    "pcre2": LibraryDefinition("pcre2-10.47", ("libpcre2-8.a",)),
    "iconv": LibraryDefinition("libiconv-1.18", ("libiconv.a", "libcharset.a")),
    "ncurses": LibraryDefinition(
        "ncurses-6.5", ("libncursesw.a", "libtinfow.a")
    ),
    "gmp": LibraryDefinition("gmp-6.3.0", ("libgmp.a",)),
    "mpfr": LibraryDefinition("mpfr-4.2.1", ("libmpfr.a",)),
    "readline": LibraryDefinition("readline-8.2", ("libreadline.a", "libhistory.a")),
    "magic": LibraryDefinition("file-5.46", ("libmagic.a",)),
    "zlib": LibraryDefinition("zlib-1.3.1", ("libz.a",)),
    "bzip2": LibraryDefinition("bzip2", ("libbz2.a",)),
    "xz": LibraryDefinition("xz", ("liblzma.a",)),
    "openssl": LibraryDefinition("openssl-3.5.0", ("libssl.a", "libcrypto.a")),
    "attr": LibraryDefinition("attr", ("libattr.a",)),
    "acl": LibraryDefinition("acl-2.3.2", ("libacl.a",)),
    "selinux": LibraryDefinition("selinux-3.7", ("libselinux.a",)),
}


PROGRAMS = {
    "bash": {
        "source": "bash-5.3-beta",
        "binary": "bash",
        "libraries": ("glibc", "readline", "ncurses", "iconv"),
        "libs": ("glibc", "readline", "history", "ncurses", "tinfo", "iconv", "charset"),
        "configure": (
            "--enable-static-link",
            "--enable-readline",
            "--with-curses",
            "--disable-nls",
        ),
    },
    "coreutils": {
        "source": "coreutils-9.6",
        "binary": "src/ls",
        "libraries": ("glibc", "acl", "attr", "iconv", "gmp", "openssl"),
        "libs": (
            "glibc",
            "acl",
            "attr",
            "cap",
            "ssl",
            "crypto",
            "iconv",
            "charset",
            "gmp",
        ),
        "configure": (
            "--disable-nls",
            "--enable-acl",
            "--enable-xattr",
            "--with-openssl",
            "--without-selinux",
        ),
    },
    "grep": {
        "source": "grep-3.11",
        "binary": "src/grep",
        "libraries": ("glibc", "pcre2", "iconv"),
        "libs": ("glibc", "pcre2", "iconv", "charset"),
        "configure": (
            "--enable-perl-regexp",
            "--disable-nls",
            "--without-included-regex",
        ),
    },
    "grep-3.11": {
        "source": "grep-3.11",
        "binary": "src/grep",
        "libraries": ("glibc", "pcre2", "iconv"),
        "libs": ("glibc", "pcre2", "iconv", "charset"),
        "configure": (
            "--enable-perl-regexp",
            "--disable-nls",
            "--without-included-regex",
        ),
    },
    "gzip": {
        "source": "gzip-1.13",
        "binary": "gzip",
        "libraries": ("glibc",),
        "libs": ("glibc",),
        "configure": ("--disable-nls",),
    },
    "gnuchess": {
        "source": "gnuchess-6.2.11",
        "binary": "src/gnuchess",
        "libraries": ("glibc", "readline", "ncurses", "iconv"),
        "libs": ("glibc", "readline", "history", "ncurses", "tinfo", "iconv", "charset"),
        # Clang does not add libm implicitly when linking this C++ target.
        "system_libs": ("-lm",),
        "configure": ("--disable-nls", "--with-readline"),
        "use_glibc_headers": False,
    },
    "inetutils": {
        "source": "inetutils-2.6",
        "binary": "src/inetd",
        "libraries": ("glibc", "readline", "ncurses"),
        "libs": ("glibc", "readline", "history", "ncurses", "tinfo"),
        "include_suffixes": {"ncurses": ("ncursesw",)},
        "configure": ("--disable-nls",),
    },
    "less": {
        "source": "less-668",
        "binary": "less",
        "libraries": ("glibc", "pcre2", "ncurses"),
        "libs": ("glibc", "pcre2", "ncurses", "tinfo"),
        "configure": ("--with-regex=pcre2",),
    },
    "make": {
        "source": "make-4.4.1",
        "binary": "make",
        "libraries": ("glibc", "iconv"),
        "libs": ("glibc", "iconv", "charset"),
        "configure": ("--disable-nls",),
    },
    "sed": {
        "source": "sed-4.9",
        "binary": "sed/sed",
        "libraries": ("glibc", "iconv", "acl", "attr", "selinux", "pcre2"),
        "libs": ("glibc", "iconv", "charset", "acl", "attr", "selinux", "pcre2"),
        "configure": (
            "--disable-nls",
            "--without-included-regex",
            "--enable-acl",
            "--with-selinux",
        ),
    },
    "gawk": {
        "source": "gawk-5.3.2",
        "binary": "gawk",
        "libraries": ("glibc", "gmp", "mpfr", "readline", "ncurses"),
        "libs": (
            "glibc",
            "gmp",
            "mpfr",
            "readline",
            "history",
            "ncurses",
            "tinfo",
            "iconv",
            "charset",
        ),
        "configure": ("--disable-nls", "--disable-extensions"),
    },
    "nano": {
        "source": "nano-8.3",
        "binary": "src/nano",
        "libraries": ("glibc", "ncurses", "magic", "zlib", "iconv"),
        "libs": ("glibc", "ncurses", "tinfo", "magic", "zlib", "iconv", "charset"),
        "include_suffixes": {"ncurses": ("ncursesw",)},
        "configure": (
            "--disable-nls",
            "--enable-color",
            "--enable-nanorc",
            "--enable-libmagic",
        ),
    },
    "openssh": {
        "source": "openssh-portable-V_10_0_P2",
        "binary": "ssh",
        "libraries": ("glibc", "zlib", "openssl"),
        "libs": ("glibc", "zlib", "crypto", "ssl", "crypt"),
        # Required again when the final ssh target is relinked for its map.
        "ldflags_suffix": ("-L.", "-Lopenbsd-compat"),
        "configure": (),
    },
    "rsync": {
        "source": "rsync-3.4.1",
        "binary": "rsync",
        "libraries": ("glibc", "acl", "attr", "iconv", "zlib"),
        "libs": (
            "glibc",
            "acl",
            "attr",
            "iconv",
            "charset",
            "zlib",
            "crypto",
            "xxhash",
            "zstd",
            "lz4",
        ),
        "configure": (
            "--disable-openssl",
            "--disable-xxhash",
            "--disable-zstd",
            "--disable-lz4",
        ),
    },
    "socat": {
        "source": "socat-1.8.0.3",
        "binary": "socat",
        "libraries": ("glibc", "readline", "openssl"),
        "libs": ("glibc", "readline", "history", "crypto", "ssl"),
        "configure": ("--enable-readline", "--enable-openssl"),
    },
    "tar": {
        "source": "tar-1.35",
        "binary": "src/tar",
        "libraries": (
            "glibc", "acl", "attr", "selinux", "pcre2", "iconv"
        ),
        "libs": (
            "glibc", "acl", "attr", "selinux", "pcre2", "iconv", "charset"
        ),
        "configure": ("--disable-nls", "--enable-acl", "--with-selinux"),
    },
    "util-linux": {
        "source": "util-linux-v2.39.3",
        "binary": "lsblk",
        "libtool_all_static": True,
        "libraries": ("glibc", "ncurses"),
        "libs": ("glibc", "ncurses", "tinfo"),
        "include_suffixes": {"ncurses": ("ncursesw",)},
        "configure": (
            "--disable-nls",
            "--disable-shared",
            "--enable-static",
            "--enable-static-programs=blkid,fdisk,losetup,mount,nsenter,sfdisk,umount,unshare",
            "--disable-bash-completion",
            "--disable-use-tty-group",
            "--disable-makeinstall-chown",
            "--disable-makeinstall-setuid",
            "--without-udev",
            "--without-systemd",
            "--without-python",
            "--without-readline",
            "--without-cap-ng",
            "--without-libz",
            "--without-libmagic",
            "--without-user",
            "--without-btrfs",
            "--without-econf",
        ),
    },
    "vim": {
        "source": "vim-v9.1.1151",
        "binary": "src/vim",
        "libraries": ("glibc", "ncurses", "acl", "attr", "iconv"),
        "libs": ("glibc", "ncurses", "tinfo", "acl", "attr", "iconv", "charset", "selinux"),
        "configure": (
            "--disable-nls",
            "--with-features=normal",
            "--enable-gui=no",
            "--disable-selinux",
            "--disable-smack",
            "--disable-gpm",
            "--disable-canberra",
            "--disable-libsodium",
        ),
        "copy_source_to_build": True,
    },
    "wget2": {
        "source": "wget2-2.2.0",
        "binary": "src/wget2",
        "libtool_all_static": True,
        "libraries": ("glibc", "iconv", "zlib", "openssl", "bzip2", "xz"),
        "libs": (
            "glibc",
            "iconv",
            "charset",
            "zlib",
            "crypto",
            "ssl",
            "bz2",
            "lzma",
            "brotlidec",
            "brotlicommon",
            "idn2",
            "unistring",
            "psl",
            "nghttp2",
            "zstd",
        ),
        "configure": (
            "--disable-nls",
            "--disable-shared",
            "--enable-static",
            "--with-ssl=openssl",
            "--with-bzip2",
            "--with-lzma",
            "--without-libpsl",
            "--without-libhsts",
            "--without-libnghttp2",
            "--without-gpgme",
            "--without-brotlidec",
            "--without-zstd",
            "--without-lzip",
            "--without-libidn2",
            "--without-libidn",
            "--without-libpcre2",
            "--without-libpcre",
            "--without-libmicrohttpd",
            "--without-plugin-support",
        ),
    },
}


DEFAULT_PROGRAMS = (
    "bash",
    "gawk",
    "gnuchess",
    "grep",
    "gzip",
    "inetutils",
    "less",
    "make",
    "nano",
    "openssh",
    "rsync",
    "sed",
    "socat",
    "tar",
    "wget2",
)

LIBSEEKER_PROJECTS = (
    "bash",
    "coreutils",
    "gawk",
    "gnuchess",
    "grep",
    "inetutils",
    "less",
    "make",
    "nano",
    "openssh",
    "rsync",
    "sed",
    "socat",
    "tar",
    "util-linux",
    "vim",
    "wget2",
)

DATASET_PROFILES = {
    "libseeker": LIBSEEKER_PROJECTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument(
        "--profile",
        choices=sorted(DATASET_PROFILES),
        help=(
            "Predefined source-project set. The libseeker profile covers the "
            "projects that produced the reference LibSeeker executable names."
        ),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help=(
            "Build the exact cases listed in an existing randomized_matrix.json "
            "instead of creating a new randomized selection."
        ),
    )
    parser.add_argument(
        "--refresh-ground-truth",
        action="store_true",
        help=(
            "Do not rebuild from scratch. Relink existing matrix cases with a "
            "dedicated final linker map, then rewrite CU and library ground truth."
        ),
    )
    parser.add_argument(
        "--program",
        action="append",
        choices=sorted(PROGRAMS),
        help=(
            "Program to build. Can be repeated. Defaults to the dataset "
            "program set."
        ),
    )
    parser.add_argument(
        "--compiler",
        action="append",
        help=(
            "Exact ELF compiler command. Can be repeated. Defaults to "
            "gcc-11, gcc-13, clang-14 and clang-18."
        ),
    )
    parser.add_argument(
        "--elf-optimizations",
        default="O0,O2,O3",
        help="Comma-separated ELF optimization levels (default: O0,O2,O3).",
    )
    parser.add_argument(
        "--library-optimization",
        action="append",
        default=[],
        metavar="LIB=OPT",
        help=(
            "Force a library optimization, for example pcre2=O3. Can be "
            "repeated. Unspecified libraries are still selected randomly."
        ),
    )
    parser.add_argument(
        "--library-source",
        action="append",
        default=[],
        metavar="LIB=SOURCE",
        help=(
            "Override the source directory for a configured library, for "
            "example openssl=openssl. Can be repeated."
        ),
    )
    parser.add_argument(
        "--lib-root",
        type=Path,
        default=LIB_ROOT,
        help=(
            "Compiler-specific library root to use for linked libraries "
            f"(default: {LIB_ROOT})."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Build-cache root (default: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=GROUND_TRUTH_ROOT,
        help=(
            "Legacy primary-target report root. The complete per-ELF ground "
            "truth is produced later by assemble_datasets.py."
        ),
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose output directory already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next variant when a build fails.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.elf_optimizations_explicit = any(
        argument == "--elf-optimizations"
        or argument.startswith("--elf-optimizations=")
        for argument in sys.argv[1:]
    )
    if args.profile == "libseeker" and not args.elf_optimizations_explicit:
        args.elf_optimizations = "O0,O2,O3,Os"
    args.elf_optimizations = tuple(
        value.strip().removeprefix("-")
        for value in args.elf_optimizations.split(",")
        if value.strip()
    )
    invalid = set(args.elf_optimizations) - set(OPTIMIZATIONS)
    if invalid:
        parser.error(f"Invalid optimizations: {', '.join(sorted(invalid))}")
    forced_library_optimizations = {}
    for value in args.library_optimization:
        if "=" not in value:
            parser.error(
                f"Invalid --library-optimization {value!r}; expected LIB=OPT"
            )
        key, optimization = (part.strip() for part in value.split("=", 1))
        optimization = optimization.removeprefix("-")
        if key not in LIBRARIES:
            parser.error(f"Unknown library in --library-optimization: {key}")
        if optimization not in OPTIMIZATIONS:
            parser.error(f"Invalid optimization for {key}: {optimization}")
        if key == "glibc" and optimization == "O0":
            parser.error("glibc O0 is not supported; use O2, O3 or Os")
        forced_library_optimizations[key] = optimization
    args.library_optimization = forced_library_optimizations
    library_sources = {}
    for value in args.library_source:
        if "=" not in value:
            parser.error(f"Invalid --library-source {value!r}; expected LIB=SOURCE")
        key, source = (part.strip() for part in value.split("=", 1))
        if key not in LIBRARIES:
            parser.error(f"Unknown library in --library-source: {key}")
        if not source:
            parser.error(f"Empty source for --library-source {key}")
        library_sources[key] = source
    args.library_source = library_sources
    args.lib_root = args.lib_root.resolve()
    args.output_root = args.output_root.resolve()
    args.ground_truth_root = args.ground_truth_root.resolve()
    return args


def compiler_id(compiler: str) -> str:
    version = subprocess.run(
        [compiler, "-dumpfullversion", "-dumpversion"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return f"{Path(compiler).name}-{version}"


def compiler_family(compiler: str) -> str:
    name = Path(compiler).name
    if name.startswith(("gcc", "g++")):
        return "gcc"
    if name.startswith("clang"):
        return "clang"
    raise ValueError(f"Unsupported compiler: {compiler}")


def cxx_compiler_for(compiler: str) -> str:
    name = Path(compiler).name
    if name.startswith("gcc"):
        return name.replace("gcc", "g++", 1)
    if name.startswith("clang"):
        return name.replace("clang", "clang++", 1)
    raise ValueError(f"Unsupported compiler: {compiler}")


def archive_path(
    key: str,
    optimization: str,
    archive_name: str,
    archive_overrides: dict[tuple[str, str], Path] | None = None,
) -> Path:
    if archive_overrides:
        override = archive_overrides.get((key, archive_name))
        if override is not None:
            if not override.is_file():
                raise FileNotFoundError(f"Missing archive: {override}")
            return override.resolve()
    definition = LIBRARIES[key]
    path = (
        LIB_ROOT / definition.source / optimization / "install/lib" / archive_name
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}")
    return path.resolve()


def library_selection(
    rng: random.Random,
    keys: tuple[str, ...],
    forced: dict[str, str],
) -> dict[str, str]:
    result = {}
    for key in keys:
        if key in forced:
            result[key] = forced[key]
        else:
            choices = OPTIMIZATIONS if key != "glibc" else GLIBC_OPTIMIZATIONS
            result[key] = rng.choice(choices)
    return result


def apply_library_sources(overrides: dict[str, str]) -> None:
    for key, source in overrides.items():
        definition = LIBRARIES[key]
        LIBRARIES[key] = LibraryDefinition(source, definition.archives)


def install_prefix_from_archive(archive: Path) -> Path:
    archive = archive.resolve()
    if archive.parent.name == "lib" and archive.parent.parent.name == "install":
        return archive.parent.parent
    return archive.parent.parent


def relocate_glibc_linker_scripts(prefix: Path) -> list[Path]:
    """Rebase absolute install prefixes in relocatable GNU ld scripts.

    Existing thesis builds may have been copied from a different home
    directory.  Most glibc archives are binary and need no adjustment, but
    files such as libm.a and libc.so are short text linker scripts containing
    the original absolute prefix.
    """
    relocated = []
    prefix_pattern = re.compile(
        r"/[^()\s]+?/install(?=/lib(?:64)?/)"
    )
    for lib_dir_name in ("lib", "lib64"):
        lib_dir = prefix / lib_dir_name
        if not lib_dir.is_dir():
            continue
        for candidate in sorted(lib_dir.iterdir()):
            if not candidate.is_file():
                continue
            try:
                payload = candidate.read_bytes()
            except OSError:
                continue
            if (
                len(payload) > 1024 * 1024
                or b"\x00" in payload
                or b"GNU ld script" not in payload
            ):
                continue
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
            updated = prefix_pattern.sub(str(prefix.resolve()), text)
            if updated != text:
                candidate.write_text(updated)
                relocated.append(candidate)
    return relocated


def ensure_ncurses_compatibility_aliases(prefix: Path) -> list[Path]:
    """Expose conventional static archive names for wide ncurses builds."""
    created = []
    for lib_dir_name in ("lib", "lib64"):
        lib_dir = prefix / lib_dir_name
        if not lib_dir.is_dir():
            continue
        for alias_name, target_name in (
            ("libncurses.a", "libncursesw.a"),
            ("libtinfo.a", "libtinfow.a"),
        ):
            alias = lib_dir / alias_name
            target = lib_dir / target_name
            if alias.exists() or not target.is_file():
                continue
            alias.symlink_to(target.name)
            created.append(alias)
    return created


def compiler_command_from_matrix(value: str, metadata: dict | None = None) -> str:
    if isinstance(metadata, dict):
        command = metadata.get("command")
        if isinstance(command, str) and command:
            return Path(command).name
    name = Path(value).name
    if name in {"gcc", "clang"}:
        return name
    versioned_command = re.match(r"^(gcc|clang)-(\d+)-\d+(?:\.\d+)+$", name)
    if versioned_command:
        return f"{versioned_command.group(1)}-{versioned_command.group(2)}"
    if re.match(r"^gcc-\d+(?:\.\d+)+$", name):
        return "gcc"
    if re.match(r"^clang-\d+(?:\.\d+)+$", name):
        return "clang"
    if re.match(r"^(gcc|clang)-\d+$", name):
        return name
    raise ValueError(f"Unsupported compiler in matrix: {value}")


def matrix_case_library_sources(case: dict) -> dict[str, str]:
    result = dict(case.get("library_sources", {}))
    records = case.get("passed_archives") or case.get("archives") or []
    for record in records:
        if not isinstance(record, dict):
            continue
        library = record.get("library")
        source = record.get("source")
        if library and source:
            result[library] = source
    return result


def matrix_case_library_selection(case: dict) -> dict[str, str]:
    result = dict(case.get("seeded_library_selection", {}))
    records = case.get("passed_archives") or case.get("archives") or []
    for record in records:
        if not isinstance(record, dict):
            continue
        library = record.get("library")
        optimization = record.get("optimization")
        if library and optimization and library not in result:
            result[library] = optimization
    if not result:
        raise ValueError(f"Missing library selection for {case.get('variant')}")
    return result


def matrix_case_archive_overrides(case: dict) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    records = case.get("passed_archives") or case.get("archives") or []
    for record in records:
        if not isinstance(record, dict):
            continue
        library = record.get("library")
        archive = record.get("archive")
        if library and archive:
            archive_path_value = Path(str(archive)).resolve()
            if not archive_path_value.is_file():
                source = record.get("source")
                optimization = record.get("optimization")
                if source and optimization:
                    archive_name = Path(str(archive)).name
                    candidates = (
                        LIB_ROOT
                        / str(source)
                        / str(optimization)
                        / install_dir
                        / archive_name
                        for install_dir in ("install/lib", "install/lib64")
                    )
                    archive_path_value = next(
                        (candidate for candidate in candidates if candidate.is_file()),
                        archive_path_value,
                    )
            result[(str(library), archive_path_value.name)] = archive_path_value
    return result


def load_matrix_cases(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Invalid matrix cases in {path}")
    return payload, cases


def localize_case_paths(case: dict, output_dir: Path) -> dict:
    localized = dict(case)
    program = str(localized["program"])
    definition = PROGRAMS[program]
    localized["source"] = str(
        (SOURCE_ROOT / definition["source"]).resolve()
    )
    localized["binary"] = str((output_dir / program).resolve())
    localized["linker_map"] = str((output_dir / f"{program}.map").resolve())

    for records_key in ("passed_archives", "archives"):
        localized_records = []
        for original in localized.get(records_key, []):
            if not isinstance(original, dict):
                localized_records.append(original)
                continue
            record = dict(original)
            archive = record.get("archive")
            source = record.get("source")
            optimization = record.get("optimization")
            if archive and source and optimization:
                archive_name = Path(str(archive)).name
                candidates = (
                    LIB_ROOT
                    / str(source)
                    / str(optimization)
                    / install_dir
                    / archive_name
                    for install_dir in ("install/lib", "install/lib64")
                )
                current = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    None,
                )
                if current is not None:
                    record["archive"] = str(current.resolve())
            localized_records.append(record)
        if records_key in localized:
            localized[records_key] = localized_records
    return localized


def read_build_info_for_case(case: dict) -> dict | None:
    binary = case.get("binary")
    build_info = (
        Path(str(binary)).resolve().parent / "build-info.json"
        if binary
        else Path()
    )
    if not build_info.is_file():
        program = case.get("program")
        variant = case.get("variant")
        if program in PROGRAMS and variant:
            build_info = (
                OUTPUT_ROOT
                / PROGRAMS[str(program)]["source"]
                / "randomized"
                / str(variant)
                / "build-info.json"
            )
    if not build_info.is_file():
        return None
    payload = json.loads(build_info.read_text())
    merged = {**case, **payload}
    if payload.get("archives"):
        merged["passed_archives"] = payload.get("passed_archives") or payload["archives"]
    return localize_case_paths(merged, build_info.parent)


def merge_local_successful_cases(new_cases: list[dict]) -> list[dict]:
    """Preserve one successful case per exact compiler and matrix cell.

    Variant directories include the full compiler version, so caches from an
    older toolchain can coexist with the current build.  The reproduction
    matrix is authoritative and deliberately retains the requested compiler
    versions. Newly built cases are applied last and replace only an exact
    compiler-command/program/optimization match.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    for info_path in sorted(
        OUTPUT_ROOT.glob("*/randomized/*/build-info.json")
    ):
        try:
            case = json.loads(info_path.read_text())
            case = localize_case_paths(case, info_path.parent)
            key = (
                str(case["program"]),
                compiler_command_from_matrix(
                    str(case["compiler"]), case.get("compiler_metadata")
                ),
                str(case["elf_optimization"]),
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        merged[key] = case

    for case in new_cases:
        key = (
            str(case["program"]),
            compiler_command_from_matrix(
                str(case["compiler"]), case.get("compiler_metadata")
            ),
            str(case["elf_optimization"]),
        )
        merged[key] = case
    return [
        merged[key]
        for key in sorted(merged)
    ]


class NullLog:
    def write(self, _value: str) -> None:
        return None

    def flush(self) -> None:
        return None


def run(command: list[str], cwd: Path, env: dict[str, str], log, dry: bool):
    rendered = shlex.join(command)
    print(f"+ {rendered}", flush=True)
    log.write(f"+ {rendered}\n")
    log.flush()
    if dry:
        return
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)


def file_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = FILE_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    FILE_HASH_CACHE[cache_key] = value
    return value


def compiler_metadata(compiler: str) -> dict[str, str]:
    return {
        "command": compiler,
        "path": str(Path(shutil.which(compiler) or compiler).resolve()),
        "version": subprocess.run(
            [compiler, "--version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()[0],
    }


def reproduction_fingerprint(
    *,
    program: str,
    compiler_name: str,
    compiler_details: dict[str, str],
    elf_optimization: str,
    selected: dict[str, str],
    context: LinkContext,
) -> dict:
    archives = []
    for entry in context.archive_metadata:
        archive = Path(entry["archive"])
        archives.append({
            "library": entry.get("library"),
            "source": entry.get("source"),
            "optimization": entry.get("optimization"),
            "compiler": entry.get("compiler"),
            "archive": str(archive.resolve()),
            "sha256": file_sha256(archive),
        })
    return {
        "schema_version": 1,
        "program": program,
        "compiler": compiler_name,
        "compiler_metadata": compiler_details,
        "elf_optimization": elf_optimization,
        "seeded_library_selection": selected,
        "library_sources": {
            key: LIBRARIES[key].source for key in sorted(selected)
        },
        "program_optimization_flags": {
            key: context.env.get(key, "")
            for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBS")
        },
        "archives": archives,
    }


def static_elf_complete(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    result = subprocess.run(
        ["readelf", "-l", "-d", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return (
        result.returncode == 0
        and "INTERP" not in result.stdout
        and "(NEEDED)" not in result.stdout
    )


def existing_case_complete(
    metadata: dict,
    fingerprint: dict,
    binary: Path,
    linker_map: Path,
    build_dir: Path,
) -> bool:
    try:
        return bool(
            metadata.get("reproduction_fingerprint") == fingerprint
            and build_dir.is_dir()
            and static_elf_complete(binary)
            and linker_map.is_file()
            and metadata.get("binary_sha256") == file_sha256(binary)
            and metadata.get("linker_map_sha256") == file_sha256(linker_map)
        )
    except (OSError, TypeError, ValueError):
        return False


def parse_linker_map(map_file: Path) -> dict[Path, set[str]]:
    included: dict[Path, set[str]] = {}
    in_archive_section = False

    for line in map_file.read_text(errors="replace").splitlines():
        if line.startswith("Archive member included"):
            in_archive_section = True
            continue
        if in_archive_section and line.startswith("Discarded input sections"):
            break
        if not in_archive_section:
            continue

        match = LINKER_MAP_MEMBER_RE.match(line)
        if not match:
            continue
        archive = Path(match.group("archive")).resolve()
        included.setdefault(archive, set()).add(match.group("member"))
    return included


def archive_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["ar", "t", str(archive)],
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_archive_ground_truth(
    archive_metadata: list[dict[str, str]],
    map_file: Path,
    binary: Path,
) -> list[dict]:
    archives, _stats = enrich_archives(
        archive_metadata,
        map_file,
        binary,
        use_symbol_fallback=False,
    )
    return archives


def archive_ground_truth_summary(archives: list[dict]) -> dict[str, int | str]:
    return {
        "method": "linker_map",
        "included_compilation_units": sum(
            int(archive.get("included_compilation_units", 0) or 0)
            for archive in archives
        ),
        "total_compilation_units": sum(
            int(archive.get("total_compilation_units", 0) or 0)
            for archive in archives
        ),
        "map_archives": sum(
            1
            for archive in archives
            if archive.get("cu_ground_truth_method") == "linker_map"
        ),
        "empty_archives": sum(
            1
            for archive in archives
            if archive.get("cu_ground_truth_method") == "none"
        ),
    }


def library_ground_truth_summary(archives: list[dict]) -> dict:
    libraries: dict[str, dict] = {}
    for archive in archives:
        library = str(archive.get("library", ""))
        if not library:
            continue
        entry = libraries.setdefault(
            library,
            {
                "present": False,
                "archives": 0,
                "included_archives": 0,
                "included_compilation_units": 0,
                "total_compilation_units": 0,
                "sources": [],
                "optimizations": [],
            },
        )
        included_cu = int(archive.get("included_compilation_units", 0) or 0)
        total_cu = int(archive.get("total_compilation_units", 0) or 0)
        entry["archives"] += 1
        entry["included_archives"] += int(included_cu > 0)
        entry["included_compilation_units"] += included_cu
        entry["total_compilation_units"] += total_cu
        entry["present"] = entry["present"] or included_cu > 0
        source = archive.get("source")
        optimization = archive.get("optimization")
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        if optimization and optimization not in entry["optimizations"]:
            entry["optimizations"].append(optimization)

    present = sorted(
        library for library, record in libraries.items() if record["present"]
    )
    absent = sorted(
        library for library, record in libraries.items() if not record["present"]
    )
    return {
        "method": "linker_map_included_compilation_units",
        "present_libraries": present,
        "absent_libraries": absent,
        "libraries": libraries,
    }


def map_ldflags(map_file: Path) -> str:
    return f"-Wl,-Map={map_file} -Wl,--cref"


def relink_final_binary(
    build_dir: Path,
    source_binary: Path,
    target: str,
    map_file: Path,
    env: dict[str, str],
    log,
    dry: bool,
    libtool_all_static: bool = False,
) -> None:
    relink_env = dict(env)
    relink_env["LDFLAGS"] = (
        f"{env.get('LDFLAGS', '')} {map_ldflags(map_file)}"
    ).strip()
    if not dry:
        source_binary.unlink(missing_ok=True)
        map_file.unlink(missing_ok=True)
    target_dir = source_binary.parent
    target_name = source_binary.name
    if not dry and not (target_dir / "Makefile").is_file():
        target_dir = build_dir
        target_name = target
    make_ldflags = relink_env["LDFLAGS"]
    if libtool_all_static:
        make_ldflags = f"{make_ldflags} -all-static"
    run(
        ["make", "V=1", f"LDFLAGS={make_ldflags}", target_name],
        target_dir,
        relink_env,
        log,
        dry,
    )
    if not dry and not source_binary.is_file():
        raise FileNotFoundError(f"Relinked binary not found: {source_binary}")
    if not dry and not map_file.is_file():
        raise FileNotFoundError(f"Final linker map not generated: {map_file}")


def prepare_link_context(
    program: str,
    compiler: str,
    elf_optimization: str,
    selected: dict[str, str],
    output_dir: Path,
    archive_overrides: dict[tuple[str, str], Path] | None = None,
    materialize_wrappers: bool = True,
) -> LinkContext:
    definition = PROGRAMS[program]
    includes: list[str] = []
    library_dirs: list[str] = []
    archives: list[str] = []
    archive_metadata: list[dict[str, str]] = []
    library_prefixes: dict[str, Path] = {}

    for key in definition["libraries"]:
        optimization = selected[key]
        library = LIBRARIES[key]
        archive_paths = [
            archive_path(key, optimization, archive_name, archive_overrides)
            for archive_name in library.archives
        ]
        prefix = install_prefix_from_archive(archive_paths[0])
        library_prefixes[key] = prefix

        if key != "glibc" or definition.get("use_glibc_headers", True):
            includes.append(f"-I{prefix / 'include'}")
            includes.extend(
                f"-I{prefix / 'include' / suffix}"
                for suffix in definition.get("include_suffixes", {}).get(key, ())
            )
        library_dirs.append(f"-L{prefix / 'lib'}")
        for archive in archive_paths:
            archives.append(str(archive))
            archive_metadata.append({
                "library": key,
                "source": library.source,
                "optimization": optimization,
                "compiler": LIB_ROOT.name,
                "archive": str(archive),
            })

    glibc_prefix = library_prefixes["glibc"]
    map_file = output_dir / f"{program}.map"
    real_cxx = cxx_compiler_for(compiler)
    wrapper_dir = output_dir / "toolchain-wrappers"
    cc_wrapper = wrapper_dir / "map-cc"
    cxx_wrapper = wrapper_dir / "map-cxx"
    if materialize_wrappers:
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        wrapper_source = SCRIPT_DIR / "linker_map_cc.py"
        for wrapper in (cc_wrapper, cxx_wrapper):
            if wrapper.is_symlink() or wrapper.exists():
                wrapper.unlink()
            wrapper.symlink_to(wrapper_source)

    env = os.environ.copy()
    env.update({
        # Keep the symlink spelling: the wrapper uses argv[0] to select C/C++.
        "CC": str(cc_wrapper),
        "CXX": str(cxx_wrapper),
        "REAL_COMPILER": str(Path(shutil.which(compiler) or compiler).resolve()),
        "REAL_CXX": str(Path(shutil.which(real_cxx) or real_cxx).resolve()),
        "CFLAGS": f"-{elf_optimization} -g",
        "CXXFLAGS": f"-{elf_optimization} -g",
        "CPPFLAGS": " ".join(includes),
        "LDFLAGS": " ".join((
            f"-static -B{glibc_prefix / 'lib'}/",
            *library_dirs,
            *definition.get("ldflags_suffix", ()),
        )),
        "LIBS": " ".join((*archives[1:], *definition.get("system_libs", ()))),
        "PKG_CONFIG_LIBDIR": ":".join(
            str(library_prefixes[key] / "lib/pkgconfig")
            for key in definition["libraries"]
        ),
    })
    if program == "coreutils":
        # coreutils builds libstdbuf.so as an intermediate helper.  Keep that
        # one shared link dynamic while every collected program link retains
        # the global -static flag.
        env["DYNAMIC_SHARED_HELPERS"] = "1"

    if "pcre2" in selected:
        pcre_archive = archive_path(
            "pcre2",
            selected["pcre2"],
            "libpcre2-8.a",
            archive_overrides,
        )
        env["PCRE_CFLAGS"] = f"-I{library_prefixes['pcre2'] / 'include'}"
        env["PCRE_LIBS"] = str(pcre_archive)
    if "ncurses" in selected:
        ncurses_prefix = library_prefixes["ncurses"]
        env["NCURSESW_CFLAGS"] = f"-I{ncurses_prefix / 'include'}"
        env["NCURSESW_LIBS"] = " ".join(
            str(archive_path("ncurses", selected["ncurses"], name, archive_overrides))
            for name in LIBRARIES["ncurses"].archives
        )

    return LinkContext(
        env=env,
        archive_metadata=archive_metadata,
        map_file=map_file,
        library_prefixes=library_prefixes,
    )


def configure_command(
    program: str,
    configure_script: Path,
    selected: dict[str, str],
    context: LinkContext,
) -> list[str]:
    definition = PROGRAMS[program]
    configure = [
        str(configure_script.resolve()),
        *definition["configure"],
    ]
    if "iconv" in selected:
        configure.append(
            f"--with-libiconv-prefix={context.library_prefixes['iconv']}"
        )
    if program == "gawk":
        configure.extend([
            f"--with-mpfr={context.library_prefixes['mpfr']}",
            f"--with-readline={context.library_prefixes['readline']}",
        ])
    if program == "inetutils":
        configure.append(
            "--with-ncurses-include-dir="
            f"{context.library_prefixes['ncurses'] / 'include' / 'ncursesw'}"
        )
    if program == "openssh":
        configure.extend([
            f"--with-ssl-dir={context.library_prefixes['openssl']}",
            f"--with-zlib={context.library_prefixes['zlib']}",
        ])
    return configure


def build_case(
    program: str,
    compiler: str,
    elf_optimization: str,
    selected: dict[str, str],
    jobs: int,
    clean: bool,
    skip_existing: bool,
    dry: bool,
    archive_overrides: dict[tuple[str, str], Path] | None = None,
) -> dict:
    definition = PROGRAMS[program]
    source_dir = SOURCE_ROOT / definition["source"]
    configure_path = definition.get("configure_path", "configure")
    configure_script = source_dir / configure_path
    bootstrap = definition.get("bootstrap")
    if bootstrap and not configure_script.is_file() and not dry:
        subprocess.run(
            list(bootstrap),
            cwd=source_dir,
            check=True,
        )
    compiler_name = compiler_id(compiler)
    variant = f"{program}_{compiler_name}_{elf_optimization}"
    output_dir = OUTPUT_ROOT / definition["source"] / "randomized" / variant
    build_dir = output_dir / "build"
    report_dir = GROUND_TRUTH_ROOT / program / variant
    binary = output_dir / program
    compiler_details = compiler_metadata(compiler)
    fingerprint_context = prepare_link_context(
        program,
        compiler,
        elf_optimization,
        selected,
        output_dir,
        archive_overrides,
        materialize_wrappers=False,
    )
    fingerprint = reproduction_fingerprint(
        program=program,
        compiler_name=compiler_name,
        compiler_details=compiler_details,
        elf_optimization=elf_optimization,
        selected=selected,
        context=fingerprint_context,
    )
    if clean and not dry:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)
    if output_dir.exists():
        if skip_existing:
            metadata_file = output_dir / "build-info.json"
            if metadata_file.is_file():
                try:
                    metadata = json.loads(metadata_file.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
                    metadata = {}
                if metadata and existing_case_complete(
                    metadata,
                    fingerprint,
                    binary,
                    fingerprint_context.map_file,
                    build_dir,
                ):
                    print(f"SKIP verified case: {variant}")
                    return localize_case_paths(metadata, output_dir)
                print(f"INVALIDATE case {variant}: fingerprint/integrity mismatch")
            if not dry:
                shutil.rmtree(output_dir, ignore_errors=True)
                shutil.rmtree(report_dir, ignore_errors=True)
        elif not (dry and clean):
            raise FileExistsError(f"Build already exists: {output_dir}")
    if not dry:
        build_dir.mkdir(parents=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        if definition.get("copy_source_to_build"):
            shutil.copytree(
                source_dir,
                build_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "autom4te.cache",
                    "config.cache",
                    "config.log",
                    "config.status",
                ),
            )
            configure_script = build_dir / configure_path
            source_dir = build_dir

    if program == "gawk" and not dry:
        # gawk 5.3.2 post-processes this file even with --disable-extensions.
        extension_dir = build_dir / "extension"
        extension_dir.mkdir()
        (extension_dir / "Makefile").touch()

    source_binary = build_dir / definition["binary"]
    if not dry:
        glibc_archive = archive_path(
            "glibc",
            selected["glibc"],
            LIBRARIES["glibc"].archives[0],
            archive_overrides,
        )
        relocated = relocate_glibc_linker_scripts(
            install_prefix_from_archive(glibc_archive)
        )
        for script in relocated:
            print(f"REBASE glibc linker script: {script}")
        if "ncurses" in selected:
            ncurses_archive = archive_path(
                "ncurses",
                selected["ncurses"],
                LIBRARIES["ncurses"].archives[0],
                archive_overrides,
            )
            aliases = ensure_ncurses_compatibility_aliases(
                install_prefix_from_archive(ncurses_archive)
            )
            for alias in aliases:
                print(f"ALIAS ncurses archive: {alias}")
    link_context = prepare_link_context(
        program,
        compiler,
        elf_optimization,
        selected,
        output_dir,
        archive_overrides,
        materialize_wrappers=not dry,
    )
    configure = configure_command(
        program,
        configure_script,
        selected,
        link_context,
    )

    if dry:
        log = NullLog()
        run(configure, build_dir, link_context.env, log, dry)
        build_command = ["make", f"-j{jobs}", "V=1"]
        if definition.get("libtool_all_static"):
            build_command.append(
                f"LDFLAGS={link_context.env['LDFLAGS']} -all-static"
            )
        run(build_command, build_dir, link_context.env, log, dry)
        relink_final_binary(
            build_dir,
            source_binary,
            definition["binary"],
            link_context.map_file,
            link_context.env,
            log,
            dry,
            bool(definition.get("libtool_all_static")),
        )
    else:
        log_path = output_dir / "build.log"
        with log_path.open("w") as log:
            run(configure, build_dir, link_context.env, log, dry)
            build_command = ["make", f"-j{jobs}", "V=1"]
            if definition.get("libtool_all_static"):
                build_command.append(
                    f"LDFLAGS={link_context.env['LDFLAGS']} -all-static"
                )
            run(build_command, build_dir, link_context.env, log, dry)
            relink_final_binary(
                build_dir,
                source_binary,
                definition["binary"],
                link_context.map_file,
                link_context.env,
                log,
                dry,
                bool(definition.get("libtool_all_static")),
            )

    if not dry:
        if not source_binary.is_file():
            raise FileNotFoundError(f"Built binary not found: {source_binary}")
        shutil.copy2(source_binary, binary)
        if not link_context.map_file.is_file():
            raise FileNotFoundError(
                f"Linker map not generated: {link_context.map_file}"
            )

    linked_archives = (
        build_archive_ground_truth(
            link_context.archive_metadata,
            link_context.map_file,
            binary,
        )
        if not dry
        else link_context.archive_metadata
    )
    metadata = {
        "variant": variant,
        "program": program,
        "source": str((SOURCE_ROOT / definition["source"]).resolve()),
        "binary": str(binary.resolve()),
        "compiler": compiler_name,
        "compiler_metadata": compiler_details,
        "elf_optimization": elf_optimization,
        "program_optimization_flags": {
            key: link_context.env.get(key, "")
            for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBS")
        },
        "link_type": "static",
        "libs": list(definition.get("libs", definition["libraries"])),
        "seeded_library_selection": selected,
        "passed_archives": link_context.archive_metadata,
        "archives": linked_archives,
        "cu_ground_truth": archive_ground_truth_summary(linked_archives),
        "library_ground_truth": library_ground_truth_summary(linked_archives),
        "inter_cu_ground_truth": inter_cu_ground_truth_summary(linked_archives),
        "linker_map": str(link_context.map_file.resolve()),
        "binary_sha256": file_sha256(binary) if not dry else None,
        "linker_map_sha256": (
            file_sha256(link_context.map_file) if not dry else None
        ),
        "reproduction_fingerprint": fingerprint,
    }
    if not dry:
        (output_dir / "build-info.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (report_dir / "ground_truth.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return metadata


def matrix_case_selected(case: dict, args: argparse.Namespace) -> bool:
    if args.program and case.get("program") not in set(args.program):
        return False
    if args.compiler:
        compiler = compiler_command_from_matrix(
            str(case.get("compiler", "")), case.get("compiler_metadata")
        )
        if compiler not in set(args.compiler):
            return False
    if args.elf_optimizations_explicit and args.elf_optimizations:
        if case.get("elf_optimization") not in set(args.elf_optimizations):
            return False
    return True


def refresh_ground_truth_case(case: dict, dry: bool) -> dict:
    case = read_build_info_for_case(case) or case
    program = case["program"]
    definition = PROGRAMS[program]
    compiler = compiler_command_from_matrix(
        case["compiler"], case.get("compiler_metadata")
    )
    elf_optimization = case["elf_optimization"]
    selected = matrix_case_library_selection(case)
    archive_overrides = matrix_case_archive_overrides(case)
    output_dir = Path(case["binary"]).resolve().parent
    build_dir = output_dir / "build"
    source_binary = build_dir / definition["binary"]
    binary = output_dir / program
    report_dir = GROUND_TRUTH_ROOT / program / case["variant"]

    if not dry:
        if not build_dir.is_dir():
            raise FileNotFoundError(f"Build directory not found: {build_dir}")
        report_dir.mkdir(parents=True, exist_ok=True)
        glibc_archive = archive_path(
            "glibc",
            selected["glibc"],
            LIBRARIES["glibc"].archives[0],
            archive_overrides,
        )
        relocated = relocate_glibc_linker_scripts(
            install_prefix_from_archive(glibc_archive)
        )
        for script in relocated:
            print(f"REBASE glibc linker script: {script}")
        if "ncurses" in selected:
            ncurses_archive = archive_path(
                "ncurses",
                selected["ncurses"],
                LIBRARIES["ncurses"].archives[0],
                archive_overrides,
            )
            aliases = ensure_ncurses_compatibility_aliases(
                install_prefix_from_archive(ncurses_archive)
            )
            for alias in aliases:
                print(f"ALIAS ncurses archive: {alias}")

    link_context = prepare_link_context(
        program,
        compiler,
        elf_optimization,
        selected,
        output_dir,
        archive_overrides,
        materialize_wrappers=not dry,
    )

    if dry:
        log = NullLog()
        relink_final_binary(
            build_dir,
            source_binary,
            definition["binary"],
            link_context.map_file,
            link_context.env,
            log,
            dry,
            bool(definition.get("libtool_all_static")),
        )
    else:
        log_path = output_dir / "ground-truth-refresh.log"
        with log_path.open("w") as log:
            relink_final_binary(
                build_dir,
                source_binary,
                definition["binary"],
                link_context.map_file,
                link_context.env,
                log,
                dry,
                bool(definition.get("libtool_all_static")),
            )
        shutil.copy2(source_binary, binary)

    linked_archives = (
        build_archive_ground_truth(
            link_context.archive_metadata,
            link_context.map_file,
            binary,
        )
        if not dry
        else link_context.archive_metadata
    )
    metadata = {
        **case,
        "binary": str(binary.resolve()),
        "linker_map": str(link_context.map_file.resolve()),
        "passed_archives": link_context.archive_metadata,
        "archives": linked_archives,
        "cu_ground_truth": archive_ground_truth_summary(linked_archives),
        "library_ground_truth": library_ground_truth_summary(linked_archives),
        "inter_cu_ground_truth": inter_cu_ground_truth_summary(linked_archives),
    }

    if not dry:
        (output_dir / "build-info.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (report_dir / "ground_truth.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return metadata


def refresh_ground_truth_from_matrix(args: argparse.Namespace) -> int:
    matrix_path = args.matrix or OUTPUT_ROOT / "randomized_matrix.json"
    input_manifest, input_cases = load_matrix_cases(matrix_path)
    refreshed_cases: list[dict] = []
    failures = []

    for case in input_cases:
        if not matrix_case_selected(case, args):
            refreshed_cases.append(case)
            continue

        runtime_case = read_build_info_for_case(case) or case
        case_library_sources = matrix_case_library_sources(runtime_case)
        LIBRARIES.clear()
        LIBRARIES.update(LIBRARIES_BASELINE)
        apply_library_sources(case_library_sources)

        try:
            refreshed = refresh_ground_truth_case(runtime_case, args.dry_run)
            refreshed_cases.append(refreshed)
            summary = refreshed["cu_ground_truth"]
            print(
                f"[GT] {refreshed['variant']}: "
                f"cu={summary['included_compilation_units']}/"
                f"{summary['total_compilation_units']} "
                f"map_archives={summary['map_archives']} "
                f"empty_archives={summary['empty_archives']}"
            )
        except (
            FileExistsError,
            FileNotFoundError,
            subprocess.CalledProcessError,
            ValueError,
        ) as error:
            refreshed_cases.append(case)
            failure = {
                "program": case.get("program"),
                "compiler": case.get("compiler"),
                "elf_optimization": case.get("elf_optimization"),
                "variant": case.get("variant"),
                "error": str(error),
            }
            failures.append(failure)
            print(
                f"[FAIL:GT] {case.get('variant')}: {error}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                raise

    manifest = {
        **input_manifest,
        "cases": refreshed_cases,
        "ground_truth_method": "final_relink_linker_map",
        "ground_truth_failures": failures,
    }
    if args.dry_run:
        print("\nDry run: matrix not written")
    else:
        matrix_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(f"\nRefreshed ground truth in: {matrix_path}")

    if failures:
        print(f"Ground-truth refresh failures: {len(failures)}", file=sys.stderr)
        return 1
    return 0


LIBRARIES_BASELINE = dict(LIBRARIES)


def main() -> int:
    global LIB_ROOT, OUTPUT_ROOT, GROUND_TRUTH_ROOT
    args = parse_args()
    LIB_ROOT = args.lib_root
    OUTPUT_ROOT = args.output_root
    GROUND_TRUTH_ROOT = args.ground_truth_root
    default_libraries = dict(LIBRARIES)
    apply_library_sources(args.library_source)
    cli_libraries = dict(LIBRARIES)
    if not LIB_ROOT.is_dir():
        raise FileNotFoundError(f"Library root not found: {LIB_ROOT}")
    requested_compilers = set(
        args.compiler or ("gcc-11", "gcc-13", "clang-14", "clang-18")
    )
    required_commands = {"make", "ar", "readelf"}
    for compiler in requested_compilers:
        required_commands.add(compiler)
        required_commands.add(cxx_compiler_for(compiler))
    for command in sorted(required_commands):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"Missing command: {command}")

    if args.refresh_ground_truth:
        return refresh_ground_truth_from_matrix(args)

    matrix = []
    failures = []
    if args.matrix:
        input_manifest, input_cases = load_matrix_cases(args.matrix)
        for case in input_cases:
            program = case["program"]
            compiler = compiler_command_from_matrix(
                case["compiler"], case.get("compiler_metadata")
            )
            elf_optimization = case["elf_optimization"]
            selected = matrix_case_library_selection(case)
            case_library_sources = matrix_case_library_sources(case)
            archive_overrides = matrix_case_archive_overrides(case)
            LIBRARIES.clear()
            LIBRARIES.update(cli_libraries)
            apply_library_sources(case_library_sources)
            try:
                matrix.append(build_case(
                    program,
                    compiler,
                    elf_optimization,
                    selected,
                    args.jobs,
                    args.clean,
                    args.skip_existing,
                    args.dry_run,
                    archive_overrides,
                ))
            except (
                FileExistsError,
                FileNotFoundError,
                subprocess.CalledProcessError,
                ValueError,
            ) as error:
                failure = {
                    "program": program,
                    "compiler": compiler,
                    "elf_optimization": elf_optimization,
                    "error": str(error),
                }
                failures.append(failure)
                print(
                    f"[FAIL] {program} {compiler} {elf_optimization}: "
                    f"{error}",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    raise
        manifest_seed = input_manifest.get("seed", args.seed)
        manifest_optimizations = input_manifest.get(
            "elf_optimizations",
            sorted({case["elf_optimization"] for case in input_cases}),
        )
        manifest_profile = input_manifest.get("profile", args.profile)
    else:
        programs = (
            args.program
            or DATASET_PROFILES.get(args.profile)
            or DEFAULT_PROGRAMS
        )
        compilers = args.compiler or (
            "gcc-11", "gcc-13", "clang-14", "clang-18"
        )
        for program in programs:
            definition = PROGRAMS[program]
            for compiler in compilers:
                if compiler_family(compiler) not in definition.get(
                    "compilers", ("gcc", "clang")
                ):
                    print(
                        f"[SKIP] {program} {compiler}: compiler not present "
                        "in the LibSeeker inventory"
                    )
                    continue
                for elf_optimization in args.elf_optimizations:
                    LIBRARIES.clear()
                    LIBRARIES.update(default_libraries)
                    apply_library_sources(args.library_source)
                    case_rng = random.Random(
                        f"{args.seed}:{program}:{compiler}:{elf_optimization}"
                    )
                    selected = library_selection(
                        case_rng,
                        definition["libraries"],
                        args.library_optimization,
                    )
                    try:
                        matrix.append(build_case(
                            program,
                            compiler,
                            elf_optimization,
                            selected,
                            args.jobs,
                            args.clean,
                            args.skip_existing,
                            args.dry_run,
                        ))
                    except (
                        FileExistsError,
                        FileNotFoundError,
                        subprocess.CalledProcessError,
                    ) as error:
                        failure = {
                            "program": program,
                            "compiler": compiler,
                            "elf_optimization": elf_optimization,
                            "error": str(error),
                        }
                        failures.append(failure)
                        print(
                            f"[FAIL] {program} {compiler} {elf_optimization}: "
                            f"{error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
        manifest_seed = args.seed
        manifest_optimizations = args.elf_optimizations
        manifest_profile = args.profile

    if not args.matrix and not args.dry_run:
        matrix = merge_local_successful_cases(matrix)
        manifest_optimizations = [
            optimization
            for optimization in OPTIMIZATIONS
            if any(
                case.get("elf_optimization") == optimization
                for case in matrix
            )
        ]

    manifest = {
        "seed": manifest_seed,
        "profile": manifest_profile,
        "elf_optimizations": manifest_optimizations,
        "cases": matrix,
        "failures": failures,
    }
    if args.dry_run:
        print("\nDry run: matrix not written")
        if failures:
            print(f"Failures: {len(failures)}", file=sys.stderr)
            return 1
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix_path = OUTPUT_ROOT / "randomized_matrix.json"
    matrix_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nMatrix: {matrix_path}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
