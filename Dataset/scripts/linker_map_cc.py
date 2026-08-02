#!/usr/bin/env python3
"""Transparent compiler-driver wrapper that records GNU linker maps.

The wrapper is intended to be used as ``CC`` and, through a symlink whose
name contains ``cxx`` or ``++``, as ``CXX``.  ``REAL_COMPILER`` is required;
``REAL_CXX`` optionally selects a different driver for the C++ symlink.

Optional environment variables:

``TOOLCHAIN_PREFIX``
    Text prepended to the selected compiler name (for example
    ``x86_64-linux-musl-`` or ``/opt/cross/bin/``).
``SYSROOT``
    Added to every driver invocation as ``--sysroot=<value>``.
``EXTRA_DRIVER_FLAGS``
    Additional driver flags, parsed with POSIX shell quoting.
``LINK_MAP_ROOT``
    Central map directory.  To avoid basename and parallel-build collisions,
    the absolute output path is mirrored below this directory.  Without it,
    the map is written next to the output.
``DYNAMIC_SHARED_HELPERS``
    When set to ``1``, remove the driver-level ``-static`` flag only from
    ``-shared`` links.  Some projects (notably coreutils' ``libstdbuf.so``)
    build a shared helper while all dataset executables must remain static.

Successful final links also produce ``<output>.link.json`` atomically.  All
non-linking compiler calls are passed through without map/metadata side
effects.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


_NON_LINKING_OPTIONS = frozenset(
    {
        "-c",
        "--compile",
        "-E",
        "--preprocess",
        "-S",
        "-M",
        "-MM",
        "-fsyntax-only",
        "--analyze",
        "-analyze",
        "-emit-ast",
        "-emit-pch",
        "-rewrite-objc",
    }
)

_QUERY_OPTIONS = frozenset(
    {
        "--help",
        "--help-hidden",
        "-help",
        "--version",
        "--target-help",
        "-dumpmachine",
        "-dumpversion",
        "-dumpfullversion",
        "-dumpspecs",
        "-print-search-dirs",
        "-print-libgcc-file-name",
        "-print-multiarch",
        "-print-multi-directory",
        "-print-multi-lib",
        "-print-multi-os-directory",
        "-print-sysroot",
        "-print-sysroot-headers-suffix",
        "-print-target-triple",
        "-print-effective-triple",
        "-print-diagnostic-categories",
        "-print-enabled-extensions",
        "-print-resource-dir",
        "--print-resource-dir",
        "-print-runtime-dir",
        "-print-rocm-search-dirs",
        "-print-targets",
        "--print-targets",
        "-print-file-name",
        "--print-file-name",
        "-print-prog-name",
        "--print-prog-name",
        "--print-supported-cpus",
        "-###",
    }
)

_QUERY_PREFIXES = (
    "--help=",
    "--version=",
    "--autocomplete=",
    "-print-file-name=",
    "-print-prog-name=",
    "--print-file-name=",
    "--print-prog-name=",
    "--print-supported-cpus=",
)

# Options whose following argv element is data rather than an input file.
# This is used only to recognize argument-less driver queries; it never
# rewrites the compiler command.
_OPTIONS_WITH_VALUE = frozenset(
    {
        "-A",
        "-arch",
        "-aux-info",
        "-B",
        "--config",
        "-D",
        "-dumpbase",
        "-dumpbase-ext",
        "-dumpdir",
        "-e",
        "-F",
        "-gcc-toolchain",
        "--gcc-toolchain",
        "-I",
        "-idirafter",
        "-iframework",
        "-imacros",
        "-include",
        "-include-pch",
        "-iprefix",
        "-iquote",
        "-isysroot",
        "-isystem",
        "-isystem-after",
        "-iwithprefix",
        "-iwithprefixbefore",
        "-L",
        "-l",
        "-mllvm",
        "-MF",
        "-MJ",
        "-MQ",
        "-MT",
        "-o",
        "--output",
        "--param",
        "-resource-dir",
        "-specs",
        "--sysroot",
        "-T",
        "-target",
        "--target",
        "-U",
        "-u",
        "-working-directory",
        "-wrapper",
        "-Xassembler",
        "-Xclang",
        "-Xlinker",
        "-Xopenmp-target",
        "-Xpreprocessor",
        "-x",
        "-z",
    }
)

_LINK_INTENT_OPTIONS = frozenset(
    {
        "-l",
        "-pie",
        "-no-pie",
        "-r",
        "-shared",
        "--shared",
        "-static",
        "-static-pie",
        "-nostartfiles",
        "-nodefaultlibs",
        "-nostdlib",
    }
)

_FORWARDED_SIGNALS = tuple(
    sig
    for sig in (
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGQUIT", None),
        getattr(signal, "SIGTERM", None),
    )
    if sig is not None
)


class WrapperConfigurationError(ValueError):
    """Raised when the wrapper environment cannot be interpreted."""


def _error(message: str) -> None:
    print(f"linker_map_cc.py: error: {message}", file=sys.stderr)


def _selected_compiler(argv0: str, environ: os._Environ[str]) -> str:
    real_compiler = environ.get("REAL_COMPILER", "").strip()
    if not real_compiler:
        raise WrapperConfigurationError("REAL_COMPILER is required")

    wrapper_name = Path(argv0).name.lower()
    use_cxx = "cxx" in wrapper_name or "++" in wrapper_name
    if use_cxx:
        compiler = environ.get("REAL_CXX", "").strip() or real_compiler
    else:
        compiler = real_compiler

    prefix = environ.get("TOOLCHAIN_PREFIX", "")
    return f"{prefix}{compiler}" if prefix else compiler


def _driver_flags(environ: os._Environ[str]) -> list[str]:
    flags: list[str] = []
    sysroot = environ.get("SYSROOT", "")
    if sysroot:
        flags.append(f"--sysroot={sysroot}")

    raw_extra = environ.get("EXTRA_DRIVER_FLAGS", "")
    if raw_extra:
        try:
            flags.extend(shlex.split(raw_extra, posix=True))
        except ValueError as exc:
            raise WrapperConfigurationError(
                f"invalid EXTRA_DRIVER_FLAGS: {exc}"
            ) from exc
    return flags


def _expand_response_files(
    arguments: Sequence[str],
    cwd: Path,
    *,
    depth: int = 0,
    active: frozenset[Path] = frozenset(),
) -> list[str]:
    """Expand readable GCC/Clang response files for classification only."""

    if depth >= 8:
        return list(arguments)

    expanded: list[str] = []
    for argument in arguments:
        if not argument.startswith("@") or argument == "@":
            expanded.append(argument)
            continue

        response_path = Path(argument[1:])
        if not response_path.is_absolute():
            response_path = cwd / response_path
        response_path = response_path.resolve(strict=False)
        if response_path in active:
            expanded.append(argument)
            continue

        try:
            # Refuse unexpectedly large files: an unread response file is
            # simply left opaque, exactly as it is for the real driver.
            if response_path.stat().st_size > 16 * 1024 * 1024:
                expanded.append(argument)
                continue
            contents = response_path.read_text(
                encoding="utf-8", errors="surrogateescape"
            )
            nested = shlex.split(contents, posix=True)
        except (OSError, ValueError):
            expanded.append(argument)
            continue

        expanded.extend(
            _expand_response_files(
                nested,
                response_path.parent,
                depth=depth + 1,
                active=active | {response_path},
            )
        )
    return expanded


def _output_argument(arguments: Sequence[str]) -> tuple[str, bool]:
    output = "a.out"
    explicit = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument in {"-o", "--output"}:
            if index + 1 < len(arguments):
                output = arguments[index + 1]
                explicit = True
                index += 2
                continue
        elif argument.startswith("-o") and len(argument) > 2:
            output = argument[2:]
            explicit = True
        elif argument.startswith("--output="):
            output = argument.split("=", 1)[1]
            explicit = True
        if argument in _OPTIONS_WITH_VALUE:
            index += 2
            continue
        index += 1
    return output, explicit


def _has_input(arguments: Sequence[str]) -> bool:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return index + 1 < len(arguments)
        if argument in _OPTIONS_WITH_VALUE:
            index += 2
            continue
        if argument == "-":
            return True
        if not argument.startswith("-"):
            return True
        index += 1
    return False


def _driver_options(arguments: Sequence[str]) -> list[str]:
    """Return real driver options, excluding their separate values."""

    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument.startswith("-") and argument != "-":
            result.append(argument)
        if argument in _OPTIONS_WITH_VALUE:
            index += 2
        else:
            index += 1
    return result


def _language_mode(arguments: Sequence[str]) -> str | None:
    mode: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument == "-x":
            if index + 1 < len(arguments):
                mode = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("-x") and len(argument) > 2:
            mode = argument[2:]
        if argument in _OPTIONS_WITH_VALUE:
            index += 2
        else:
            index += 1
    return mode


def _linker_arguments(arguments: Sequence[str]) -> list[str]:
    """Return linker arguments embedded in common compiler-driver forms."""

    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument.startswith("-Wl,"):
            result.extend(argument[4:].split(","))
        elif argument == "-Xlinker" and index + 1 < len(arguments):
            result.append(arguments[index + 1])
            index += 1
        elif argument in {"-Map", "--Map"} or argument.startswith(
            ("-Map=", "--Map=")
        ):
            result.append(argument)
        index += 1
    return result


def _existing_linker_map(arguments: Sequence[str]) -> str | None:
    linker_arguments = _linker_arguments(arguments)
    for index, argument in enumerate(linker_arguments):
        for prefix in ("-Map=", "--Map="):
            if argument.startswith(prefix):
                value = argument[len(prefix) :]
                if value:
                    return value
        if argument in {"-Map", "--Map"} and index + 1 < len(
            linker_arguments
        ):
            value = linker_arguments[index + 1]
            if value:
                return value
    return None


def _is_query(
    arguments: Sequence[str], *, has_input: bool, explicit_output: bool
) -> bool:
    driver_options = _driver_options(arguments)
    if any(
        argument in _QUERY_OPTIONS
        or argument.startswith(_QUERY_PREFIXES)
        for argument in driver_options
    ):
        return True

    # Historical compiler probes used by configure.  With an actual input or
    # output, -v/-V may merely request verbose compilation and must not suppress
    # a real link.
    if not has_input and not explicit_output:
        if any(
            argument in {"-v", "-V", "-qversion"}
            for argument in driver_options
        ):
            return True
        linker_arguments = _linker_arguments(arguments)
        if any(
            argument in {"--help", "-help", "--version", "-v"}
            for argument in linker_arguments
        ):
            return True
    return False


def _is_final_link(arguments: Sequence[str]) -> bool:
    if not arguments:
        return False
    driver_options = _driver_options(arguments)
    if any(argument in _NON_LINKING_OPTIONS for argument in driver_options):
        return False

    # Header/PCH language modes produce compiler artifacts even without -c.
    if _language_mode(arguments) in {
        "c-header",
        "c++-header",
        "objective-c-header",
        "objective-c++-header",
    }:
        return False

    _, explicit_output = _output_argument(arguments)
    has_input = _has_input(arguments)
    if _is_query(
        arguments, has_input=has_input, explicit_output=explicit_output
    ):
        return False
    if explicit_output or has_input:
        return True
    return any(
        argument in _LINK_INTENT_OPTIONS
        or (argument.startswith("-l") and len(argument) > 2)
        or argument.startswith("-Wl,")
        for argument in driver_options
    )


def _absolute_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    # abspath normalizes the spelling without following a symlink used as the
    # output itself; the compiler receives the original spelling unchanged.
    return Path(os.path.abspath(path))


def _mirrored_map_path(output: Path, root: Path) -> Path:
    # Mirroring the absolute output path keeps maps for separate build trees
    # distinct even when every compiler invocation uses the same basename.
    relative_parts = output.parts[1:] if output.is_absolute() else output.parts
    mirrored_output = root.joinpath(*relative_parts)
    return Path(f"{mirrored_output}.map")


def _map_path(output: Path, cwd: Path, environ: os._Environ[str]) -> Path:
    raw_root = environ.get("LINK_MAP_ROOT", "")
    if not raw_root:
        return Path(f"{output}.map")
    root = _absolute_path(raw_root, cwd)
    return _mirrored_map_path(output, root)


def _timestamp_utc() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _run_with_signal_forwarding(command: Sequence[str]) -> int:
    try:
        process = subprocess.Popen(command)
    except FileNotFoundError:
        _error(f"compiler not found: {command[0]}")
        return 127
    except PermissionError:
        _error(f"compiler is not executable: {command[0]}")
        return 126
    except OSError as exc:
        _error(f"cannot execute {command[0]}: {exc}")
        return 126

    previous_handlers: dict[signal.Signals, object] = {}
    received_signal: list[int] = []

    def forward(signum: int, _frame: object) -> None:
        received_signal.append(signum)
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    try:
        for forwarded_signal in _FORWARDED_SIGNALS:
            previous_handlers[forwarded_signal] = signal.getsignal(forwarded_signal)
            signal.signal(forwarded_signal, forward)
        return_code = process.wait()
    finally:
        for forwarded_signal, previous_handler in previous_handlers.items():
            signal.signal(forwarded_signal, previous_handler)

    if return_code < 0:
        return 128 + -return_code
    if received_signal and return_code == 0:
        return 128 + received_signal[-1]
    return return_code


def _is_discarded_output(output: Path) -> bool:
    try:
        return output == Path(os.devnull).resolve(strict=False)
    except OSError:
        return str(output) == os.devnull


def main(argv: Sequence[str] | None = None) -> int:
    user_arguments = list(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd().resolve(strict=False)

    try:
        compiler = _selected_compiler(sys.argv[0], os.environ)
        driver_flags = _driver_flags(os.environ)
    except WrapperConfigurationError as exc:
        _error(str(exc))
        return 2

    effective_arguments = [*driver_flags, *user_arguments]
    analysis_arguments = _expand_response_files(effective_arguments, cwd)
    if (
        os.environ.get("DYNAMIC_SHARED_HELPERS") == "1"
        and any(value in {"-shared", "--shared"} for value in analysis_arguments)
    ):
        effective_arguments = [
            value
            for value in effective_arguments
            if value not in {"-static", "-static-pie"}
        ]
        analysis_arguments = _expand_response_files(effective_arguments, cwd)
    output_argument, _ = _output_argument(analysis_arguments)
    output = _absolute_path(output_argument, cwd)

    command = [compiler, *effective_arguments]
    final_link = _is_final_link(analysis_arguments) and not _is_discarded_output(
        output
    )
    linker_map: Path | None = None

    if final_link:
        existing_map = _existing_linker_map(analysis_arguments)
        if existing_map is not None:
            linker_map = _absolute_path(existing_map, cwd)
        else:
            linker_map = _map_path(output, cwd, os.environ)
            if os.environ.get("LINK_MAP_ROOT", ""):
                try:
                    linker_map.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    _error(f"cannot create linker-map directory: {exc}")
                    return 1
            command.append(f"-Wl,-Map={linker_map},--cref")

    return_code = _run_with_signal_forwarding(command)
    if return_code != 0 or not final_link or linker_map is None:
        return return_code

    metadata_path = Path(f"{output}.link.json")
    metadata: dict[str, object] = {
        "command": command,
        "compiler": compiler,
        "cwd": str(cwd),
        "map": str(linker_map),
        "output": str(output),
        "timestamp": _timestamp_utc(),
    }
    try:
        _write_json_atomic(metadata_path, metadata)
    except OSError as exc:
        _error(f"cannot write link metadata {metadata_path}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
