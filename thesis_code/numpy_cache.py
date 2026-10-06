#!/usr/bin/env python3
"""Memory-mappable serialization for matching-ready ``CodeUnit`` features.

The format deliberately stores exactly the evidence consumed by the matching
and experiment pipeline: function/block boundaries, instruction counts, block
embeddings, internal/external calls, symbol visibility, and .rodata.  Original
instruction text, CFG edges, and raw block bytes are analysis intermediates;
their counts/sizes are retained and the canonical ELF/archive remains the
source from which they can be regenerated.
"""

from __future__ import annotations

from collections.abc import Sequence
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from asm import Block, CodeUnit, Function


NUMPY_STORAGE_FORMAT = 1
METADATA_FILE = "metadata.json"
ENTRY_FILES = {
    METADATA_FILE,
    "functions.npy",
    "blocks.npy",
    "embeddings.npy",
    "call_targets.npy",
    "external_string_indices.npy",
    "rodata_string_indices.npy",
    "string_offsets.npy",
    "strings.bin",
    "rodata.bin",
}

FUNCTION_ADDRESS = 0
FUNCTION_SIZE = 1
FUNCTION_BLOCK_START = 2
FUNCTION_BLOCK_STOP = 3
FUNCTION_CALL_START = 4
FUNCTION_CALL_STOP = 5
FUNCTION_EXTERNAL_START = 6
FUNCTION_EXTERNAL_STOP = 7
FUNCTION_NAME_INDEX = 8
FUNCTION_IS_GLOBAL = 9
FUNCTION_FIELD_COUNT = 10

BLOCK_ADDRESS = 0
BLOCK_INSTRUCTION_COUNT = 1
BLOCK_RAW_SIZE = 2
BLOCK_EMBEDDING_START = 3
BLOCK_EMBEDDING_STOP = 4
BLOCK_FIELD_COUNT = 5


class CountedInstructions(Sequence[str]):
    """A zero-payload sequence retaining a cached block's instruction count."""

    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [""] * len(range(*index.indices(self.count)))
        if not -self.count <= index < self.count:
            raise IndexError(index)
        return ""


class SizedRawBytes(Sequence[int]):
    """A zero-payload byte-like sequence retaining only a block's byte size."""

    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return bytes(len(range(*index.indices(self.size))))
        if not -self.size <= index < self.size:
            raise IndexError(index)
        return 0


def entry_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def is_complete_entry(
    directory: Path, expected_identity: dict[str, Any] | None = None
) -> bool:
    """Cheap structural check used before deciding whether analysis is needed."""
    try:
        if not directory.is_dir():
            return False
        if not ENTRY_FILES.issubset(item.name for item in directory.iterdir()):
            return False
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        return (
            metadata.get("storage_format") == NUMPY_STORAGE_FORMAT
            and isinstance(metadata.get("identity"), dict)
            and (
                expected_identity is None
                or metadata["identity"] == expected_identity
            )
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_string_table(directory: Path, strings: list[str]) -> None:
    offsets = np.zeros(len(strings) + 1, dtype=np.uint64)
    with (directory / "strings.bin").open("wb") as stream:
        position = 0
        for index, value in enumerate(strings):
            encoded = value.encode("utf-8", errors="surrogatepass")
            stream.write(encoded)
            position += len(encoded)
            offsets[index + 1] = position
    np.save(directory / "string_offsets.npy", offsets, allow_pickle=False)


def _embedding_layout(unit: CodeUnit) -> tuple[np.dtype, int]:
    dtype: np.dtype | None = None
    total_values = 0
    for function in unit.functions:
        for block in function.blocks:
            if block.embedding is None:
                continue
            array = np.asarray(block.embedding)
            dtype = array.dtype if dtype is None else np.result_type(dtype, array.dtype)
            total_values += array.size
    return np.dtype(dtype or np.float32), total_values


def write_entry(destination: Path, identity: dict[str, Any], unit: CodeUnit) -> Path:
    """Atomically write one entry, safely across concurrent cache builders."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A small fixed lock pool (one file per two-hex-digit key prefix) avoids
    # tens of thousands of lock inodes while preventing same-key publication
    # races between sharded builders.
    lock_path = destination.parent / ".numpy-cache-write.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            return _write_entry_locked(destination, identity, unit)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _write_entry_locked(
    destination: Path, identity: dict[str, Any], unit: CodeUnit
) -> Path:
    if is_complete_entry(destination, identity):
        return destination

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        function_count = len(unit.functions)
        block_count = sum(len(function.blocks) for function in unit.functions)
        functions = np.zeros((function_count, FUNCTION_FIELD_COUNT), dtype=np.uint64)
        blocks = np.zeros((block_count, BLOCK_FIELD_COUNT), dtype=np.uint64)
        call_targets: list[int] = []
        external_string_indices: list[int] = []
        strings: list[str] = []

        embedding_dtype, embedding_value_count = _embedding_layout(unit)
        embeddings = np.lib.format.open_memmap(
            temporary / "embeddings.npy",
            mode="w+",
            dtype=embedding_dtype,
            shape=(embedding_value_count,),
        )

        block_index = 0
        embedding_position = 0
        for function_index, function in enumerate(unit.functions):
            name_index = len(strings)
            strings.append(str(function.name))
            function_block_start = block_index
            for block in function.blocks:
                array = None if block.embedding is None else np.asarray(block.embedding)
                embedding_start = embedding_position
                if array is not None:
                    flattened = array.reshape(-1)
                    embedding_position += flattened.size
                    embeddings[embedding_start:embedding_position] = flattened
                blocks[block_index] = (
                    int(block.address),
                    len(block.instructions),
                    len(block.raw_bytes) if block.raw_bytes is not None else 0,
                    embedding_start,
                    embedding_position,
                )
                block_index += 1

            function_call_start = len(call_targets)
            call_targets.extend(sorted(int(value) for value in function.call_targets))
            function_call_stop = len(call_targets)

            function_external_start = len(external_string_indices)
            for symbol in sorted(str(value) for value in function.external_call_symbols):
                external_string_indices.append(len(strings))
                strings.append(symbol)
            function_external_stop = len(external_string_indices)

            functions[function_index] = (
                int(function.address),
                int(function.size),
                function_block_start,
                block_index,
                function_call_start,
                function_call_stop,
                function_external_start,
                function_external_stop,
                name_index,
                int(bool(function.is_global_symbol)),
            )
        embeddings.flush()
        del embeddings

        rodata_string_start = len(strings)
        strings.extend(str(value) for value in unit.rodata_strings)
        rodata_string_stop = len(strings)
        _write_string_table(temporary, strings)

        np.save(temporary / "functions.npy", functions, allow_pickle=False)
        np.save(temporary / "blocks.npy", blocks, allow_pickle=False)
        np.save(
            temporary / "call_targets.npy",
            np.asarray(call_targets, dtype=np.uint64),
            allow_pickle=False,
        )
        np.save(
            temporary / "external_string_indices.npy",
            np.asarray(external_string_indices, dtype=np.uint64),
            allow_pickle=False,
        )
        np.save(
            temporary / "rodata_string_indices.npy",
            np.arange(rodata_string_start, rodata_string_stop, dtype=np.uint64),
            allow_pickle=False,
        )
        with (temporary / "rodata.bin").open("wb") as stream:
            stream.write(bytes(unit.rodata_bytes))

        metadata = {
            "storage_format": NUMPY_STORAGE_FORMAT,
            "identity": identity,
            "unit": {
                "name": unit.name,
                "file_path": unit.file_path,
                "unit_type": unit.unit_type,
                "symbol_fallback_count": int(unit.symbol_fallback_count),
                "pseudo_block_fallback_count": int(unit.pseudo_block_fallback_count),
                "rodata_section_count": int(unit.rodata_section_count),
            },
            "counts": {
                "functions": function_count,
                "blocks": block_count,
                "embedding_values": embedding_value_count,
                "call_targets": len(call_targets),
                "external_call_symbols": len(external_string_indices),
                "rodata_strings": len(unit.rodata_strings),
                "rodata_bytes": len(unit.rodata_bytes),
            },
            "matching_features": {
                "block_embeddings": True,
                "instruction_counts": True,
                "function_boundaries": True,
                "numeric_call_graph": True,
                "external_call_symbols": True,
                "symbol_visibility": True,
                "rodata_bytes": True,
                "rodata_strings": True,
            },
            "regenerable_intermediates_omitted": [
                "normalized_instruction_text",
                "function_cfg_edges",
                "raw_block_bytes",
            ],
        }
        # Written last: its presence is the entry's commit marker.
        _write_json(temporary / METADATA_FILE, metadata)

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


class _StringTable:
    def __init__(self, directory: Path):
        # Offsets and strings are consumed while reconstructing a CodeUnit and
        # are not referenced afterwards.  Loading these small tables eagerly
        # avoids retaining two file descriptors per cached compilation unit.
        self.offsets = np.load(
            directory / "string_offsets.npy", allow_pickle=False
        )
        self.data = (directory / "strings.bin").read_bytes()

    def get(self, index: int) -> str:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.data[start:stop].decode("utf-8", errors="surrogatepass")


def read_entry(
    directory: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], CodeUnit]:
    """Read one entry while keeping its dominant embedding payload mmap-backed."""
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    if metadata.get("storage_format") != NUMPY_STORAGE_FORMAT:
        raise ValueError("unsupported NumPy cache storage format")
    identity = metadata.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("missing cache identity")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("cache identity mismatch")

    # These structural arrays are only needed while rebuilding the Python
    # objects below.  Keeping each one memory-mapped used to retain roughly
    # ten descriptors per CU and exhausted RLIMIT_NOFILE on large archives.
    functions_data = np.load(directory / "functions.npy", allow_pickle=False)
    blocks_data = np.load(directory / "blocks.npy", allow_pickle=False)
    embeddings = np.load(
        directory / "embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    call_targets = np.load(
        directory / "call_targets.npy", allow_pickle=False
    )
    external_indices = np.load(
        directory / "external_string_indices.npy", allow_pickle=False
    )
    rodata_string_indices = np.load(
        directory / "rodata_string_indices.npy", allow_pickle=False
    )
    strings = _StringTable(directory)

    # Embeddings dominate cache size and remain mmap-backed.  All other entry
    # data is small or is already copied into Function/Block objects.
    resources: list[Any] = [embeddings]
    functions: list[Function] = []
    for function_row in functions_data:
        function_blocks: list[Block] = []
        for block_index in range(
            int(function_row[FUNCTION_BLOCK_START]),
            int(function_row[FUNCTION_BLOCK_STOP]),
        ):
            block_row = blocks_data[block_index]
            block = Block(
                address=int(block_row[BLOCK_ADDRESS]),
                instructions=CountedInstructions(
                    int(block_row[BLOCK_INSTRUCTION_COUNT])
                ),
                raw_bytes=SizedRawBytes(int(block_row[BLOCK_RAW_SIZE])),
            )
            embedding_start = int(block_row[BLOCK_EMBEDDING_START])
            embedding_stop = int(block_row[BLOCK_EMBEDDING_STOP])
            if embedding_stop > embedding_start:
                block.embedding = embeddings[embedding_start:embedding_stop]
            function_blocks.append(block)

        external_symbols = {
            strings.get(int(external_indices[index]))
            for index in range(
                int(function_row[FUNCTION_EXTERNAL_START]),
                int(function_row[FUNCTION_EXTERNAL_STOP]),
            )
        }
        numeric_targets = {
            int(call_targets[index])
            for index in range(
                int(function_row[FUNCTION_CALL_START]),
                int(function_row[FUNCTION_CALL_STOP]),
            )
        }
        functions.append(
            Function(
                name=strings.get(int(function_row[FUNCTION_NAME_INDEX])),
                address=int(function_row[FUNCTION_ADDRESS]),
                size=int(function_row[FUNCTION_SIZE]),
                blocks=function_blocks,
                call_targets=numeric_targets,
                external_call_symbols=external_symbols,
                is_global_symbol=bool(function_row[FUNCTION_IS_GLOBAL]),
            )
        )

    rodata_data = (directory / "rodata.bin").read_bytes()
    rodata_strings = [strings.get(int(index)) for index in rodata_string_indices]
    unit_metadata = metadata["unit"]
    unit = CodeUnit(
        name=str(unit_metadata["name"]),
        file_path=str(unit_metadata["file_path"]),
        functions=functions,
        unit_type=str(unit_metadata["unit_type"]),
        symbol_fallback_count=int(unit_metadata["symbol_fallback_count"]),
        pseudo_block_fallback_count=int(unit_metadata["pseudo_block_fallback_count"]),
        rodata_bytes=rodata_data,
        rodata_strings=rodata_strings,
        rodata_section_count=int(unit_metadata["rodata_section_count"]),
    )
    # Keep mmap/file handles alive as long as the CodeUnit and its views exist.
    unit._numpy_cache_resources = resources
    unit._numpy_cache_directory = directory.as_posix()
    return identity, unit


def validate_matching_equivalence(original: CodeUnit, converted: CodeUnit) -> None:
    """Raise if any feature consumed by matching changed during conversion."""
    scalar_fields = (
        "unit_type",
        "symbol_fallback_count",
        "pseudo_block_fallback_count",
        "rodata_section_count",
    )
    for field in scalar_fields:
        if getattr(original, field) != getattr(converted, field):
            raise ValueError(f"unit field differs: {field}")
    if bytes(original.rodata_bytes) != bytes(converted.rodata_bytes):
        raise ValueError(".rodata bytes differ")
    if list(original.rodata_strings) != list(converted.rodata_strings):
        raise ValueError(".rodata strings differ")
    if len(original.functions) != len(converted.functions):
        raise ValueError("function count differs")

    for function_index, (left_function, right_function) in enumerate(
        zip(original.functions, converted.functions)
    ):
        for field in ("name", "address", "size", "is_global_symbol"):
            if getattr(left_function, field) != getattr(right_function, field):
                raise ValueError(f"function {function_index} field differs: {field}")
        if left_function.call_targets != right_function.call_targets:
            raise ValueError(f"function {function_index} call targets differ")
        if left_function.external_call_symbols != right_function.external_call_symbols:
            raise ValueError(f"function {function_index} external symbols differ")
        if left_function.resolved_call_targets != right_function.resolved_call_targets:
            raise ValueError(f"function {function_index} resolved calls differ")
        if len(left_function.blocks) != len(right_function.blocks):
            raise ValueError(f"function {function_index} block count differs")

        for block_index, (left_block, right_block) in enumerate(
            zip(left_function.blocks, right_function.blocks)
        ):
            if left_block.address != right_block.address:
                raise ValueError(
                    f"function {function_index} block {block_index} address differs"
                )
            if len(left_block.instructions) != len(right_block.instructions):
                raise ValueError(
                    f"function {function_index} block {block_index} instruction count differs"
                )
            if len(left_block.raw_bytes or b"") != len(right_block.raw_bytes or b""):
                raise ValueError(
                    f"function {function_index} block {block_index} raw size differs"
                )
            if (left_block.embedding is None) != (right_block.embedding is None):
                raise ValueError(
                    f"function {function_index} block {block_index} embedding presence differs"
                )
            if left_block.embedding is not None and not np.array_equal(
                np.asarray(left_block.embedding).reshape(-1),
                np.asarray(right_block.embedding).reshape(-1),
            ):
                raise ValueError(
                    f"function {function_index} block {block_index} embedding differs"
                )
