#!/usr/bin/env python3
"""Persistent cache for radare2 analysis and PalmTree block embeddings.

Cache files are local, trusted pickle files.  Their key includes every input
that changes parsing or embeddings, but deliberately excludes matching
thresholds so threshold-only experiments can reuse the expensive work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any

from asm import CodeUnit, parse_r2_file
from model import PalmTree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE_DIR = (
    REPO_ROOT.parent
    / f"{REPO_ROOT.name}_artifacts"
    / "function_matching_cache"
)

# Bump this only for a serialization-layout change.  Changes to parsing and
# embedding source files are detected separately by implementation_signature.
CACHE_FORMAT_VERSION = 1
IMPLEMENTATION_FILES = (
    SCRIPT_DIR / "asm.py",
    SCRIPT_DIR / "model.py",
    SCRIPT_DIR / "palmtree" / "eval_utils.py",
    SCRIPT_DIR / "palmtree" / "vocab.py",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_file_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def radare2_version() -> str:
    """Return a stable one-line r2 version, including an unavailable marker."""
    try:
        result = subprocess.run(
            ["r2", "-v"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable:{type(error).__name__}"
    return (result.stdout or result.stderr).splitlines()[0].strip()


def has_complete_embeddings(unit: CodeUnit) -> bool:
    """True when every non-empty block has a cached embedding."""
    return all(
        block.embedding is not None
        for function in unit.functions
        for block in function.blocks
        if block.instructions
    )


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    writes: int


class AnalysisEmbeddingCache:
    """Content-addressed store for fully embedded ``CodeUnit`` objects."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    @staticmethod
    def key(identity: dict[str, Any]) -> str:
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def entry_path(self, identity: dict[str, Any]) -> Path:
        key = self.key(identity)
        return self.root / "entries" / key[:2] / f"{key}.pickle"

    def load(self, identity: dict[str, Any], file_path: Path) -> CodeUnit | None:
        path = self.entry_path(identity)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, dict):
                return None
            if payload.get("cache_format") != CACHE_FORMAT_VERSION:
                return None
            if payload.get("identity") != identity:
                return None
            unit = payload.get("code_unit")
            if not isinstance(unit, CodeUnit) or not has_complete_embeddings(unit):
                return None
        except (
            OSError,
            EOFError,
            pickle.PickleError,
            AttributeError,
            ImportError,
            TypeError,
            ValueError,
        ):
            return None

        # Object members are extracted into a new temporary directory on each
        # run.  Keep cached analysis but expose the current path/name.
        unit.file_path = file_path.as_posix()
        unit.name = file_path.name
        unit.unit_type = str(identity["unit_type"])
        unit.type = unit.unit_type
        return unit

    def store(self, identity: dict[str, Any], unit: CodeUnit) -> Path:
        if not has_complete_embeddings(unit):
            raise ValueError("Refusing to cache a CodeUnit with missing embeddings")

        destination = self.entry_path(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_format": CACHE_FORMAT_VERSION,
            "identity": identity,
            "code_unit": unit,
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        return destination


class CachedCodeUnitLoader:
    """Load cached CodeUnits and lazily run r2/PalmTree only on cache misses."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str,
        pooling: str,
        asm_normalization: str,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        write_cache: bool = True,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PalmTree model not found: {self.model_path}")
        self.vocab_path = SCRIPT_DIR / "palmtree" / "model" / "vocab"
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"PalmTree vocabulary not found: {self.vocab_path}")

        self.device = device
        self.pooling = pooling
        self.asm_normalization = asm_normalization
        self.cache = AnalysisEmbeddingCache(cache_dir) if cache_dir is not None else None
        self.write_cache = bool(write_cache)
        self._model: PalmTree | None = None
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._common_identity = {
            "cache_format": CACHE_FORMAT_VERSION,
            "asm_normalization": asm_normalization,
            "palmtree_pooling": pooling,
            "palmtree_model_sha256": sha256_file(self.model_path),
            "palmtree_vocab_sha256": sha256_file(self.vocab_path),
            "implementation_sha256": _combined_file_hash(IMPLEMENTATION_FILES),
            "radare2_version": radare2_version(),
        }

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def stats(self) -> CacheStats:
        return CacheStats(self._hits, self._misses, self._writes)

    def identity(self, file_path: Path | str, unit_type: str) -> dict[str, Any]:
        path = Path(file_path)
        stat = path.stat()
        return {
            **self._common_identity,
            "binary_sha256": sha256_file(path),
            "binary_size": stat.st_size,
            "unit_type": unit_type,
        }

    def _get_model(self) -> PalmTree:
        if self._model is None:
            model = PalmTree("Palm Tree")
            model.load(
                self.model_path.as_posix(),
                device=self.device,
                pooling=self.pooling,
            )
            self._model = model
        return self._model

    def load(self, file_path: Path | str, *, unit_type: str) -> CodeUnit:
        path = Path(file_path).resolve()
        identity = self.identity(path, unit_type)
        if self.cache is not None:
            cached = self.cache.load(identity, path)
            if cached is not None:
                self._hits += 1
                return cached

        self._misses += 1
        unit = parse_r2_file(
            path.as_posix(),
            asm_model=self._get_model(),
            unit_type=unit_type,
            asm_normalization=self.asm_normalization,
        )
        if self.cache is not None and self.write_cache:
            self.cache.store(identity, unit)
            self._writes += 1
        return unit
