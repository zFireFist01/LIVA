#!/usr/bin/env python3
"""Persistent mmap cache for radare2 analysis and PalmTree block embeddings.

New entries use matching-ready NumPy arrays. Trusted legacy pickle entries are
read only as a migration fallback. Keys include every input that changes
parsing or embeddings, but deliberately exclude matching thresholds so
threshold-only experiments can reuse the expensive work.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any

from asm import CodeUnit, parse_r2_file
from model import PalmTree
from numpy_cache import is_complete_entry, read_entry, write_entry


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "Dataset" / "cache"
PACKAGED_IDENTITY_MANIFEST = "experiment_elf_provenance.jsonl"

# This is the analysis-feature identity version, not the on-disk storage
# version. Moving the same features from pickle to NumPy must preserve keys.
# Parsing/embedding changes are detected separately by implementation_signature.
CACHE_FORMAT_VERSION = 1
IMPLEMENTATION_FILES = (
    SCRIPT_DIR / "asm.py",
    SCRIPT_DIR / "model.py",
    SCRIPT_DIR / "palmtree" / "eval_utils.py",
    SCRIPT_DIR / "palmtree" / "vocab.py",
)

# These parser-only configuration changes deliberately suppress DWARF and
# radare2 local-variable bookkeeping.  Neither is persisted in CodeUnit nor
# consumed by matching, so entries produced before the change remain feature
# compatible.  Keep this mapping exact: any later implementation edit gets a
# new signature normally and cannot silently reuse older entries.
FEATURE_EQUIVALENT_IMPLEMENTATION_SIGNATURES = {
    "2a1822e8422264ff62d196083aca119aa141ac1edebeda33c8824e4dbb7af49e": (
        "f078f6afa29fd0bbac369b5e7262699e70f39f9ec48bf5295efa9729d34b0296"
    ),
}


class CacheOnlyMissError(RuntimeError):
    """Raised when a read-only packaged cache cannot satisfy a request."""


def packaged_common_identity(cache_dir: Path | str) -> dict[str, Any]:
    """Read the cache-creator identity without consulting local radare2."""
    cache_root = Path(cache_dir).expanduser().resolve()
    provenance_path = cache_root / PACKAGED_IDENTITY_MANIFEST
    try:
        with provenance_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    break
            else:
                raise ValueError("manifest is empty")
        if not isinstance(record, dict):
            raise ValueError("first record is not a JSON object")
        identity = record.get("cache_identity")
        if not isinstance(identity, dict):
            raise ValueError("first record has no cache_identity object")
        common_identity = {
            key: value
            for key, value in identity.items()
            if key not in {"binary_sha256", "binary_size", "unit_type"}
        }
        required = {
            "cache_format",
            "asm_normalization",
            "palmtree_pooling",
            "palmtree_model_sha256",
            "palmtree_vocab_sha256",
            "implementation_sha256",
            "radare2_version",
        }
        missing = sorted(required - common_identity.keys())
        if missing:
            raise ValueError(f"cache identity is missing: {', '.join(missing)}")
        return common_identity
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid packaged cache identity in {provenance_path}: {error}"
        ) from error


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


def implementation_signature() -> str:
    raw_signature = _combined_file_hash(IMPLEMENTATION_FILES)
    return FEATURE_EQUIVALENT_IMPLEMENTATION_SIGNATURES.get(
        raw_signature,
        raw_signature,
    )


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

    def legacy_entry_path(self, identity: dict[str, Any]) -> Path:
        key = self.key(identity)
        return self.root / "entries" / key[:2] / f"{key}.pickle"

    def entry_path(self, identity: dict[str, Any]) -> Path:
        key = self.key(identity)
        return self.root / "entries" / key[:2] / f"{key}.numpy"

    def function_count(self, identity: dict[str, Any]) -> int | None:
        """Read a CU's function count without mapping its numeric payload."""
        metadata_path = self.entry_path(identity) / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("identity") != identity:
                return None
            count = metadata["counts"]["functions"]
            if not isinstance(count, int) or count < 0:
                return None
            return count
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Legacy pickle entries do not expose cheap metadata.  Callers
            # safely fall back to loading those entries before filtering.
            return None

    def archive_index_path(self, archive_sha256: str) -> Path:
        return (
            self.root
            / "archives"
            / archive_sha256[:2]
            / f"{archive_sha256}.json"
        )

    def load(self, identity: dict[str, Any], file_path: Path) -> CodeUnit | None:
        path = self.entry_path(identity)
        unit = None
        if is_complete_entry(path, identity):
            try:
                _stored_identity, unit = read_entry(
                    path, expected_identity=identity
                )
                if not isinstance(unit, CodeUnit) or not has_complete_embeddings(unit):
                    unit = None
            except (
                OSError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                unit = None
        if unit is None:
            legacy_path = self.legacy_entry_path(identity)
            if not legacy_path.is_file():
                return None
            try:
                with legacy_path.open("rb") as stream:
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

    def contains(self, identity: dict[str, Any]) -> bool:
        """Return whether a structurally complete NumPy or legacy entry exists."""
        if is_complete_entry(self.entry_path(identity), identity):
            return True
        return self.legacy_entry_path(identity).is_file()

    @contextmanager
    def analysis_lock(self, identity: dict[str, Any]):
        """Serialize creation of the same content-addressed identity."""
        key = self.key(identity)
        lock_path = self.root / "locks" / key[:2] / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def store(self, identity: dict[str, Any], unit: CodeUnit) -> Path:
        if not has_complete_embeddings(unit):
            raise ValueError("Refusing to cache a CodeUnit with missing embeddings")

        destination = self.entry_path(identity)
        return write_entry(destination, identity, unit)


class CachedCodeUnitLoader:
    """Load cached units, optionally forbidding all analysis on cache misses."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str,
        pooling: str,
        asm_normalization: str,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        write_cache: bool = True,
        cache_only: bool = False,
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
        self.cache_only = bool(cache_only)
        if self.cache_only and cache_dir is None:
            raise ValueError("cache-only mode requires a cache directory")
        self.cache = AnalysisEmbeddingCache(cache_dir) if cache_dir is not None else None
        self.write_cache = bool(write_cache) and not self.cache_only
        self._model: PalmTree | None = None
        self._hits = 0
        self._misses = 0
        self._writes = 0
        local_identity = {
            "cache_format": CACHE_FORMAT_VERSION,
            "asm_normalization": asm_normalization,
            "palmtree_pooling": pooling,
            "palmtree_model_sha256": sha256_file(self.model_path),
            "palmtree_vocab_sha256": sha256_file(self.vocab_path),
            "implementation_sha256": implementation_signature(),
        }
        if self.cache_only:
            assert cache_dir is not None
            cached_identity = packaged_common_identity(cache_dir)
            incompatible = {
                key: (cached_identity.get(key), expected)
                for key, expected in local_identity.items()
                if cached_identity.get(key) != expected
            }
            if incompatible:
                details = ", ".join(
                    f"{key}=cached:{cached!r}/local:{local!r}"
                    for key, (cached, local) in incompatible.items()
                )
                raise ValueError(
                    f"packaged cache is incompatible with this runtime: {details}"
                )
            self._common_identity = cached_identity
        else:
            self._common_identity = {
                **local_identity,
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

    def load(
        self,
        file_path: Path | str,
        *,
        unit_type: str,
        identity: dict[str, Any] | None = None,
    ) -> CodeUnit:
        path = Path(file_path).resolve()
        identity = identity or self.identity(path, unit_type)
        if identity.get("unit_type") != unit_type:
            raise ValueError("precomputed identity has the wrong unit type")
        if self.cache is not None:
            cached = self.cache.load(identity, path)
            if cached is not None:
                self._hits += 1
                return cached
            if self.cache_only:
                self._misses += 1
                raise CacheOnlyMissError(
                    f"cache-only miss for {unit_type} {path} "
                    f"(key={self.cache.key(identity)})"
                )
            # Cross-process double-check after acquiring the creation lock.
            with self.cache.analysis_lock(identity):
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
                if self.write_cache:
                    self.cache.store(identity, unit)
                    self._writes += 1
                return unit

        self._misses += 1
        return parse_r2_file(
            path.as_posix(),
            asm_model=self._get_model(),
            unit_type=unit_type,
            asm_normalization=self.asm_normalization,
        )

    def load_archive(
        self,
        archive_path: Path | str,
        *,
        min_functions: int = 0,
    ) -> list[CodeUnit] | None:
        """Load indexed members, skipping undersized CUs before mmap loading."""
        inspected = self.inspect_archive_index(
            archive_path,
            min_functions=min_functions,
        )
        if inspected is None:
            if self.cache_only:
                raise CacheOnlyMissError(
                    f"cache-only archive index miss for "
                    f"{Path(archive_path).expanduser().resolve()}"
                )
            return None
        index_path, index = inspected
        index = self.ensure_archive_function_counts(index_path, index)
        archive = Path(archive_path).expanduser().resolve()
        try:
            units: list[CodeUnit] = []
            for member in index["members"]:
                identity = {
                    **self._common_identity,
                    "binary_sha256": member["binary_sha256"],
                    "binary_size": member["binary_size"],
                    "unit_type": CodeUnit.TYPE_CU,
                }
                function_count = member.get("function_count")
                if (
                    isinstance(function_count, int)
                    and function_count < min_functions
                ):
                    continue
                occurrence = int(member.get("occurrence", 1))
                synthetic_name = str(member["name"])
                if occurrence > 1:
                    synthetic_name += f"#{occurrence}"
                synthetic_path = Path(f"{archive.as_posix()}!/{synthetic_name}")
                unit = self.cache.load(identity, synthetic_path)
                if unit is None:
                    if self.cache_only:
                        raise CacheOnlyMissError(
                            f"cache-only CU miss for {synthetic_path} "
                            f"(key={self.cache.key(identity)})"
                        )
                    return None
                if unit.get_num_functions() >= min_functions:
                    units.append(unit)
        except CacheOnlyMissError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if self.cache_only:
                raise CacheOnlyMissError(
                    f"invalid cache-only archive data for {archive}"
                ) from error
            return None
        self._hits += len(units)
        return units

    def ensure_archive_function_counts(
        self,
        index_path: Path,
        index: dict[str, Any],
    ) -> dict[str, Any]:
        """Upgrade an archive index using only per-entry metadata files."""
        if self.cache is None:
            return index
        changed = False
        for member in index.get("members", []):
            count = member.get("function_count")
            if isinstance(count, int) and count >= 0:
                continue
            try:
                identity = {
                    **self._common_identity,
                    "binary_sha256": member["binary_sha256"],
                    "binary_size": member["binary_size"],
                    "unit_type": CodeUnit.TYPE_CU,
                }
            except (KeyError, TypeError):
                continue
            count = self.cache.function_count(identity)
            if count is None:
                continue
            member["function_count"] = count
            changed = True

        if not changed or self.cache_only:
            return index

        index["schema"] = 2
        temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, index_path)
        finally:
            temporary.unlink(missing_ok=True)
        return index

    def inspect_archive_index(
        self,
        archive_path: Path | str,
        *,
        min_functions: int = 0,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Validate an archive index without mapping CodeUnit payloads.

        When function counts are present, entries that the caller will discard
        are not touched on disk.  Cache-building/audit callers retain the
        default and therefore validate every member.
        """
        if self.cache is None:
            return None
        archive = Path(archive_path).expanduser().resolve()
        archive_sha256 = sha256_file(archive)
        index_path = self.cache.archive_index_path(archive_sha256)
        if not index_path.is_file():
            return None
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if (
                index.get("archive_sha256") != archive_sha256
                or index.get("archive_size") != archive.stat().st_size
                or index.get("common_identity") != self._common_identity
            ):
                return None
            for member in index["members"]:
                identity = {
                    **self._common_identity,
                    "binary_sha256": member["binary_sha256"],
                    "binary_size": member["binary_size"],
                    "unit_type": CodeUnit.TYPE_CU,
                }
                if self.cache.key(identity) != member["cache_key"]:
                    return None
                function_count = member.get("function_count")
                if (
                    isinstance(function_count, int)
                    and function_count < min_functions
                ):
                    continue
                if not self.cache.contains(identity):
                    return None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return index_path, index

    def store_archive_index(
        self,
        archive_path: Path | str,
        members: list[tuple[str, int, dict[str, Any]]],
    ) -> Path | None:
        """Atomically record archive provenance and its member cache keys."""
        if self.cache is None:
            return None
        archive = Path(archive_path).expanduser().resolve()
        archive_sha256 = sha256_file(archive)
        destination = self.cache.archive_index_path(archive_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        member_rows = []
        for name, occurrence, identity in members:
            member = {
                "name": name,
                "occurrence": occurrence,
                "binary_sha256": identity["binary_sha256"],
                "binary_size": identity["binary_size"],
                "cache_key": self.cache.key(identity),
            }
            function_count = self.cache.function_count(identity)
            if function_count is not None:
                member["function_count"] = function_count
            member_rows.append(member)

        payload = {
            "schema": 2,
            "archive_sha256": archive_sha256,
            "archive_size": archive.stat().st_size,
            "common_identity": self._common_identity,
            "members": member_rows,
        }
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
