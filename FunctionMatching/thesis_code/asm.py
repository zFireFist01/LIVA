"""Module for representing assembly functions and binaries extracted from binary dumps."""

import os
import re
from itertools import groupby
from subprocess import PIPE, run
import r2pipe
import numpy as np
import networkx as nx
from networkx import to_numpy_array


ELF_TYPE: dict[str, str] = {
    "REL":  "ET_REL",
    "EXEC": "ET_EXEC",
    "DYN":  "ET_DYN",
    "CORE": "ET_CORE",
}


class Block:
    """Represents a basic block inside a function's CFG."""

    def __init__(self, address: int, instructions: list[str], raw_bytes: bytes = None):
        self.address = address              # Start address of the block
        self.instructions = instructions    # List of asm instruction strings
        self.raw_bytes = raw_bytes          # Optional raw bytes of the block
        self.embedding: np.ndarray = None   # Mean-pooled embedding of instructions

    def __repr__(self) -> str:
        return f"<Block @ 0x{self.address:x}, {len(self.instructions)} insns>"

    def get_num_instructions(self) -> int:
        return len(self.instructions)

    def compute_embedding(self, embeddings: np.ndarray) -> None:
        """Mean-pool instruction embeddings into a single block embedding."""
        assert len(embeddings) > 0
        assert len(embeddings) == len(self.instructions)
        self.embedding = np.mean(embeddings, axis=0)


class Function:
    """Represents an assembly function extracted from a binary dump."""

    def __init__(
        self,
        name: str,
        address: int,
        blocks: list[Block] = None,
        cfg: nx.DiGraph = None,
    ):
        self.name = name                            # Function name (from symbol table)
        self.address = address                      # Entry-point address
        self.blocks: list[Block] = sorted(
            blocks or [], key=lambda b: b.address
        )
        # CFG: nodes are block addresses, edges are control-flow transitions.
        self.cfg: nx.DiGraph = cfg if cfg is not None else nx.DiGraph()
        self.embedding: np.ndarray = None   # Graph-level function embedding
        self.graph_repr: np.ndarray = None  # Adjacency matrix of cfg

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def instructions(self) -> list[str]:
        """Flat list of all instructions across all basic blocks."""
        return [instr for block in self.blocks for instr in block.instructions]

    def __repr__(self) -> str:
        return (
            f"<Function {self.name} @ 0x{self.address:x}, "
            f"{len(self.blocks)} blocks, {len(self.instructions)} instructions>"
        )

    def __str__(self) -> str:
        return self.name

    def get_num_instructions(self) -> int:
        return sum(b.get_num_instructions() for b in self.blocks)

    def get_num_blocks(self) -> int:
        return len(self.blocks)

    def get_blocks_embeddings(self) -> list[np.ndarray]:
        return [b.embedding for b in self.blocks]

    def compute_embeddings(self, asm_model, graph_model=None) -> None:
        """Compute instruction→block embeddings (PalmTree) and optionally
        the graph-level function embedding (struct2vec GraphNetwork)."""
        flat_instrs = self.instructions          # property: flat list of str
        if not flat_instrs:
            return

        all_embeddings = asm_model.get_embedding(flat_instrs)

        # Slice embeddings back per block
        idx = 0
        for block in self.blocks:
            n = block.get_num_instructions()
            block.compute_embedding(all_embeddings[idx : idx + n])
            idx += n

        # Adjacency matrix with rows/cols ordered by sorted block addresses
        nodelist = [b.address for b in self.blocks]
        self.graph_repr = to_numpy_array(self.cfg, nodelist=nodelist)

        if graph_model is not None:
            import tensorflow as tf
            embeddings_t = tf.constant(self.get_blocks_embeddings(), dtype=tf.float32)
            self.embedding = graph_model(
                self.graph_repr, embeddings_t, training=False
            ).numpy()


class Binary:
    """Represents a binary file with its extracted assembly functions."""

    def __init__(self, name: str, file_path: str, functions: list[Function] = None, blobs: list[list[Function]] = None):
        self.name = name
        self.file_path = file_path
        self.functions = functions or []
        self.blobs = blobs or [self.functions]  # Default: all functions in one blob

    def __repr__(self) -> str:
        return f"<Binary {self.name}, {len(self.functions)} functions>"

    def __str__(self) -> str:
        return self.name

    def get_num_functions(self) -> int:
        return len(self.functions)

    def compute_embeddings(self, asm_model, graph_model=None) -> None:
        """Compute embeddings for all functions in the binary."""
        for function in self.functions:
            function.compute_embeddings(asm_model=asm_model, graph_model=graph_model)


def parse_r2_file(file_path: str, asm_model=None, graph_model=None) -> "Binary":
    """Parse a binary or object file using r2pipe and return a Binary object."""
    r2 = r2pipe.open(file_path, flags=["-2"])

    try:
        r2.cmd("aa")

        info = r2.cmdj("ij") or {}
        bin_type = info.get("bin", {}).get("type", "")
        elf_type = next(
            (v for k, v in ELF_TYPE.items() if k in bin_type.upper()),
            "ET_NONE",
        )

        functions: list[Function] = []

        # ------------------------------------------------------------
        # Case 1: normal binaries / shared libs
        # ------------------------------------------------------------
        funcs_raw = r2.cmdj("aflj") or []

        # Fallback for ET_REL: use symbol table, because aflj is often poor
        if elf_type == "ET_REL" or not funcs_raw:
            readelf = run(
                ["readelf", "--syms", "--wide", file_path],
                stdout=PIPE,
                universal_newlines=True,
            )

            funcs_raw = []
            for line in readelf.stdout.splitlines():
                if "FUNC" not in line:
                    continue

                parts = line.split()
                # Typical shape:
                # Num: Value Size Type Bind Vis Ndx Name
                # e.g. 12: 0000000000000000 42 FUNC GLOBAL DEFAULT 1 myfunc
                try:
                    value_hex = parts[1]
                    size = int(parts[2])
                    typ = parts[3]
                    name = parts[-1]
                except Exception:
                    continue

                if typ != "FUNC":
                    continue
                if size == 0:
                    continue

                try:
                    offset = int(value_hex, 16)
                except ValueError:
                    continue

                funcs_raw.append({
                    "name": name,
                    "offset": offset,
                    "size": size,
                })

        #print(f"[DEBUG] parse_r2_file({os.path.basename(file_path)}): candidate funcs = {len(funcs_raw)}")

        for f in funcs_raw:
            name: str = f.get("name", "")
            addr: int = f.get("offset", 0)
            size: int = f.get("size", 0)

            if size == 0:
                continue
            if name.startswith("sym.imp."):
                continue

            # Force analysis of the function at addr if possible
            try:
                r2.cmd(f"af @ {addr}")
            except Exception:
                pass

            blocks_json = r2.cmdj(f"afbj @ {addr}") or []

            # Fallback: create one pseudo-block from linear disasm
            if not blocks_json:
                ins_json = r2.cmdj(f"pDj {size} @ {addr}") or []

                instructions = []
                raw_bytes_list = []

                for op in ins_json:
                    asm = op.get("disasm", "")
                    if not asm or op.get("type", "") == "invalid":
                        continue

                    asm = re.sub(r",", " ", asm)
                    asm = re.sub(r"  +", " ", asm).strip()
                    instructions.append(asm)

                    hex_bytes = op.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                if instructions:
                    raw_bytes = b"".join(raw_bytes_list) if raw_bytes_list else None
                    block = Block(address=addr, instructions=instructions, raw_bytes=raw_bytes)
                    cfg = nx.DiGraph()
                    cfg.add_node(addr)
                    functions.append(Function(name=name, address=addr, blocks=[block], cfg=cfg))
                continue

            blocks: list[Block] = []
            cfg = nx.DiGraph()

            for bb in blocks_json:
                block_addr = bb.get("addr", bb.get("offset", 0))
                block_size = bb.get("size", 0)

                ins_json = r2.cmdj(f"pDj {block_size} @ {block_addr}") or []
                instructions = []
                raw_bytes_list = []

                for op in ins_json:
                    asm = op.get("disasm", "")
                    if not asm or op.get("type", "") == "invalid":
                        continue

                    asm = re.sub(r",", " ", asm)
                    asm = re.sub(r"  +", " ", asm).strip()
                    instructions.append(asm)

                    hex_bytes = op.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                if not instructions:
                    continue

                raw_bytes = b"".join(raw_bytes_list) if raw_bytes_list else None
                blocks.append(Block(address=block_addr, instructions=instructions, raw_bytes=raw_bytes))
                cfg.add_node(block_addr)

                for edge_key in ("jump", "fail"):
                    target = bb.get(edge_key)
                    if target is not None and target != 0:
                        cfg.add_edge(block_addr, target)

            if not blocks:
                continue

            known = {b.address for b in blocks}
            bad_edges = [(u, v) for u, v in cfg.edges() if v not in known]
            cfg.remove_edges_from(bad_edges)

            functions.append(Function(name=name, address=addr, blocks=blocks, cfg=cfg))

    finally:
        r2.quit()

    blobs: list[list[Function]] | None = None
    if elf_type == "ET_REL":
        readelf = run(
            ["readelf", "--syms", "--wide", file_path],
            stdout=PIPE,
            universal_newlines=True,
        )
        sym_lines = [l for l in readelf.stdout.splitlines() if "FUNC " in l]
        sym_entries = []
        for l in sym_lines:
            parts = l.split()
            try:
                value = int(parts[1], 16)
                name = parts[-1]
                sym_entries.append((value, name))
            except Exception:
                continue

        sym_entries.sort(key=lambda x: x[0])

        lookup = {f.name: f for f in functions}
        blobs = []
        for _, group in groupby(sym_entries, key=lambda x: x[0]):
            blob = [lookup[n] for _, n in group if n in lookup]
            if blob:
                blobs.append(blob)

    b = Binary(
        name=os.path.basename(file_path),
        file_path=file_path,
        functions=functions,
        blobs=blobs,
    )
    b.compute_embeddings(asm_model=asm_model, graph_model=graph_model)
    return b