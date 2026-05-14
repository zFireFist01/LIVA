"""Module for representing assembly functions and binaries extracted from binary dumps."""

import os
import re
from subprocess import PIPE, run
import r2pipe
import numpy as np
import networkx as nx
from networkx import to_numpy_array


def normalize_instruction_text(instruction: str) -> str:
    """Normalize instruction text in the same way used by parse_r2_file."""
    asm_text = re.sub(r",", " ", instruction)
    return re.sub(r"  +", " ", asm_text).strip()


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
        size: int = 0,
        blocks: list[Block] = None,
        cfg: nx.DiGraph = None,
        call_targets: set[int] = None,
    ):
        self.name = name                            # Function name (from symbol table)
        self.address = address                      # Entry-point address
        self.size = size                            # Function size in bytes, when known
        self.blocks: list[Block] = sorted(
            blocks or [], key=lambda b: b.address
        )
        # CFG: nodes are block addresses, edges are control-flow transitions.
        self.cfg: nx.DiGraph = cfg if cfg is not None else nx.DiGraph()
        self.call_targets: set[int] = call_targets or set()
        self.resolved_call_targets: set[int] = set()
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

    def get_end_address(self) -> int:
        """Best-effort exclusive end address for interval-based call resolution."""
        if self.size > 0:
            return self.address + self.size
        if not self.blocks:
            return self.address
        return max(
            block.address + (len(block.raw_bytes) if block.raw_bytes else 1)
            for block in self.blocks
        )

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


class CodeUnit:
    """Represents an analyzed ELF binary or compilation unit."""

    TYPE_ELF = "ELF"
    TYPE_CU = "CU"

    def __init__(
        self,
        name: str,
        file_path: str,
        functions: list[Function] = None,
        blobs: list[list[Function]] = None,
        unit_type: str = TYPE_ELF,
        readelf_fallback_count: int = 0,
        pseudo_block_fallback_count: int = 0,
    ):
        self.name = name
        self.file_path = file_path
        self.unit_type = unit_type
        self.type = unit_type
        self.readelf_fallback_count = readelf_fallback_count
        self.pseudo_block_fallback_count = pseudo_block_fallback_count
        self.functions = sorted(functions or [], key=lambda f: f.address)
        self.blobs = blobs or [self.functions]  # Default: all functions in one blob
        self.call_graph: nx.DiGraph = nx.DiGraph()
        self._resolve_internal_calls()

    def __repr__(self) -> str:
        return (
            f"<CodeUnit {self.name}, type={self.unit_type}, "
            f"{len(self.functions)} functions, "
            f"readelf_fallback_count={self.readelf_fallback_count}, "
            f"pseudo_block_fallback_count={self.pseudo_block_fallback_count}>"
        )

    def __str__(self) -> str:
        return self.name

    def get_num_functions(self) -> int:
        return len(self.functions)

    def _find_function_containing(self, address: int) -> Function | None:
        for function in self.functions:
            if function.address == address:
                return function

        candidates = [
            function
            for function in self.functions
            if function.address <= address < function.get_end_address()
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda function: function.get_end_address() - function.address,
        )

    def _resolve_internal_calls(self) -> None:
        """Resolve numeric call targets to functions in this code unit, without symbols."""
        self.call_graph.clear()
        for function in self.functions:
            function.resolved_call_targets.clear()
            self.call_graph.add_node(function.address)

        for caller in self.functions:
            for target in caller.call_targets:
                callee = self._find_function_containing(target)
                if callee is None or callee.address == caller.address:
                    continue
                caller.resolved_call_targets.add(callee.address)
                self.call_graph.add_edge(caller.address, callee.address)

    def compute_embeddings(self, asm_model, graph_model=None) -> None:
        """Compute embeddings for all functions in the code unit."""
        for function in self.functions:
            function.compute_embeddings(asm_model=asm_model, graph_model=graph_model)


def _extract_relocation_targets(relocations: list[dict]) -> dict[int, int]:
    """Map relocation field addresses to numeric symbol values, when available."""
    relocation_targets = {}
    for relocation in relocations:
        relocation_address = relocation.get("vaddr")
        symbol_address = relocation.get("sym_va")
        if (
            isinstance(relocation_address, int)
            and isinstance(symbol_address, int)
            and symbol_address > 0
        ):
            relocation_targets[relocation_address] = symbol_address
    return relocation_targets


def _extract_call_target(instruction: dict, relocation_targets: dict[int, int] = None) -> int | None:
    """Return a numeric direct-call target from radare2 JSON, if present."""
    if not str(instruction.get("type", "")).startswith("call"):
        return None

    relocation_targets = relocation_targets or {}
    instruction_address = instruction.get("offset")
    if not isinstance(instruction_address, int):
        instruction_address = instruction.get("addr")
    if isinstance(instruction_address, int):
        for relocation_address in (instruction_address, instruction_address + 1):
            relocated_target = relocation_targets.get(relocation_address)
            if relocated_target is not None:
                return relocated_target

    if instruction.get("reloc"):
        return None

    for key in ("jump", "ptr", "val"):
        call_target = instruction.get(key)
        if isinstance(call_target, int) and call_target > 0:
            return call_target
        if isinstance(call_target, str):
            try:
                return int(call_target, 16)
            except ValueError:
                continue

    match = re.search(r"\b0x[0-9a-fA-F]+\b", instruction.get("disasm", ""))
    if match:
        return int(match.group(0), 16)

    return None


def parse_r2_file(
    file_path: str,
    asm_model=None,
    graph_model=None,
    unit_type: str = CodeUnit.TYPE_ELF,
) -> "CodeUnit":
    """Parse a binary or object file using r2pipe and return a CodeUnit object."""
    r2 = r2pipe.open(file_path, flags=["-2"])
    
    try:
        r2.cmd("aa")
        relocation_targets = _extract_relocation_targets(r2.cmdj("irj") or [])

        functions: list[Function] = []
        readelf_fallback_count = 0
        pseudo_block_fallback_count = 0

        # ------------------------------------------------------------
        # Case 1: normal binaries / shared libs
        # ------------------------------------------------------------
        raw_functions = r2.cmdj("aflj") or []

        # Fallback: use symbol table when aflj is poor or empty
        if not raw_functions:
            readelf_fallback_count += 1
            readelf = run(
                ["readelf", "--syms", "--wide", file_path],
                stdout=PIPE,
                universal_newlines=True,
            )

            raw_functions = []
            for line in readelf.stdout.splitlines():
                if "FUNC" not in line:
                    continue

                parts = line.split()
                # Typical shape:
                # Num: Value Size Type Bind Vis Ndx Name
                # e.g. 12: 0000000000000000 42 FUNC GLOBAL DEFAULT 1 myfunc
                try:
                    value_hex = parts[1]
                    function_size = int(parts[2])
                    symbol_type = parts[3]
                    function_name = parts[-1]
                except Exception:
                    continue

                if symbol_type != "FUNC":
                    continue
                if function_size == 0:
                    continue

                try:
                    function_address = int(value_hex, 16)
                except ValueError:
                    continue

                raw_functions.append({
                    "name": function_name,
                    "offset": function_address,
                    "size": function_size,
                })

        #print(f"[DEBUG] parse_r2_file({os.path.basename(file_path)}): candidate funcs = {len(raw_functions)}")

        for raw_function in raw_functions:
            function_name: str = raw_function.get("name", "")
            function_address: int = raw_function.get(
                "offset",
                raw_function.get("addr", 0),
            )
            function_size: int = raw_function.get("size", 0)

            if function_size == 0:
                continue
            if function_name.startswith("sym.imp."):
                continue

            # Force analysis of this function entry point if possible.
            try:
                r2.cmd(f"af @ {function_address}")
            except Exception:
                pass

            blocks_json = r2.cmdj(f"afbj @ {function_address}") or []

            # Fallback: create one pseudo-block from linear disasm
            if not blocks_json:
                pseudo_block_fallback_count += 1
                instructions_json = r2.cmdj(f"pDj {function_size} @ {function_address}") or []

                instructions = []
                raw_bytes_list = []
                call_targets = set()

                for instruction_json in instructions_json:
                    asm = instruction_json.get("disasm", "")
                    if not asm or instruction_json.get("type", "") == "invalid":
                        continue

                    asm = normalize_instruction_text(asm)
                    instructions.append(asm)

                    hex_bytes = instruction_json.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                    call_target = _extract_call_target(instruction_json, relocation_targets)
                    if call_target is not None:
                        call_targets.add(call_target)

                if instructions:
                    raw_bytes = b"".join(raw_bytes_list) if raw_bytes_list else None
                    block = Block(address=function_address, instructions=instructions, raw_bytes=raw_bytes)
                    function_cfg = nx.DiGraph()
                    function_cfg.add_node(function_address)
                    functions.append(Function(
                        name=function_name,
                        address=function_address,
                        size=function_size,
                        blocks=[block],
                        cfg=function_cfg,
                        call_targets=call_targets,
                    ))
                continue

            blocks: list[Block] = []
            function_cfg = nx.DiGraph()
            call_targets = set()

            for block_json in blocks_json:
                block_address = block_json.get("addr", block_json.get("offset", 0))
                block_size = block_json.get("size", 0)

                instructions_json = r2.cmdj(f"pDj {block_size} @ {block_address}") or []
                instructions = []
                raw_bytes_list = []

                for instruction_json in instructions_json:
                    asm = instruction_json.get("disasm", "")
                    if not asm or instruction_json.get("type", "") == "invalid":
                        continue

                    asm = normalize_instruction_text(asm)
                    instructions.append(asm)

                    hex_bytes = instruction_json.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                    call_target = _extract_call_target(instruction_json, relocation_targets)
                    if call_target is not None:
                        call_targets.add(call_target)

                if not instructions:
                    continue

                raw_bytes = b"".join(raw_bytes_list) if raw_bytes_list else None
                blocks.append(Block(address=block_address, instructions=instructions, raw_bytes=raw_bytes))
                function_cfg.add_node(block_address)

                for edge_key in ("jump", "fail"):
                    edge_target = block_json.get(edge_key)
                    if edge_target is not None and edge_target != 0:
                        function_cfg.add_edge(block_address, edge_target)

            if not blocks:
                continue

            known_block_addresses = {block.address for block in blocks}
            edges_outside_function = [
                (source_block, target_block)
                for source_block, target_block in function_cfg.edges()
                if target_block not in known_block_addresses
            ]
            function_cfg.remove_edges_from(edges_outside_function)

            functions.append(Function(
                name=function_name,
                address=function_address,
                size=function_size,
                blocks=blocks,
                cfg=function_cfg,
                call_targets=call_targets,
            ))

    finally:
        r2.quit()

    parsed_code_unit = CodeUnit(
        name=os.path.basename(file_path),
        file_path=file_path,
        unit_type=unit_type,
        readelf_fallback_count=readelf_fallback_count,
        pseudo_block_fallback_count=pseudo_block_fallback_count,
        functions=functions,
    )
    if asm_model is not None:
        parsed_code_unit.compute_embeddings(asm_model=asm_model, graph_model=graph_model)
    return parsed_code_unit
