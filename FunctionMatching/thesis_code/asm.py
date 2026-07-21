"""Module for representing assembly functions and binaries extracted from binary dumps."""

import os
import re
import r2pipe
import numpy as np
import networkx as nx


ASM_NORMALIZATION_LEGACY = "legacy"
ASM_NORMALIZATION_V2 = "v2"
ASM_NORMALIZATION_MODES = (
    ASM_NORMALIZATION_LEGACY,
    ASM_NORMALIZATION_V2,
)

_INSTRUCTION_PREFIXES = {
    "bnd",
    "lock",
    "rep",
    "repe",
    "repne",
    "repnz",
    "repz",
}
_MNEMONIC_ALIASES = {
    # radare2 uses movabs for the imm64 encoding; PalmTree's vocabulary and
    # most Intel disassemblers represent the same operation as mov.
    "movabs": "mov",
}
_CONTROL_FLOW_TYPES = {
    "call",
    "ccall",
    "cjmp",
    "jmp",
    "rcall",
    "rjmp",
    "ucall",
    "ujmp",
}
_CONTROL_FLOW_MNEMONICS = {
    "call",
    "callf",
    "jmp",
    "jmpf",
    "loop",
    "loope",
    "loopne",
    "loopnz",
    "loopz",
    "xbegin",
}
_REGISTER_RE = re.compile(
    r"^(?:"
    r"(?:r(?:1[0-5]|[8-9])(?:b|d|w)?)|"
    r"(?:[re]?(?:ax|bx|cx|dx|si|di|bp|sp|ip))|"
    r"(?:[abcd][hl])|"
    r"(?:[sd]il|[bs]pl)|"
    r"(?:[cdefgs]s)|"
    r"(?:[xyz]mm(?:[12]?[0-9]|3[01]))|"
    r"(?:mm[0-7])|(?:st(?:\([0-7]\)|[0-7]))|"
    r"(?:k[0-7])|(?:bnd[0-3])|"
    r"(?:cr(?:[0-9]|1[0-5]))|(?:dr(?:[0-9]|1[0-5]))|"
    r"(?:rflags|eflags|flags|mxcsr|eiz|riz)"
    r")$"
)
_SYMBOL_REFERENCE_RE = re.compile(
    r"(?<![0-9a-z_])"
    r"(?:sym(?:\.imp)?|fcn|loc|obj|str|reloc|section|segment)"
    r"\.[^\s,\[\]+*():]+",
    flags=re.IGNORECASE,
)
_RIP_MEMORY_RE = re.compile(
    r"\[[^\]]*\b(?:rip|eip)\b[^\]]*\]",
    flags=re.IGNORECASE,
)
_HEX_RE = re.compile(r"^0x[0-9a-f]+$")
_DECIMAL_RE = re.compile(r"^[0-9]+$")
_IDENTIFIER_RE = re.compile(r"^[a-z_$?][0-9a-z_$?]*$")
_PUNCTUATION_RE = re.compile(r"([\[\]+\-*:\]])")
_OPERAND_KEYWORDS = {
    "byte",
    "dword",
    "far",
    "fs",
    "gs",
    "near",
    "oword",
    "qword",
    "tword",
    "word",
    "xmmword",
    "ymmword",
    "zmmword",
}


def _legacy_normalize_instruction_text(instruction: str) -> str:
    asm_text = re.sub(r",", " ", instruction)
    return re.sub(r"  +", " ", asm_text).strip()


def _is_control_flow(mnemonic: str, instruction_type: str) -> bool:
    return (
        instruction_type.lower() in _CONTROL_FLOW_TYPES
        or mnemonic in _CONTROL_FLOW_MNEMONICS
        or (mnemonic.startswith("j") and len(mnemonic) > 1)
    )


def _is_indirect_control_operand(operand: str) -> bool:
    stripped = operand.strip().lower()
    if "[" in stripped or "(" in stripped:
        return True

    first = stripped.split(maxsplit=1)[0] if stripped else ""
    first = first.lstrip("%")
    return bool(_REGISTER_RE.fullmatch(first))


def _reference_values(*values: object) -> set[int]:
    return {
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }


def _normalize_operand_token(token: str, references: set[int]) -> str | None:
    if token == "ptr":
        # radare2 normally omits Intel's redundant "ptr" qualifier and the
        # PalmTree vocabulary does not contain it.
        return None
    if token in {"[", "]", "+", "-", "*", ":", "address"}:
        return token

    if _HEX_RE.fullmatch(token):
        value = int(token, 16)
        return "address" if value in references else hex(value)
    if _DECIMAL_RE.fullmatch(token):
        value = int(token, 10)
        return "address" if value in references else hex(value)

    register_or_keyword = token.lstrip("%")
    if (
        _REGISTER_RE.fullmatch(register_or_keyword)
        or register_or_keyword in _OPERAND_KEYWORDS
    ):
        return register_or_keyword

    # Remaining dotted/@ operands are textual relocation or symbol identities.
    # They must not become embedding features in the stripped-ELF setting.
    if "." in token or "@" in token or _IDENTIFIER_RE.fullmatch(token):
        return "address"
    return token


def _normalize_instruction_v2(
    instruction: str,
    *,
    instruction_type: str = "",
    jump: int | None = None,
    ptr: int | None = None,
) -> str:
    """Normalize Intel/radare2 text without using symbol identities."""
    text = instruction.split(";", maxsplit=1)[0].strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""

    words = text.split(maxsplit=2)
    mnemonic_tokens = [words[0]]
    if words[0] in _INSTRUCTION_PREFIXES and len(words) >= 2:
        mnemonic_tokens.append(words[1])
        operand = words[2] if len(words) == 3 else ""
    else:
        operand = text.split(maxsplit=1)[1] if len(words) >= 2 else ""

    mnemonic_tokens[-1] = _MNEMONIC_ALIASES.get(
        mnemonic_tokens[-1],
        mnemonic_tokens[-1],
    )
    mnemonic = mnemonic_tokens[-1]
    if (
        operand
        and _is_control_flow(mnemonic, instruction_type)
        and not _is_indirect_control_operand(operand)
    ):
        operand = "address"
    else:
        # RIP/EIP-relative expressions encode a location, but their concrete
        # displacement and any recovered label are build-specific.
        operand = _RIP_MEMORY_RE.sub("[ address ]", operand)
        operand = _SYMBOL_REFERENCE_RE.sub("address", operand)
        # Default x86 segments add no useful distinction and are absent from
        # the pretrained vocabulary. FS/GS are deliberately retained.
        operand = re.sub(r"\b(?:cs|ds|es|ss)\s*:", "", operand)

    operand = operand.replace(",", " ")
    operand = _PUNCTUATION_RE.sub(r" \1 ", operand)
    references = _reference_values(jump, ptr)
    normalized_operands = []
    for token in operand.split():
        normalized = _normalize_operand_token(token, references)
        if normalized is not None:
            normalized_operands.append(normalized)

    return " ".join((*mnemonic_tokens, *normalized_operands))


def normalize_instruction_text(
    instruction: str,
    mode: str = ASM_NORMALIZATION_V2,
    *,
    instruction_type: str = "",
    jump: int | None = None,
    ptr: int | None = None,
) -> str:
    """Normalize one instruction using the requested versioned strategy."""
    if mode == ASM_NORMALIZATION_LEGACY:
        return _legacy_normalize_instruction_text(instruction)
    if mode == ASM_NORMALIZATION_V2:
        return _normalize_instruction_v2(
            instruction,
            instruction_type=instruction_type,
            jump=jump,
            ptr=ptr,
        )
    raise ValueError(
        f"Unknown assembly normalization '{mode}'; "
        f"expected one of {', '.join(ASM_NORMALIZATION_MODES)}"
    )


def normalize_instruction_record(
    instruction: dict,
    mode: str = ASM_NORMALIZATION_V2,
) -> str:
    """Normalize a radare2 pDj record without consulting symbol tables."""
    return normalize_instruction_text(
        str(instruction.get("disasm") or instruction.get("opcode") or ""),
        mode=mode,
        instruction_type=str(instruction.get("type", "")),
        jump=instruction.get("jump"),
        ptr=instruction.get("ptr"),
    )


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

    def compute_embeddings(self, asm_model) -> None:
        """Compute instruction embeddings and mean-pool them per block."""
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


class CodeUnit:
    """Represents an analyzed ELF binary or compilation unit."""

    TYPE_ELF = "ELF"
    TYPE_CU = "CU"

    def __init__(
        self,
        name: str,
        file_path: str,
        functions: list[Function] = None,
        unit_type: str = TYPE_ELF,
        symbol_fallback_count: int = 0,
        pseudo_block_fallback_count: int = 0,
        rodata_bytes: bytes = b"",
        rodata_strings: list[str] = None,
        rodata_section_count: int = 0,
    ):
        self.name = name
        self.file_path = file_path
        self.unit_type = unit_type
        self.type = unit_type
        self.symbol_fallback_count = symbol_fallback_count
        self.pseudo_block_fallback_count = pseudo_block_fallback_count
        self.rodata_bytes = rodata_bytes
        self.rodata_strings = rodata_strings or []
        self.rodata_section_count = rodata_section_count
        self.functions = sorted(functions or [], key=lambda f: f.address)
        self.call_graph: nx.DiGraph = nx.DiGraph()
        self.resolve_internal_calls()

    def __repr__(self) -> str:
        return (
            f"<CodeUnit {self.name}, type={self.unit_type}, "
            f"{len(self.functions)} functions, "
            f"symbol_fallback_count={self.symbol_fallback_count}, "
            f"pseudo_block_fallback_count={self.pseudo_block_fallback_count}, "
            f"rodata_sections={self.rodata_section_count}, "
            f"rodata_bytes={len(self.rodata_bytes)}, "
            f"rodata_strings={len(self.rodata_strings)}>"
        )

    def __str__(self) -> str:
        return self.name

    def get_num_functions(self) -> int:
        return len(self.functions)

    def get_rodata_size(self) -> int:
        return len(self.rodata_bytes)

    def find_function_containing(self, address: int) -> Function | None:
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

    def resolve_internal_calls(self) -> None:
        """Resolve numeric call targets to functions in this code unit, without symbols."""
        self.call_graph.clear()
        for function in self.functions:
            function.resolved_call_targets.clear()
            self.call_graph.add_node(function.address)

        for caller in self.functions:
            for target in caller.call_targets:
                callee = self.find_function_containing(target)
                if callee is None or callee.address == caller.address:
                    continue
                caller.resolved_call_targets.add(callee.address)
                self.call_graph.add_edge(caller.address, callee.address)

    def compute_embeddings(self, asm_model) -> None:
        """Compute block embeddings for all functions in the code unit."""
        for function in self.functions:
            function.compute_embeddings(asm_model=asm_model)


def extract_relocation_targets(relocations: list[dict]) -> dict[int, int]:
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


def extract_call_target(instruction: dict, relocation_targets: dict[int, int] = None) -> int | None:
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


def extract_ascii_strings(data: bytes, min_length: int = 4) -> list[str]:
    pattern = rb"[\x20-\x7e]{" + str(min_length).encode("ascii") + rb",}"
    return [
        match.group(0).decode("utf-8", errors="replace")
        for match in re.finditer(pattern, data)
    ]


def extract_rodata(r2) -> tuple[bytes, list[str], int]:
    """Extract .rodata* bytes and printable strings through radare2."""
    rodata_sections: list[bytes] = []
    rodata_strings: list[str] = []

    for section in r2.cmdj("iSj") or []:
        name = str(section.get("name", ""))
        if name != ".rodata" and not name.startswith(".rodata."):
            continue

        size = int(section.get("size") or section.get("vsize") or 0)
        address = section.get("vaddr")
        if size <= 0 or not isinstance(address, int):
            continue

        try:
            section_hex = r2.cmd(f"p8 {size} @ {address}") or ""
            section_bytes = bytes.fromhex(section_hex.strip())
        except (TypeError, ValueError):
            continue
        if len(section_bytes) != size:
            continue

        rodata_sections.append(section_bytes)
        rodata_strings.extend(extract_ascii_strings(section_bytes))

    return b"\x00".join(rodata_sections), rodata_strings, len(rodata_sections)


def extract_function_symbols(r2) -> list[dict]:
    """Return function-shaped records from radare2's symbol table."""
    functions = []
    for symbol in r2.cmdj("isj") or []:
        if str(symbol.get("type", "")).upper() not in ("FUNC", "FUNCTION"):
            continue

        size = symbol.get("size", 0)
        address = symbol.get("vaddr")
        if not isinstance(size, int) or size <= 0 or not isinstance(address, int):
            continue

        functions.append({
            "name": symbol.get("realname") or symbol.get("name", ""),
            "offset": address,
            "size": size,
        })
    return functions


def parse_r2_file(
    file_path: str,
    asm_model=None,
    unit_type: str = CodeUnit.TYPE_ELF,
    asm_normalization: str = ASM_NORMALIZATION_V2,
) -> "CodeUnit":
    """Parse a binary or object file using r2pipe and return a CodeUnit object."""
    if asm_normalization not in ASM_NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown assembly normalization '{asm_normalization}'; "
            f"expected one of {', '.join(ASM_NORMALIZATION_MODES)}"
        )
    r2 = r2pipe.open(file_path, flags=["-2"])
    
    try:
        rodata_bytes, rodata_strings, rodata_section_count = extract_rodata(r2)
        r2.cmd("aa")
        relocation_targets = extract_relocation_targets(r2.cmdj("irj") or [])

        functions: list[Function] = []
        symbol_fallback_count = 0
        pseudo_block_fallback_count = 0

        # ------------------------------------------------------------
        # Case 1: normal binaries / shared libs
        # ------------------------------------------------------------
        raw_functions = r2.cmdj("aflj") or []

        # Fallback: use radare2's symbol table when analysis finds no functions.
        if not raw_functions:
            symbol_fallback_count += 1
            raw_functions = extract_function_symbols(r2)

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

                    asm = normalize_instruction_record(
                        instruction_json,
                        mode=asm_normalization,
                    )
                    if not asm:
                        continue
                    instructions.append(asm)

                    hex_bytes = instruction_json.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                    call_target = extract_call_target(instruction_json, relocation_targets)
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

                    asm = normalize_instruction_record(
                        instruction_json,
                        mode=asm_normalization,
                    )
                    if not asm:
                        continue
                    instructions.append(asm)

                    hex_bytes = instruction_json.get("bytes", "")
                    raw_bytes_list.append(bytes.fromhex(hex_bytes) if hex_bytes else b"")

                    call_target = extract_call_target(instruction_json, relocation_targets)
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
        symbol_fallback_count=symbol_fallback_count,
        pseudo_block_fallback_count=pseudo_block_fallback_count,
        functions=functions,
        rodata_bytes=rodata_bytes,
        rodata_strings=rodata_strings,
        rodata_section_count=rodata_section_count,
    )
    if asm_model is not None:
        parsed_code_unit.compute_embeddings(asm_model=asm_model)
    return parsed_code_unit
