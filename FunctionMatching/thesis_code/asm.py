"""Module for representing assembly functions and binaries extracted from binary dumps."""

import typing


class Function:
    """Represents an assembly function extracted from a binary dump."""

    def __init__(self, name: str, address: int, instructions: list[str], raw_bytes: bytes = None):
        self.name = name                    # Function name (from symbol table)
        self.address = address              # Start address of the function
        self.instructions = instructions    # List of assembly instruction strings
        self.raw_bytes = raw_bytes          # Optional raw bytes

    def __repr__(self) -> str:
        return f"<Function {self.name} @ 0x{self.address:x}, {len(self.instructions)} instructions>"

    def __str__(self) -> str:
        return self.name

    def get_num_instructions(self) -> int:
        return len(self.instructions)


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