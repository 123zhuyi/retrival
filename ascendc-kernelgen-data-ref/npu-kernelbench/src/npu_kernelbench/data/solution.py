"""Solution model — kernel implementation specification with build and source files."""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString


class BuildMethod(str, Enum):
    """Build method for Ascend C kernel solutions."""

    BISHENG = "bisheng"
    """Direct bisheng compiler invocation."""
    CMAKE = "cmake"
    """CMake-based build using ascendc.cmake macros."""


class SupportedLanguages(str, Enum):
    """Supported programming languages for solution implementations."""

    PYTHON = "python"
    """Pure PyTorch / Python kernel implementations."""
    ASCENDC = "ascendc"
    """Ascend C (CANN) native kernel implementations."""
    TRITON = "triton"
    """Triton-based kernel implementations."""
    TILELANG = "tilelang"
    """TileLang DSL kernel implementations."""
    CATLASS = "catlass"
    """Catlass C++ template library based on Ascend C."""


class SupportedHardware(str, Enum):
    """Supported hardware targets for solution implementations."""

    ASCEND_910B = "ascend910b"
    ASCEND_910_93 = "ascend910_93"
    ASCEND_950 = "ascend950"
    LOCAL = "LOCAL"


class SourceFile(BaseModelWithDocstrings):
    """A single source code file in a solution implementation."""

    path: NonEmptyString
    """Relative path of the file (e.g., 'src/kernel.cpp', 'main.py')."""
    content: NonEmptyString
    """Complete text content of the source file."""

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(f"Absolute path not allowed: {self.path}")
        if ".." in src_path.parts:
            raise ValueError(f"Parent directory traversal not allowed: {self.path}")
        return self


class BuildSpec(BaseModelWithDocstrings):
    """Build specification for a solution implementation."""

    languages: list[SupportedLanguages]
    """Programming languages used in the solution."""
    target_hardware: list[SupportedHardware] = Field(min_length=1)
    """Hardware targets this solution is compatible with."""
    entry_point: NonEmptyString
    """Entry point in format '{file_path}::{function_name}'."""
    dependencies: list[NonEmptyString] = Field(default_factory=list)
    """Optional list of required libraries or packages."""
    destination_passing_style: bool = True
    """Whether solution uses destination-passing style."""
    # Ascend C build options
    build_method: BuildMethod | None = None
    """Build method for AscendC solutions (None defaults to bisheng)."""
    bisheng_cflags: list[str] = Field(default_factory=list)
    """Extra compiler flags passed to bisheng."""
    ld_flags: list[str] = Field(default_factory=list)
    """Extra linker flags."""

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "BuildSpec":
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry point format: {self.entry_point}. "
                f'Expected "<file_path>::<function_name>".'
            )
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete kernel implementation for a given Definition."""

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    _hash_cache: str = PrivateAttr()

    name: NonEmptyString
    """Unique name for this solution."""
    definition: NonEmptyString
    """Name of the Definition this solution implements."""
    author: NonEmptyString
    """Name of the author or agent that created this solution."""
    spec: BuildSpec
    """Technical specifications for building and executing this solution."""
    sources: list[SourceFile] = Field(min_length=1)
    """Source code files for the complete implementation."""
    declared_kernel: Optional[str] = None
    """Optional declared kernel name for anti-hack signature verification."""
    description: Optional[str] = None
    """Optional human-readable description of the solution's approach."""

    @model_validator(mode="after")
    def _validate_entry_in_sources(self) -> "Solution":
        entry_file = self.spec.entry_point.split("::")[0]
        source_paths = {s.path for s in self.sources}
        if entry_file not in source_paths:
            raise ValueError(
                f"Entry source file '{entry_file}' not found in sources"
            )
        # Check for duplicates
        if len(source_paths) != len(self.sources):
            raise ValueError("Duplicate source file paths found")
        return self

    def get_entry_path(self) -> Path:
        return Path(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        return self.spec.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        entry_path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        for s in (
            self.name,
            self.definition,
            *[lang.value for lang in self.spec.languages],
            self.spec.entry_point,
            *self.spec.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())
        return h.hexdigest()

    def hash(self) -> str:
        return self._hash_cache

    def __hash__(self) -> int:
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return self._hash_cache == other._hash_cache
