"""Tests for Solution, BuildSpec, and SourceFile models."""

import pytest
from pydantic import ValidationError

from npu_kernelbench.data.solution import (
    Solution,
    BuildSpec,
    SourceFile,
    SupportedLanguages,
    SupportedHardware,
)


class TestSupportedLanguages:
    def test_values(self):
        assert SupportedLanguages.PYTHON == "python"
        assert SupportedLanguages.ASCENDC == "ascendc"
        assert SupportedLanguages.TRITON == "triton"
        assert SupportedLanguages.TILELANG == "tilelang"
        assert SupportedLanguages.CATLASS == "catlass"


class TestSupportedHardware:
    def test_values(self):
        assert SupportedHardware.ASCEND_910B == "ascend910b"


class TestSourceFile:
    def test_valid_source(self):
        sf = SourceFile(path="kernel.py", content="def run(x): return x")
        assert sf.path == "kernel.py"
        assert sf.content == "def run(x): return x"

    def test_absolute_path_rejected(self):
        with pytest.raises(ValidationError):
            SourceFile(path="/etc/passwd", content="x")

    def test_parent_traversal_rejected(self):
        with pytest.raises(ValidationError):
            SourceFile(path="../secret.py", content="x")


class TestBuildSpec:
    def test_default_build(self):
        bs = BuildSpec(
            languages=[SupportedLanguages.TILELANG],
            target_hardware=[SupportedHardware.ASCEND_910B],
            entry_point="main.py::run",
        )
        assert bs.bisheng_cflags == []
        assert bs.ld_flags == []

    def test_bisheng_cflags_and_ld_flags(self):
        bs = BuildSpec(
            languages=[SupportedLanguages.ASCENDC],
            target_hardware=[SupportedHardware.ASCEND_910B],
            entry_point="kernel.cpp::run",
            bisheng_cflags=["-O2", "-ffast-math"],
            ld_flags=["-lcublas"],
        )
        assert bs.bisheng_cflags == ["-O2", "-ffast-math"]
        assert bs.ld_flags == ["-lcublas"]

    def test_entry_point_format_invalid(self):
        with pytest.raises(ValidationError):
            BuildSpec(
                languages=[SupportedLanguages.TILELANG],
                target_hardware=[SupportedHardware.ASCEND_910B],
                entry_point="no_double_colon",
            )


class TestSolution:
    def test_valid_solution(self):
        sol = Solution(
            name="gelu_pytorch_v1",
            definition="gelu",
            author="test",
            spec=BuildSpec(
                languages=[SupportedLanguages.TILELANG],
                target_hardware=[SupportedHardware.ASCEND_910B],
                entry_point="main.py::run",
            ),
            sources=[
                SourceFile(path="main.py", content="def run(x): return x"),
            ],
        )
        assert sol.name == "gelu_pytorch_v1"
        assert sol.get_entry_path().name == "main.py"
        assert sol.get_entry_symbol() == "run"

    def test_hash_deterministic(self):
        sol1 = Solution(
            name="test",
            definition="def1",
            author="a",
            spec=BuildSpec(
                languages=[SupportedLanguages.TILELANG],
                target_hardware=[SupportedHardware.ASCEND_910B],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="x")],
        )
        sol2 = Solution(
            name="test",
            definition="def1",
            author="a",
            spec=BuildSpec(
                languages=[SupportedLanguages.TILELANG],
                target_hardware=[SupportedHardware.ASCEND_910B],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="x")],
        )
        assert sol1.hash() == sol2.hash()
        assert sol1 == sol2

    def test_entry_file_not_in_sources(self):
        with pytest.raises(ValidationError):
            Solution(
                name="test",
                definition="def1",
                author="a",
                spec=BuildSpec(
                    languages=[SupportedLanguages.TILELANG],
                    target_hardware=[SupportedHardware.ASCEND_910B],
                    entry_point="missing.py::run",
                ),
                sources=[SourceFile(path="other.py", content="x")],
            )

    def test_solution_hash_is_sha256_length(self):
        """Solution hash should be 64 hex chars (SHA256)."""
        sol = Solution(
            name="test_hash",
            definition="test_def",
            author="test",
            spec=BuildSpec(
                languages=[SupportedLanguages.TILELANG],
                target_hardware=[SupportedHardware.ASCEND_910B],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        )
        h = sol.hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
