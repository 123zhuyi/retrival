"""Tests for BuilderRegistry."""

import pytest
from npu_kernelbench.builder.registry import BuilderRegistry, get_registry
from npu_kernelbench.data.definition import Definition, TensorSpec, AxisConst
from npu_kernelbench.data.solution import (
    Solution, BuildSpec, SourceFile, SupportedLanguages, SupportedHardware
)


@pytest.fixture
def simple_definition():
    return Definition(
        name="test_add",
        op_type="elemwise",
        axes={"n": AxisConst(type="const", value=8)},
        inputs={"x": TensorSpec(shape=["n"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["n"], dtype="float32")},
        reference="def run(x):\n    return x + 1\n",
    )


@pytest.fixture
def python_solution():
    source = SourceFile(path="kernel.py", content="def my_kernel(x):\n    return x + 1\n")
    spec = BuildSpec(
        languages=[SupportedLanguages.PYTHON],
        target_hardware=[SupportedHardware.ASCEND_910B],
        entry_point="kernel.py::my_kernel",
        destination_passing_style=False,
    )
    return Solution(name="test_sol", definition="test_add", author="test", spec=spec, sources=[source])


class TestBuilderRegistry:
    def test_build_python_solution(self, simple_definition, python_solution, tmp_path):
        registry = BuilderRegistry()
        result = registry.build(simple_definition, python_solution, work_dir=str(tmp_path))
        # build() is compile-only and returns None. Source staging is done by
        # Runner.pre_build, not by the builder.
        assert result is None

    def test_get_registry_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
