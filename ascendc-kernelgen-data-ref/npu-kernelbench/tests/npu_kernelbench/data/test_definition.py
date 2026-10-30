"""Tests for Definition model."""

import pytest
from pydantic import ValidationError

from npu_kernelbench.data.definition import Definition, TensorSpec, AxisSpec
from npu_kernelbench.data.shapes import AxisConst, AxisVar
from npu_kernelbench.data.dtypes import DType


VALID_DEFINITION = {
    "name": "gelu",
    "axes": {
        "N": {"type": "var", "description": "batch size"},
        "D": {"type": "var", "description": "hidden dim"},
    },
    "inputs": {
        "x": {"shape": ["N", "D"], "dtype": "float32", "description": "input tensor"},
    },
    "outputs": {
        "y": {"shape": ["N", "D"], "dtype": "float32", "description": "output tensor"},
    },
    "reference": (
        "import torch\n"
        "def run(x):\n"
        "    return torch.nn.functional.gelu(x, approximate='none')\n"
    ),
}


class TestTensorSpec:
    def test_valid_tensor_spec(self):
        ts = TensorSpec(shape=["N", "D"], dtype=DType.FLOAT32)
        assert ts.shape == ["N", "D"]
        assert ts.dtype == DType.FLOAT32

    def test_scalar_tensor(self):
        ts = TensorSpec(shape=None, dtype=DType.FLOAT32)
        assert ts.shape is None

    def test_dtype_is_optional_defaults_none(self):
        ts = TensorSpec()
        assert ts.dtype is None
        assert ts.shape is None

    def test_npu_style_no_shape_no_dtype(self):
        ts = TensorSpec(description="supports float16, bfloat16, float32")
        assert ts.shape is None
        assert ts.dtype is None
        assert ts.description == "supports float16, bfloat16, float32"

    def test_community_style_still_works(self):
        ts = TensorSpec(shape=["N", "D"], dtype="float32")
        assert ts.shape == ["N", "D"]
        assert ts.dtype == DType.FLOAT32


class TestDefinition:
    def test_valid_definition(self):
        d = Definition(**VALID_DEFINITION)
        assert d.name == "gelu"
        assert len(d.inputs) == 1
        assert len(d.outputs) == 1
        assert len(d.axes) == 2

    def test_reference_must_have_run_function(self):
        bad = {**VALID_DEFINITION, "reference": "x = 1"}
        with pytest.raises(ValidationError, match="run"):
            Definition(**bad)

    def test_reference_must_be_valid_python(self):
        bad = {**VALID_DEFINITION, "reference": "def run(x): return x @!@"}
        with pytest.raises(ValidationError):
            Definition(**bad)

    def test_input_output_names_no_overlap(self):
        bad = {
            **VALID_DEFINITION,
            "outputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        }
        with pytest.raises(ValidationError):
            Definition(**bad)

    def test_axis_reference_validation(self):
        bad = {
            **VALID_DEFINITION,
            "inputs": {"x": {"shape": ["UNKNOWN"], "dtype": "float32"}},
        }
        with pytest.raises(ValidationError):
            Definition(**bad)

    def test_const_axes_property(self):
        d = Definition(
            name="test",
            axes={"K": AxisConst(value=128)},
            inputs={"x": {"shape": ["K"], "dtype": "float32"}},
            outputs={"y": {"shape": ["K"], "dtype": "float32"}},
            reference="def run(x):\n    return x",
        )
        assert d.const_axes == {"K": 128}

    def test_var_axes_property(self):
        d = Definition(
            name="test",
            axes={
                "N": AxisVar(description="batch"),
                "D": AxisConst(value=64),
            },
            inputs={
                "x": {"shape": ["N", "D"], "dtype": "float32"},
            },
            outputs={
                "y": {"shape": ["N", "D"], "dtype": "float32"},
            },
            reference="def run(x):\n    return x",
        )
        assert d.var_axes == ["N"]

    def test_get_axes_values(self):
        d = Definition(
            name="test",
            axes={
                "N": AxisVar(description="batch"),
                "D": AxisConst(value=64),
            },
            inputs={
                "x": {"shape": ["N", "D"], "dtype": "float32"},
            },
            outputs={
                "y": {"shape": ["N", "D"], "dtype": "float32"},
            },
            reference="def run(x):\n    return x",
        )
        shapes = [(32, 64)]
        axes_values = d.get_axes_values(shapes)
        assert axes_values == {"N": 32}

    def test_custom_inputs_entrypoint(self):
        ref = (
            "def run(x):\n    return x\n\n"
            "def generate_inputs(axes, device):\n"
            "    import torch\n"
            "    return {'x': torch.randn(axes['N'], axes['D'], device=device)}\n"
        )
        d = Definition(
            name="test",
            axes={
                "N": AxisVar(description="batch"),
                "D": AxisConst(value=64),
            },
            inputs={
                "x": {"shape": ["N", "D"], "dtype": "float32"},
            },
            outputs={
                "y": {"shape": ["N", "D"], "dtype": "float32"},
            },
            reference=ref,
            custom_inputs_entrypoint="generate_inputs",
        )
        assert d.custom_inputs_entrypoint == "generate_inputs"

    def test_get_input_shapes(self):
        d = Definition(
            name="test",
            axes={
                "N": AxisVar(description="batch"),
                "D": AxisConst(value=64),
            },
            inputs={
                "x": {"shape": ["N", "D"], "dtype": "float32"},
            },
            outputs={
                "y": {"shape": ["N", "D"], "dtype": "float32"},
            },
            reference="def run(x):\n    return x",
        )
        shapes = d.get_input_shapes({"N": 32})
        assert shapes == {"x": (32, 64)}

    def test_npu_style_definition_no_shape_no_dtype(self):
        d = Definition(
            name="FusionAttention",
            axes={},
            inputs={
                "query": TensorSpec(description="shape (B,S,N*D); supports float16/bfloat16/float32"),
                "key": TensorSpec(description="same as query"),
            },
            outputs={
                "output": TensorSpec(description="same as query"),
            },
            reference="def run(query, key):\n    return query + key",
        )
        assert d.name == "FusionAttention"
        assert d.axes == {}
        assert d.const_axes == {}
        assert d.var_axes == []
        assert d.inputs["query"].shape is None
        assert d.inputs["query"].dtype is None

    def test_axis_validation_skipped_when_shape_is_none(self):
        d = Definition(
            name="test",
            axes={},
            inputs={
                "x": TensorSpec(description="no shape, no dtype"),
            },
            outputs={
                "y": TensorSpec(description="no shape, no dtype"),
            },
            reference="def run(x):\n    return x",
        )
        assert d.inputs["x"].shape is None

    def test_get_input_shapes_returns_none_for_npu_style(self):
        d = Definition(
            name="test",
            axes={},
            inputs={
                "x": TensorSpec(description="NPU-style"),
            },
            outputs={
                "y": TensorSpec(description="NPU-style"),
            },
            reference="def run(x):\n    return x",
        )
        shapes = d.get_input_shapes()
        assert shapes == {"x": None}

    def test_get_output_shapes(self):
        d = Definition(
            name="test",
            axes={
                "N": AxisVar(description="batch"),
                "D": AxisConst(value=64),
            },
            inputs={
                "x": {"shape": ["N", "D"], "dtype": "float32"},
            },
            outputs={
                "y": {"shape": ["N", "D"], "dtype": "float32"},
            },
            reference="def run(x):\n    return x",
        )
        shapes = d.get_output_shapes({"N": 32})
        assert shapes == {"y": (32, 64)}
