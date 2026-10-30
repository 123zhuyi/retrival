"""Tests for Workload, InputSpec, and ToleranceSpec models."""

import pytest
from pydantic import ValidationError

from npu_kernelbench.data.workload import (
    Workload,
    ToleranceSpec,
    RandomInput,
    ScalarInput,
    CustomInput,
    SafetensorsInput,
    InputSpec,
)


class TestRandomInput:
    def test_default_type(self):
        ri = RandomInput()
        assert ri.type == "random"

    def test_default_shape_is_none(self):
        ri = RandomInput()
        assert ri.shape is None

    def test_with_concrete_shape(self):
        ri = RandomInput(shape=[1, 128, 1024])
        assert ri.shape == [1, 128, 1024]

    def test_with_shape_and_dtype(self):
        ri = RandomInput(shape=[2, 256, 2048], dtype="float16")
        assert ri.shape == [2, 256, 2048]
        assert ri.dtype == "float16"

    def test_with_range(self):
        ri = RandomInput(shape=[32], dtype="int64", range=[0, 127])
        assert ri.shape == [32]
        assert ri.dtype == "int64"
        assert ri.range == [0, 127]

    def test_default_range_is_none(self):
        ri = RandomInput()
        assert ri.range is None


class TestScalarInput:
    def test_int_scalar(self):
        si = ScalarInput(value=42)
        assert si.type == "scalar"
        assert si.value == 42

    def test_float_scalar(self):
        si = ScalarInput(value=3.14)
        assert si.type == "scalar"
        assert si.value == 3.14

    def test_bool_scalar(self):
        si = ScalarInput(value=True)
        assert si.type == "scalar"
        assert si.value is True

    def test_list_scalar(self):
        si = ScalarInput(value=[4096])
        assert si.type == "scalar"
        assert si.value == [4096]

    def test_str_scalar(self):
        si = ScalarInput(value="none")
        assert si.type == "scalar"
        assert si.value == "none"


class TestCustomInput:
    def test_default_type(self):
        ci = CustomInput()
        assert ci.type == "custom"


class TestSafetensorsInput:
    def test_valid(self):
        si = SafetensorsInput(path="data.safetensors", tensor_key="weight")
        assert si.type == "safetensors"
        assert si.path == "data.safetensors"
        assert si.tensor_key == "weight"


class TestToleranceSpec:
    def test_defaults(self):
        ts = ToleranceSpec()
        assert ts.max_atol == 1e-2
        assert ts.max_rtol == 1e-2
        assert ts.required_matched_ratio == 0.9
        assert ts.max_error_cap is None
        assert ts.allow_negative_inf is False

    def test_custom(self):
        ts = ToleranceSpec(max_atol=1e-5, max_rtol=1e-4, max_error_cap=0.1)
        assert ts.max_atol == 1e-5
        assert ts.max_error_cap == 0.1


class TestWorkload:
    def test_valid_workload(self):
        w = Workload(
            axes={"N": 32, "D": 128},
            inputs={
                "x": RandomInput(),
                "alpha": ScalarInput(value=1.0),
            },
            uuid="wl-001",
        )
        assert w.axes == {"N": 32, "D": 128}
        assert w.uuid == "wl-001"

    def test_custom_and_non_custom_mixed_rejected(self):
        with pytest.raises(ValidationError):
            Workload(
                axes={"N": 32},
                inputs={
                    "x": RandomInput(),
                    "y": CustomInput(),
                },
                uuid="wl-002",
            )

    def test_all_custom_inputs_valid(self):
        w = Workload(
            axes={"N": 32},
            inputs={"x": CustomInput(), "y": CustomInput()},
            uuid="wl-003",
        )
        assert isinstance(w.inputs["x"], CustomInput)

    def test_get_scalar_inputs(self):
        w = Workload(
            axes={"N": 32},
            inputs={
                "x": RandomInput(),
                "mode": ScalarInput(value="tanh"),
            },
            uuid="wl-004",
        )
        scalars = w.get_scalar_inputs()
        assert scalars == {"mode": "tanh"}

    def test_uuid_is_first_field(self):
        w = Workload(
            uuid="wl-first",
            axes={"N": 32},
            inputs={"x": RandomInput()},
        )
        # uuid should be the first field in model_fields
        fields = list(Workload.model_fields.keys())
        assert fields[0] == "uuid"

    def test_axes_can_be_none(self):
        w = Workload(
            uuid="wl-no-axes",
            axes=None,
            inputs={"x": RandomInput(shape=[1, 128], dtype="float16")},
        )
        assert w.axes is None

    def test_npu_style_workload_no_axes(self):
        w = Workload(
            uuid="fa-001",
            inputs={
                "query": RandomInput(shape=[1, 128, 1024], dtype="float16"),
                "key": RandomInput(shape=[1, 128, 1024], dtype="float16"),
            },
        )
        assert w.uuid == "fa-001"
        assert w.axes is None
        assert w.inputs["query"].shape == [1, 128, 1024]
        assert w.inputs["query"].dtype == "float16"
