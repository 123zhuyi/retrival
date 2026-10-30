"""Tests for input tensor generation in generator.py."""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from npu_kernelbench.data.definition import Definition, TensorSpec
from npu_kernelbench.data.workload import (
    CustomInput,
    RandomInput,
    ScalarInput,
    Workload,
)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRandTensor:
    def test_float32(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 3], torch.float32, torch.device("cpu"))
        assert t.shape == (2, 3)
        assert t.dtype == torch.float32

    def test_float16(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 3], torch.float16, torch.device("cpu"))
        assert t.shape == (2, 3)
        assert t.dtype == torch.float16

    def test_bfloat16(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 3], torch.bfloat16, torch.device("cpu"))
        assert t.shape == (2, 3)
        assert t.dtype == torch.bfloat16

    def test_int32(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 3], torch.int32, torch.device("cpu"))
        assert t.shape == (2, 3)
        assert t.dtype == torch.int32

    def test_bool(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 3], torch.bool, torch.device("cpu"))
        assert t.shape == (2, 3)
        assert t.dtype == torch.bool

    def test_int8(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([5], torch.int8, torch.device("cpu"))
        assert t.dtype == torch.int8
        assert t.min().item() >= -128
        assert t.max().item() <= 127

    def test_float8_e4m3fn(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([2, 2], torch.float8_e4m3fn, torch.device("cpu"))
        assert t.shape == (2, 2)
        assert t.dtype == torch.float8_e4m3fn

    def test_float4_e2m1fn_x2(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        # float4_e2m1fn_x2 .to() conversion requires CUDA/NPU; skip on CPU
        try:
            t = _rand_tensor([2, 2], torch.float4_e2m1fn_x2, torch.device("cpu"))
            assert t.shape == (2, 2)
            assert t.dtype == torch.float4_e2m1fn_x2
        except NotImplementedError:
            pytest.skip("float4_e2m1fn_x2 not supported on CPU")

    def test_scalar_shape(self):
        from npu_kernelbench.evaluator.generator import _rand_tensor

        t = _rand_tensor([], torch.float32, torch.device("cpu"))
        assert t.ndim == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGenInputs:
    def _make_def(self, name="test_def"):
        return Definition(
            name=name,
            axes={"N": {"type": "var"}, "D": {"type": "var"}},
            inputs={
                "x": TensorSpec(shape=["N", "D"], dtype="float32"),
                "alpha": TensorSpec(shape=["D"], dtype="float32"),
            },
            outputs={"y": TensorSpec(shape=["N", "D"], dtype="float32")},
            reference="def run(x, alpha):\n    return x + alpha.unsqueeze(0)",
        )

    def _make_wl(self, inputs=None, uuid="wl-1"):
        return Workload(
            axes={"N": 4, "D": 8},
            inputs=inputs or {},
            uuid=uuid,
        )

    def test_returns_list_in_definition_order(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl()
        result = gen_inputs(d, w, "cpu", 42)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (4, 8)  # x
        assert result[1].shape == (8,)  # alpha

    def test_random_input_no_override_uses_definition_dtype(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl(inputs={"x": RandomInput()})
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].dtype == torch.float32

    def test_random_input_dtype_override(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl(inputs={"x": RandomInput(dtype="float16")})
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].dtype == torch.float16

    def test_scalar_input(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl(inputs={"alpha": ScalarInput(value=3.14)})
        result = gen_inputs(d, w, "cpu", 42)
        assert result[1] == 3.14
        assert not isinstance(result[1], torch.Tensor)

    def test_mixed_scalar_and_random(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl(
            inputs={
                "x": RandomInput(dtype="bfloat16"),
                "alpha": ScalarInput(value=2),
            }
        )
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].dtype == torch.bfloat16
        assert result[1] == 2

    def test_reproducible_with_same_seed(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl()
        r1 = gen_inputs(d, w, "cpu", 42)
        r2 = gen_inputs(d, w, "cpu", 42)
        assert torch.allclose(r1[0], r2[0])
        assert torch.allclose(r1[1], r2[1])

    def test_different_seeds_different_output(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()
        w = self._make_wl()
        r1 = gen_inputs(d, w, "cpu", 42)
        r2 = gen_inputs(d, w, "cpu", 999)
        assert not torch.allclose(r1[0], r2[0])

    def test_custom_input_uses_custom_fn(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = Definition(
            name="custom_test",
            axes={"N": {"type": "var"}},
            inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
            reference="def run(x): return x",
            custom_inputs_entrypoint="make_inputs",
        )
        w = Workload(
            axes={"N": 3},
            inputs={"x": CustomInput()},
            uuid="wl-custom",
        )

        def custom_fn(axes, device):
            return {"x": torch.tensor([10.0, 20.0, 30.0], device=device)}

        result = gen_inputs(d, w, "cpu", 42, custom_inputs_fn=custom_fn)
        assert torch.allclose(result[0], torch.tensor([10.0, 20.0, 30.0]))

    def test_custom_input_missing_fn_raises(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = Definition(
            name="custom_test",
            axes={"N": {"type": "var"}},
            inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
            reference="def run(x): return x",
            custom_inputs_entrypoint="make_inputs",
        )
        w = Workload(
            axes={"N": 3},
            inputs={"x": CustomInput()},
            uuid="wl-custom",
        )

        with pytest.raises(RuntimeError, match="custom_inputs_fn"):
            gen_inputs(d, w, "cpu", 42)

    def _make_npu_def(self):
        return Definition(
            name="npu_test",
            axes={},
            inputs={
                "x": TensorSpec(description="NPU-style input"),
                "alpha": TensorSpec(description="NPU-style scalar input"),
            },
            outputs={"y": TensorSpec(description="NPU-style output")},
            reference="def run(x, alpha):\n    return x * alpha",
        )

    def _make_npu_wl(self, uuid="npu-wl-1"):
        return Workload(
            uuid=uuid,
            inputs={
                "x": RandomInput(shape=[4, 8], dtype="float32"),
                "alpha": ScalarInput(value=2.0),
            },
        )

    def test_npu_path_shape_and_dtype_from_workload(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_npu_def()
        w = self._make_npu_wl()
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].shape == (4, 8)
        assert result[0].dtype == torch.float32
        assert result[1] == 2.0

    def test_npu_path_missing_shape_raises(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_npu_def()
        w = Workload(
            uuid="npu-wl-err",
            inputs={"x": RandomInput(dtype="float32"), "alpha": ScalarInput(value=2.0)},
        )
        with pytest.raises(ValueError, match="Cannot determine shape"):
            gen_inputs(d, w, "cpu", 42)

    def test_npu_path_missing_dtype_raises(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_npu_def()
        w = Workload(
            uuid="npu-wl-err2",
            inputs={"x": RandomInput(shape=[4, 8]), "alpha": ScalarInput(value=2.0)},
        )
        with pytest.raises(ValueError, match="Cannot determine dtype"):
            gen_inputs(d, w, "cpu", 42)

    def test_fallback_definition_dtype_when_randominput_no_dtype(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()  # community-style: dtype="float32"
        w = self._make_wl(inputs={"x": RandomInput()})
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].dtype == torch.float32

    def test_npu_dtype_overrides_definition_dtype(self):
        from npu_kernelbench.evaluator.generator import gen_inputs

        d = self._make_def()  # dtype="float32"
        w = self._make_wl(inputs={"x": RandomInput(dtype="bfloat16")})
        result = gen_inputs(d, w, "cpu", 42)
        assert result[0].dtype == torch.bfloat16
