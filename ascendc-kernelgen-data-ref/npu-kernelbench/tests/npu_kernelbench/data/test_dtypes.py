"""Tests for dtype enum and torch conversion."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from npu_kernelbench.data.dtypes import (
    DType,
    dtype_str_to_python_dtype,
    is_dtype_integer,
)


class TestDType:
    def test_float32_value(self):
        assert DType.FLOAT32 == "float32"
        assert DType.FLOAT32.value == "float32"

    def test_all_supported_types(self):
        expected = {
            "float64", "float32", "float16", "bfloat16",
            "float8_e4m3fn", "float8_e5m2",
            "int64", "int32", "int16", "int8", "uint8",
            "bool",
        }
        values = {d.value for d in DType}
        assert expected.issubset(values)

    def test_str_to_python_dtype(self):
        assert dtype_str_to_python_dtype("float32") == float
        assert dtype_str_to_python_dtype("int64") == int
        assert dtype_str_to_python_dtype("int32") == int
        assert dtype_str_to_python_dtype("uint8") == int
        assert dtype_str_to_python_dtype("bool") == bool

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError):
            dtype_str_to_python_dtype("complex64")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchDtypeConversion:
    def test_dtype_str_to_torch_dtype(self):
        from npu_kernelbench.data.dtypes import dtype_str_to_torch_dtype
        import torch

        assert dtype_str_to_torch_dtype("float32") == torch.float32
        assert dtype_str_to_torch_dtype("float16") == torch.float16
        assert dtype_str_to_torch_dtype("bfloat16") == torch.bfloat16
        assert dtype_str_to_torch_dtype("int64") == torch.int64
        assert dtype_str_to_torch_dtype("uint8") == torch.uint8
        assert dtype_str_to_torch_dtype("bool") == torch.bool

    def test_is_dtype_integer(self):
        import torch

        assert is_dtype_integer(torch.int32) is True
        assert is_dtype_integer(torch.int64) is True
        assert is_dtype_integer(torch.uint8) is True
        assert is_dtype_integer(torch.bool) is True
        assert is_dtype_integer(torch.float32) is False
