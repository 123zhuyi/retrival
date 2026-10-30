"""Data type definitions and conversions for NPU kernel benchmarking."""

from __future__ import annotations

from enum import Enum
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DType(str, Enum):
    """Supported data types for tensors."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


@cache
def _get_dtype_str_to_python_dtype() -> dict[str, type]:
    return {
        "float64": float,
        "float32": float,
        "float16": float,
        "bfloat16": float,
        "float8_e4m3fn": float,
        "float8_e5m2": float,
        "float4_e2m1fn_x2": float,
        "int64": int,
        "int32": int,
        "int16": int,
        "int8": int,
        "uint8": int,
        "bool": bool,
    }


def dtype_str_to_python_dtype(dtype_str: str) -> type:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_python_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_dtype_str_to_torch_dtype() -> dict[str, "torch.dtype"]:
    import torch

    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> "torch.dtype":
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_torch_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_integer_dtypes() -> frozenset["torch.dtype"]:
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: "torch.dtype") -> bool:
    """Check if dtype is an integer or boolean type."""
    return dtype in _get_integer_dtypes()
