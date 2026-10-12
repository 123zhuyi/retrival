#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK golden and deterministic packed-mask inputs for BernoulliMask."""

import numpy as np


__spec__ = {"bernoulli_mask": "BernoulliMaskTestSpec"}


_DTYPE_MAP = {
    "float16": np.float16,
    "fp16": np.float16,
    "float32": np.float32,
    "fp32": np.float32,
    "double": np.float64,
    "float64": np.float64,
    "fp64": np.float64,
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}


def _output_dtype(kwargs):
    output_dtypes = kwargs.get("output_dtypes", ("float32",))
    dtype_name = str(output_dtypes[0]).lower()
    if "bfloat16" in dtype_name or "bf16" in dtype_name:
        try:
            from ml_dtypes import bfloat16
        except ImportError as exc:
            raise RuntimeError(
                "TTK bfloat16 cases require the optional ml-dtypes package"
            ) from exc
        return bfloat16
    try:
        return _DTYPE_MAP[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported BernoulliMask output dtype: {dtype_name}"
        ) from exc


class BernoulliMaskTestSpec:
    """Decode each packed byte from least-significant bit to most-significant bit."""

    @staticmethod
    def customize_inputs(mask, **kwargs):
        del kwargs
        pattern = np.array(
            [0x00, 0x01, 0x02, 0x80, 0xA5, 0x5A, 0x7F, 0xFF],
            dtype=np.uint8,
        )
        values = np.resize(pattern, mask.size).reshape(mask.shape)
        return (values,)

    @staticmethod
    def golden(mask, *, output_shape, **kwargs):
        shape = tuple(int(dim) for dim in output_shape)
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        unpacked = np.unpackbits(
            np.asarray(mask, dtype=np.uint8).reshape(-1),
            bitorder="little",
        )[:elements]
        return [unpacked.reshape(shape).astype(_output_dtype(kwargs), copy=False)]

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
    }
