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

"""Golden for BroadcastTo operator (Kernel/GEIR and E2E)."""

__spec__ = {
    "broadcast_to": "BroadcastToKernelSpec",
    "torch.broadcast_to": "TorchBroadcastToSpec",
}

import numpy as np
import torch


def _parse_shape(shape):
    if isinstance(shape, np.ndarray):
        return shape.tolist()
    if isinstance(shape, torch.Tensor):
        return shape.tolist()
    return list(shape)


class BroadcastToImpl:
    def __call__(self, x, shape):
        shape_val = _parse_shape(shape)
        return torch.broadcast_to(x, tuple(shape_val))


class BroadcastToKernelSpec:
    def golden(x, shape, **kwargs):
        dtype = x.dtype
        dtype_str = str(dtype)

        if "bfloat16" in dtype_str or "float16" in dtype_str:
            x = x.astype("float32")
        elif (
            "hifloat8" in dtype_str
            or "float8_e5m2" in dtype_str
            or "float8_e4m3fn" in dtype_str
        ):
            x = x.view(np.int8)
        x_t = torch.from_numpy(x)

        shape_val = _parse_shape(shape)
        result = torch.broadcast_to(x_t, tuple(shape_val)).contiguous()

        result_np = result.numpy()
        if (
            "hifloat8" in dtype_str
            or "float8_e5m2" in dtype_str
            or "float8_e4m3fn" in dtype_str
        ):
            result_np = result_np.view(dtype)
        else:
            result_np = result_np.astype(dtype)
        return [result_np]

    third_party = {"torch": BroadcastToImpl}
    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
        "hifloat8": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "binary_equal"},
    }


class TorchBroadcastToSpec:
    def golden(x, shape, **kwargs):
        shape_val = _parse_shape(shape)
        return [torch.broadcast_to(x, tuple(shape_val)).contiguous()]

    third_party = {"torch": BroadcastToImpl}
    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
        "hifloat8": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "binary_equal"},
    }
