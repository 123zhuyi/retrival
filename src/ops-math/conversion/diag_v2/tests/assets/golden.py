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

__spec__ = {
    "diag_v2": "DiagV2KernelSpec",
    "aclnnDiag": "AclnnDiagSpec",
}

import numpy as np
import torch
import ml_dtypes


def _np_to_torch(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(arr)


def _torch_to_np(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _diag_core(x, diagonal):
    x_np = _torch_to_np(x) if isinstance(x, torch.Tensor) else np.asarray(x)
    x_np = np.ascontiguousarray(x_np)
    return np.diag(x_np, k=int(diagonal))


_BINARY_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "complex64": {"standard": "binary_equal"},
}


class DiagV2KernelSpec:
    def golden(x, diagonal=0, **kwargs):
        y = _diag_core(x, diagonal)
        return [y]

    third_party = {"torch": "torch.diag"}

    tolerance = _BINARY_TOLERANCE


class AclnnDiagSpec:
    def golden(self, diagonal=0, out=None, **kwargs):
        y = _diag_core(self, diagonal)
        return [y]

    third_party = {"torch": "torch.diag"}

    tolerance = _BINARY_TOLERANCE
