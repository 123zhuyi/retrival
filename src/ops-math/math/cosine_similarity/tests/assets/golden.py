#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


def _dtype(x):
    return str(x.dtype).split(".")[-1]


def _validate(input_x1, input_x2, dim, eps):
    if input_x1.shape != input_x2.shape:
        raise ValueError(
            f"input_x1 and input_x2 must have the same shape, got {input_x1.shape} and {input_x2.shape}"
        )
    if not 1 <= input_x1.ndim <= 8 or any(size == 0 for size in input_x1.shape):
        raise ValueError(
            f"inputs must have rank 1-8 without zero dimensions, got {input_x1.shape}"
        )
    if not -input_x1.ndim <= dim < input_x1.ndim:
        raise ValueError(
            f"dim must be in [{-input_x1.ndim}, {input_x1.ndim - 1}], got {dim}"
        )
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if _dtype(input_x1) != _dtype(input_x2) or _dtype(input_x1) not in (
        "float32",
        "float64",
    ):
        raise TypeError(
            "CosineSimilarity expects float32 inputs or their promoted float64 golden inputs"
        )


class CosineSimilaritySpec:
    def golden(input_x1, input_x2, *, dim=1, eps=1e-8, **kwargs):
        import torch

        _validate(input_x1, input_x2, dim, eps)
        x1 = torch.from_numpy(np.ascontiguousarray(input_x1))
        x2 = torch.from_numpy(np.ascontiguousarray(input_x2))
        return torch.nn.functional.cosine_similarity(x1, x2, dim=dim, eps=eps).numpy()

    tolerance = {"float32": {"standard": "stat_rel_err"}}


__spec__ = {"cosine_similarity": "CosineSimilaritySpec"}
