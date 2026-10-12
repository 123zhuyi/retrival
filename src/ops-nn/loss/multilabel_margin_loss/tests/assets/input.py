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

import numpy as np
import hashlib

__input__ = {"kernel": {"multilabel_margin_loss": "multilabel_margin_loss_input"}}


def multilabel_margin_loss_input(x, target, *, reduction="mean", **kwargs):
    """
    Input generator for MultilabelMarginLoss. Names/order follow multilabel_margin_loss_def.cpp
    inputs (x, target). x is kept as the framework-generated float tensor; target is REPLACED with a
    valid PyTorch-MultiLabelMarginLoss target row: a random-length prefix of a class-index permutation
    (values in [0, C)), the remainder filled with the -1 sentinel. This is the only shape of target the
    op (and torch golden) is defined on; a raw random int32 tensor would be meaningless.

    Deterministic per testcase_name so kernel and golden observe the identical target.
    """
    shape = target.shape
    if len(shape) == 1:
        n_rows, c = 1, int(shape[0])
    else:
        n_rows, c = int(shape[0]), int(shape[1])

    name = kwargs.get("testcase_name", "multilabel")
    seed = (int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32)) if c > 0 else 0
    rng = np.random.default_rng(seed)

    tgt = np.full((n_rows, c), -1, dtype=np.int32)
    for r in range(n_rows):
        if c == 0:
            continue
        k = int(rng.integers(0, c + 1))  # number of valid labels in this row: 0..C
        if k > 0:
            labels = rng.permutation(c)[:k].astype(np.int32)
            tgt[r, :k] = labels

    return [x, tgt.reshape(shape)]
