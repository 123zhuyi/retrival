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

__input__ = {"kernel": {"map_index": "map_index_input"}}


def map_index_input(x, data_seq, level_index=None, *, transpose=False, **kwargs):
    """
    Input function for map_index.
    All the parameters (names and order) follow @map_index_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    x_length = x.shape[0]
    x = np.random.uniform(low=-10000, high=10000, size=x_length).astype(np.int32)
    x[x < 0] = -1
    N = int(data_seq.shape[0] / x_length)
    position = np.random.choice(np.arange(0, 4096 + 1), size=N, replace=False)
    index = np.random.randint(0, N)
    data_seq = []
    for i in range(N):
        row = []
        for val in x:
            if val == -1:
                row.append(position[i])
            else:
                row.append(val)
        data_seq.append(row)
    data_seq = np.array(data_seq).astype(np.int32)
    x[x < 0] = position[index]
    data_seq = data_seq.flatten()
    if level_index is not None:
        return [x, data_seq, level_index]
    else:
        return [x, data_seq, None]
