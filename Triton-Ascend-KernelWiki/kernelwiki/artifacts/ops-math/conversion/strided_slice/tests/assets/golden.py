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


__golden__ = {
    "kernel": {
        "strided_slice": "strided_slice_golden"
    }
}


def _compute_indices(start, stop, step):
    if step > 0:
        return list(range(start, stop, step))
    elif step < 0:
        return list(range(start, stop, step))
    else:
        return [start]


def strided_slice_golden(x, begin, end, strides, begin_mask=0, end_mask=0,
                         ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, **kwargs):
    '''
    Kernel golden for strided_slice.
    All the parameters follow @strided_slice_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    '''
    begin = np.asarray(begin).flatten().tolist()
    end = np.asarray(end).flatten().tolist()
    strides = np.asarray(strides).flatten().tolist()

    input_dtype = x.dtype
    if str(input_dtype) == "bfloat16":
        x = x.astype(np.float32)
    elif input_dtype == np.bool_:
        x = x.view(np.uint8)

    ndim = x.ndim
    spec_len = len(begin)

    specs = []
    for i in range(spec_len):
        specs.append({
            'begin': begin[i], 'end': end[i], 'stride': strides[i],
            'bm': bool(begin_mask & (1 << i)),
            'em': bool(end_mask & (1 << i)),
            'ell': bool(ellipsis_mask & (1 << i)),
            'new': bool(new_axis_mask & (1 << i)),
            'shrink': bool(shrink_axis_mask & (1 << i)),
        })

    ell_count = sum(1 for s in specs if s['ell'])
    if ell_count > 0:
        non_ell = spec_len - ell_count
        ell_expand = ndim - non_ell
        new_specs = []
        for s in specs:
            if s['ell']:
                for _ in range(ell_expand):
                    new_specs.append({
                        'begin': 0, 'end': 0, 'stride': 1,
                        'bm': True, 'em': True, 'ell': False, 'new': False, 'shrink': False,
                    })
            else:
                new_specs.append(s)
        specs = new_specs

    slice_indices = []
    squeeze_dims = []
    newaxis_positions = []
    dim = 0

    for i, s in enumerate(specs):
        if s['new']:
            newaxis_positions.append(i)
            continue

        dim_size = x.shape[dim]
        step = s['stride'] if s['stride'] != 0 else 1

        if s['bm']:
            start = 0 if step > 0 else dim_size - 1
        else:
            start = s['begin']
            if start < 0:
                start += dim_size

        if s['em']:
            stop = dim_size if step > 0 else -1
        else:
            stop = s['end']
            if stop < 0:
                stop += dim_size

        start = max(0, min(start, dim_size - 1)) if dim_size > 0 else 0
        if not s['em']:
            stop = max(-1, min(stop, dim_size)) if step > 0 else max(-1, min(stop, dim_size - 1))

        indices = _compute_indices(start, stop, step)
        if len(indices) == 0:
            if step > 0:
                indices = [max(0, min(start, dim_size - 1))] if dim_size > 0 else [0]
            else:
                indices = [max(0, min(start, dim_size - 1))] if dim_size > 0 else [0]

        slice_indices.append(indices)

        if s['shrink']:
            squeeze_dims.append(len(slice_indices) - 1)

        dim += 1

    while dim < ndim:
        slice_indices.append(list(range(x.shape[dim])))
        dim += 1

    result = x
    for d in range(len(slice_indices)):
        idx = slice_indices[d]
        result = np.take(result, idx, axis=d)

    if squeeze_dims:
        result = np.squeeze(result, axis=tuple(squeeze_dims))

    if newaxis_positions:
        shape = list(result.shape)
        offset = 0
        for pos in sorted(newaxis_positions):
            insert_pos = pos + offset
            if insert_pos > len(shape):
                insert_pos = len(shape)
            shape.insert(insert_pos, 1)
            offset += 1
        result = result.reshape(shape)

    if str(input_dtype) == "bfloat16":
        result = result.astype(input_dtype)
    elif input_dtype == np.bool_:
        result = result.view(np.bool_)

    return result
