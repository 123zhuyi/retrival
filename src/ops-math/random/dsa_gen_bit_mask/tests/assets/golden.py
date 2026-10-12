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
import torch
from functools import reduce

PHILOX_W32_0 = 0x9E3779B9
PHILOX_W32_1 = 0xBB67AE85
PHILOX_M4x32_0 = 0xD2511F53
PHILOX_M4x32_1 = 0xCD9E8D57
CURAND_2POW32_INV = 2.3283064e-10
CURAND_2POW32_INV_2PI = 2.3283064e-10 * 6.2831855
MAX_THREADS_PER_PROCESSOR = 2048
MAX_BLOCK_NUMS = 2147483647
PHILOX_BLOCK_THREAD = 512
STEP = 4


def mulhilo32(a, b):
    product = np.uint64(a) * b.astype(np.uint64)
    hi = (product >> 32) & 0xFFFFFFFF
    lo = product & 0xFFFFFFFF
    return hi.astype(np.uint32), lo.astype(np.uint32)


def _philox4x32round(ctr, key):
    hi0, lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0])
    hi1, lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2])
    return np.array(
        [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0], dtype=np.uint32
    )


def rand_Philox4x32_10(c, k):
    k = k.copy()
    for i in range(9):
        c = _philox4x32round(c, k)
        k[0] += PHILOX_W32_0
        k[1] += PHILOX_W32_1
        k[0] &= 0xFFFFFFFF
        k[1] &= 0xFFFFFFFF
    return _philox4x32round(c, k)


class RandStatePhilox4_32_10:
    def __init__(self):
        self.ctr = np.array([0, 0, 0, 0], dtype=np.uint32)
        self.key = np.array([0, 0], dtype=np.uint32)
        self.STATE = 0
        self.output = np.array([0, 0, 0, 0], dtype=np.uint32)


def Philox_State_Incr_hi(s, n):
    nlo = n & 0xFFFFFFFF
    nhi = (n >> 32) & 0xFFFFFFFF
    s.ctr[2] += nlo
    if s.ctr[2] < nlo:
        nhi += 1
    s.ctr[3] += nhi
    s.ctr[3] &= 0xFFFFFFFF


def Philox_State_Incr(s, n=None):
    if n is None:
        for i in range(STEP):
            s.ctr[i] += 1
            if s.ctr[i] != 0:
                break
    else:
        nlo = n & 0xFFFFFFFF
        nhi = (n >> 32) & 0xFFFFFFFF
        s.ctr[0] += nlo
        if s.ctr[0] < nlo:
            nhi += 1
        s.ctr[1] += nhi
        if s.ctr[1] < nhi:
            nhi += 1
        s.ctr[2] += nhi
        if s.ctr[2] < nhi:
            nhi += 1
        s.ctr[3] += nhi
        s.ctr &= 0xFFFFFFFF


def skipahead_sequence(n, state):
    Philox_State_Incr_hi(state, n)
    state.output = rand_Philox4x32_10(state.ctr, state.key)


def skipahead(n, state):
    state.STATE += n % 4
    n //= 4
    if state.STATE > 3:
        n += 1
        state.STATE -= 4
    Philox_State_Incr(state, n)
    state.output = rand_Philox4x32_10(state.ctr, state.key)


def rand_init(seed, subsequence, offset, state):
    state.ctr = np.array([0, 0, 0, 0], dtype=np.uint32)
    state.key[0] = seed & 0xFFFFFFFF
    state.key[1] = (seed >> 32) & 0xFFFFFFFF
    state.STATE = 0
    skipahead_sequence(subsequence, state)
    skipahead(offset, state)


def curand(state):
    r = state.output[state.STATE]
    state.STATE += 1
    if state.STATE == 4:
        Philox_State_Incr(state)
        state.output = rand_Philox4x32_10(state.ctr, state.key)
        state.STATE = 0
    return r


def rand4(state):
    tmp = state.output.copy()
    Philox_State_Incr(state)
    state.output = rand_Philox4x32_10(state.ctr, state.key)
    if state.STATE == 0:
        return tmp
    r = np.zeros(STEP, dtype=np.uint32)
    if state.STATE == 1:
        r[0], r[1], r[2], r[3] = tmp[1], tmp[2], tmp[3], state.output[0]
    elif state.STATE == 2:
        r[0], r[1], r[2], r[3] = tmp[2], tmp[3], state.output[0], state.output[1]
    elif state.STATE == 3:
        r[0], r[1], r[2], r[3] = (
            tmp[3],
            state.output[0],
            state.output[1],
            state.output[2],
        )
    return r


def rand_uniform4(state):
    x = rand4(state)
    y = np.array(
        [
            x[0] * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2),
            x[1] * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2),
            x[2] * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2),
            x[3] * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2),
        ],
        dtype=np.float32,
    )
    return y


def bernhoulli_h20(seed, offset, prob, total_ele):
    result = np.zeros(total_ele, dtype=np.uint32)
    minBlockNums = (
        total_ele + MAX_THREADS_PER_PROCESSOR - 1
    ) // MAX_THREADS_PER_PROCESSOR
    minBlockNums = min(minBlockNums, MAX_BLOCK_NUMS)
    total_thread = minBlockNums * PHILOX_BLOCK_THREAD
    repeat_time = ((total_ele + STEP - 1) // STEP + total_thread - 1) // total_thread
    for i in range(total_thread):
        s = RandStatePhilox4_32_10()
        rand_init(seed, i, offset, s)
        for j in range(repeat_time):
            x = rand_uniform4(s)
            realIndex = i * STEP + total_thread * STEP * j
            if realIndex >= total_ele:
                break
            result[realIndex] = 1 if x[0] <= prob[realIndex] else 0
            if realIndex + 1 >= total_ele:
                break
            result[realIndex + 1] = 1 if x[1] <= prob[realIndex + 1] else 0
            if realIndex + 2 >= total_ele:
                break
            result[realIndex + 2] = 1 if x[2] <= prob[realIndex + 2] else 0
            if realIndex + 3 >= total_ele:
                break
            result[realIndex + 3] = 1 if x[3] <= prob[realIndex + 3] else 0
    return result


class _NumpyBFloat16:
    def __new__(cls):
        return np.dtype("float32")


def numpy_bfloat16():
    return np.dtype("float32")


__golden__ = {
    "aclnn": {
        "aclnnBernoulli": "aclnn_bernoulli_golden",
    }
}


def aclnn_bernoulli_golden(self, prob, seed=0, offset=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnBernoulli.
    Parameters follow @aclnnBernoulliGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    if hasattr(prob, "item"):
        prob_val = prob.item()
    else:
        prob_val = prob
    if hasattr(seed, "item"):
        seed = seed.item()
    if hasattr(offset, "item"):
        offset = offset.item()

    seed = np.int64(seed).astype(np.uint64)
    offset = np.int64(offset).astype(np.uint64)

    x_shape = list(self.shape)
    tol = reduce(lambda x, y: x * y, x_shape) if x_shape else 1
    if tol == 0:
        return [torch.empty(x_shape), torch.empty(x_shape)]

    prob_arr = np.array([prob_val])
    out_total_ele = int(reduce(lambda x, y: x * y, x_shape)) if x_shape else 1
    if prob_arr.size == 1:
        prob_arr = np.broadcast_to(prob_arr, out_total_ele)

    result = bernhoulli_h20(
        seed=seed, offset=offset, prob=prob_arr, total_ele=out_total_ele
    )
    return [torch.from_numpy(result).to(out.dtype)]
