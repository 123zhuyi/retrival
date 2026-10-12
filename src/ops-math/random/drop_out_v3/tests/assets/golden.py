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
from functools import reduce
from typing import List

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = np.float32


__golden__ = {
    "kernel": {
        "drop_out_v3": "drop_out_v3_golden"
    }
}


PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHILOX_W_32 = [0x9E3779B9, 0xBB67AE85]
VAL_1 = 0
VAL_2 = 1
VAL_3 = 2
VAL_4 = 3
MASK_32 = 0xffffffff


class PhiloxRandom(object):
    @staticmethod
    def philox4_round(counter, key, philox_m, len_w, mask_w):
        prod = philox_m[VAL_1] * counter[VAL_1]
        hi_1 = prod >> len_w
        lo_1 = prod & mask_w
        prod = philox_m[VAL_2] * counter[VAL_3]
        hi_2 = prod >> len_w
        lo_2 = prod & mask_w
        counter[VAL_1] = hi_2 ^ counter[VAL_2] ^ key[VAL_1]
        counter[VAL_2] = lo_2
        counter[VAL_3] = hi_1 ^ counter[VAL_4] ^ key[VAL_2]
        counter[VAL_4] = lo_1

    @staticmethod
    def philox4_bumpkey(key, philox_w, mask_w):
        key[VAL_1] = (key[VAL_1] + philox_w[VAL_1]) & mask_w
        key[VAL_2] = (key[VAL_2] + philox_w[VAL_2]) & mask_w

    def philox(self, counter, key, philox_round, philox_m, philox_bumpkey, philox_w, len_w, mask_w, rounds):
        for i in range(rounds - 1):
            philox_round(counter, key, philox_m, len_w, mask_w)
            philox_bumpkey(key, philox_w, mask_w)
        philox_round(counter, key, philox_m, len_w, mask_w)
        return counter

    def philox4_32(self, counter, key, rounds):
        return self.philox(counter, key, self.philox4_round, PHILOX_M4_32, self.philox4_bumpkey, PHILOX_W_32, 32,
                           MASK_32, rounds)

    def inc_counter(self, counter):
        for i in range(4):
            counter[i] = (counter[i] + 1) & MASK_32
            if counter[i] != 0:
                return counter
        return counter

    def philox_random(self, rounds, counter, key, count):
        ret = list()
        for i in range((count + 255) // 256 * 256 // 4):
            ret.extend(self.philox4_32(counter[:], key[:], rounds))
            counter = self.inc_counter(counter[:])
        return np.array(ret)[:count]


def gen_key_and_counter(threadIdx: int, seed: int, offset: int) -> (List, List):
    key_ = [0] * 2
    counter_ = [0] * 4
    key_[0] = seed & MASK_32
    key_[1] = (seed >> 32) & MASK_32
    counter_[2] = threadIdx & MASK_32
    counter_[3] = (threadIdx >> 32) & MASK_32
    counter_[0] = offset & MASK_32
    counter_[1] = (offset >> 32) & MASK_32
    return key_, counter_


def philox(rounds: int, counter: List, key: List, count: int) -> List:
    obj = PhiloxRandom()
    return obj.philox_random(rounds, counter, key, count)


def update_prob_type(prob, dtype):
    if dtype in ["bfloat16", "bfloat16_t"]:
        return np.array(prob).astype(_bf16).astype(np.cfloat).astype(_bf16).item()
    elif dtype in ["half", "float16", "float16_t"]:
        return np.array(prob).astype(np.half).astype(np.cfloat).astype(np.half).item()
    return prob


def compare_scalar(rst_lst: List, prob):
    rst_np = np.array(rst_lst)
    prob_np = np.array(prob)
    mask = (rst_np <= prob_np)
    rst_np[mask] = 1
    rst_np[~mask] = 0
    return rst_np


def binary_array_to_uint8(binary_array):
    binary_array = np.array(binary_array, dtype=np.uint8)
    padding = 8 - (len(binary_array) % 8)
    if padding != 8:
        binary_array = np.pad(binary_array, (0, padding), mode='constant', constant_values=0)
    uint8_values = np.packbits(binary_array, bitorder='little')
    return uint8_values


def curand_uniform(x: List[int]) -> List[float]:
    CURAND_2POW32_INV = 2 ** (-32)
    return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0)


def uniform_pt(philox_random, prob, dtype):
    uniform_out = curand_uniform(philox_random)
    prob = update_prob_type(prob, dtype)
    return uniform_out, prob


def GetVectorSize(eleCount, T_size):
    vecSize = 8
    if eleCount % 2 != 0:
        return 1
    optimalVecSize = 16 // T_size
    vecSize = min(vecSize, optimalVecSize)
    while vecSize > 1:
        canVectorize = ((eleCount % vecSize) == 0)
        if not canVectorize:
            vecSize = vecSize // 2
        else:
            break
    return vecSize


def drop_out_v3_compute(x_in, prob, seed, offset, rounds=10):
    count = reduce(lambda x, y: x * y, x_in.shape)
    if prob == 1.0:
        y_out = np.zeros(x_in.shape, dtype=x_in.dtype)
        binary_array = np.zeros((count + 7) // 8, dtype=np.uint8)
        return y_out, binary_array
    blockSize = 256
    maxThreadsPerMultiProcessor = 2048
    blocksPerSM = maxThreadsPerMultiProcessor // blockSize
    multiProcessorCount = 78
    grid = (count + blockSize - 1) // blockSize
    grid = min(multiProcessorCount * blocksPerSM, grid)
    totalThreads = grid * blockSize

    T_size = 4 if x_in.dtype == np.float32 else 2
    vecSize = GetVectorSize(count, T_size)

    prob = 1.0 - prob
    mask_out = np.zeros(count, dtype=bool)
    y_out = x_in.flatten().copy()

    if vecSize == 1:
        for idx in range(0, count, vecSize):
            threadIdx = idx % totalThreads
            repeatCount = idx // totalThreads
            (key, counter) = gen_key_and_counter(threadIdx, seed, offset // 4 + repeatCount // 4)
            philox_random = philox(rounds, counter, key, 4)
            (uniform_out, prob) = uniform_pt(philox_random, prob, "float")
            mask = compare_scalar(uniform_out, prob)
            mask_out[idx] = mask[repeatCount % 4]
    else:
        fixOffset = vecSize
        if vecSize == 2:
            fixOffset = 4
        for idx in range(0, count, vecSize):
            vecIndx = idx // vecSize
            threadIdx = vecIndx % totalThreads
            repeatCount = vecIndx // totalThreads
            (key, counter) = gen_key_and_counter(threadIdx, seed, offset // 4 + repeatCount * fixOffset // 4)
            philox_random = philox(rounds, counter, key, vecSize)
            (uniform_out, prob) = uniform_pt(philox_random, prob, "float")
            mask = compare_scalar(uniform_out, prob)
            mask_out[idx:idx + vecSize] = mask

    mask_bool = mask_out[:count]
    y_out[mask_bool] = y_out[mask_bool] * (1 / prob)
    y_out[~mask_bool] = y_out[~mask_bool] * 0
    mask_uint8 = binary_array_to_uint8(mask_out)
    return y_out.reshape(x_in.shape), mask_uint8


def drop_out_v3_golden(x, noise_shape=None, p=None, seed=None, offset=None, **kwargs):
    '''
    Kernel golden for drop_out_v3.
    All the parameters follow @drop_out_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    x_type = x.dtype
    p_val = float(np.array(p).flatten()[0])
    seed_val = int(np.array(seed).flatten()[0])
    offset_val = int(np.array(offset).flatten()[1])
    offset_val = int(np.uint64(offset_val))
    dst, mask = drop_out_v3_compute(x, p_val, seed_val, offset_val)
    return [dst.astype(x_type, copy=False), mask.astype(np.uint8)]
