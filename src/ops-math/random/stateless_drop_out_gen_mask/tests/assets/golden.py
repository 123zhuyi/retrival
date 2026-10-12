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

import struct
from functools import reduce
from typing import List
import numpy as np
from numpy import int16, half


__golden__ = {
    "kernel": {
        "stateless_drop_out_gen_mask": "stateless_drop_out_gen_mask_golden"
    }
}


PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHILOX_W_32 = [0x9E3779B9, 0xBB67AE85]
VAL_1 = 0
VAL_2 = 1
VAL_3 = 2
VAL_4 = 3
MASK_32 = 0xffffffff


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow
            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError("ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                                "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`")
    return bfloat16


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
        return self.philox(counter, key, self.philox4_round, PHILOX_M4_32, self.philox4_bumpkey, PHILOX_W_32, 32, MASK_32, rounds)

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


def _philox(rounds, counter, key, count):
    obj = PhiloxRandom()
    return obj.philox_random(rounds, counter, key, count)


def _curand_uniform(x):
    CURAND_2POW32_INV = np.float32(2 ** (-32))
    return np.float32(x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0))


def _gen_key_and_counter(seed, seed1, offset):
    key = [0] * 2
    counter = [0] * 4
    key[0] = seed & MASK_32
    key[1] = (seed >> 32) & MASK_32
    counter[0] = offset[0] & MASK_32
    counter[1] = (offset[0] >> 32) & MASK_32
    counter[2] = offset[1] & MASK_32
    counter[3] = (offset[1] >> 32) & MASK_32
    return key, counter


def _update_prob_pytorch_val(prob, dtype):
    if dtype in ["bfloat16", "bfloat16_t"]:
        return np.array(prob).astype(_numpy_bfloat16()).astype(np.complex128).astype(_numpy_bfloat16())
    elif dtype in ["half", "float16", "float16_t"]:
        return np.array(prob).astype(np.half).astype(np.complex128).astype(np.half)
    return prob


def _uniform_pt(philox_random, prob, dtype):
    uniform_out = _curand_uniform(philox_random)
    prob = _update_prob_pytorch_val(prob, dtype)
    return uniform_out, prob


def _compare_scalar(rst_lst, prob):
    rst_np = np.array(rst_lst)
    mask = (rst_np < prob)
    rst_np[mask] = 1
    rst_np[~mask] = 0
    return rst_np


def _stateless_drop_out_gen_mask(shape, prob, seed, seed1, offset, dtype):
    rounds = 10
    count = reduce(lambda x, y: x * y, shape)
    (key, counter) = _gen_key_and_counter(seed, seed1, offset)
    philox_random = _philox(rounds, counter, key, count)
    (uniform_out, prob) = _uniform_pt(philox_random, prob, dtype)
    mask_out = _compare_scalar(uniform_out, prob)
    b32_out = mask_out.astype(np.bool_)
    b8_out = np.packbits(b32_out, bitorder='little')
    return b8_out


def stateless_drop_out_gen_mask_golden(shape, prob, seed, seed1, offset=None, **kwargs):
    '''
    Kernel golden for stateless_drop_out_gen_mask.
    All the parameters follow @stateless_drop_out_gen_mask_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    out_type_str = kwargs.get("output_dtypes", ["uint8"])[0]
    out_type = getattr(np, out_type_str)
    prob_type = kwargs.get("input_dtypes", [None, "float"])[1]

    in_shape = shape.tolist() if isinstance(shape, np.ndarray) else list(shape)
    seed_val = int(seed) if isinstance(seed, np.ndarray) else int(seed)
    seed1_val = int(seed1) if isinstance(seed1, np.ndarray) else int(seed1)
    offset_val = offset.tolist() if isinstance(offset, np.ndarray) else list(offset)
    if offset_val is None:
        offset_val = [0, 0]

    res = _stateless_drop_out_gen_mask(in_shape, prob, seed_val, seed1_val, offset_val, prob_type)
    return res.astype(out_type)
