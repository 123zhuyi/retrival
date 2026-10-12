#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list


def gen_data_and_golden(input_shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)

    size = np.prod(input_shape)
    np.random.seed(1234)
    inputs = np.random.random(size).reshape(input_shape).astype(np_type)
    if d_type == "float32":
        exp_array = inputs.view(np.uint8).reshape(-1, 4)[:, 3]
        mantissa = inputs.view(np.uint8).reshape(-1, 4)[:, :3].flatten()
    else:
        exp_array = inputs.view(np.uint8).reshape(-1, 2)[:, 1]
        mantissa = inputs.view(np.uint8).reshape(-1, 2)[:, 0]

    inputs.astype(np_type).tofile(f"{d_type}_input.bin")
    hist = np.bincount(exp_array, minlength=256)
    fixed = gen_encode_golden(exp_array, hist)
    hist.astype(np.int32).tofile(f"golden_pdf.bin")
    fixed.astype(np.uint16).tofile(f"fixed.bin")
    inputs.astype(np_type).tofile(f"{d_type}_golden_recover.bin")
    mantissa.tofile(f"mantissa.bin")


def gen_encode_golden(exp_array, hist):
    fixed = np.zeros_like(exp_array, dtype=np.uint8).view(np.uint16)
    ranking = rank_elements_with_index(hist)
    exp_sort_idx = ranking[exp_array]
    exp_bit_num = np.array([max(1, int(x).bit_length()) for x in exp_sort_idx])
    buffer = np.zeros((4096), dtype=np.int32)
    block_bit_num = np.zeros((64), dtype=np.int32)
    meta_info = np.zeros(128, dtype=np.int32)
    meta_info[0] = 12138
    meta_info[1] = 1
    meta_info[2] = exp_array.size // 64
    meta_info[3] = exp_array.size // 64
    meta_info[4] = 0
    
    max_bit = np.max(np.array([exp_bit_num]).reshape(-1, 64), axis=1)
    big_loop = exp_array.size // 4096
    outputs = [meta_info.view(np.uint16)]
    output_acculmulate_size = 0
    for i in range(big_loop):
        max_bit_this_loop = max_bit[i * 64: (i + 1) * 64]
        shr_scale = (2 ** max_bit_this_loop).reshape(64, -1)
        block_bit_num += max_bit_this_loop
        buffer = (buffer.reshape(-1, 64) * shr_scale).flatten()
        buffer += exp_sort_idx[i * 4096: (i + 1) *4096]
        cmp_mask = block_bit_num > 16
        sum_cmp = np.sum(cmp_mask)
        if sum_cmp > 0:
            reduce_mask = np.tile(cmp_mask.reshape(-1, 1), 64).flatten()
            outputs.append((buffer[reduce_mask] & 0xffff).astype(np.uint16))
            buffer[reduce_mask] = buffer[reduce_mask] >> 16
            outputs.append((max_bit_this_loop).astype(np.uint16))
            block_bit_num[cmp_mask] = block_bit_num[cmp_mask]  - 16
            output_acculmulate_size += np.sum(reduce_mask) * 2 + 64 * 2
        else:
            outputs.append((max_bit_this_loop).astype(np.uint16))
            output_acculmulate_size += 64 * 2
    
    outputs.append((buffer & 0xffff).astype(np.uint16))
    output_acculmulate_size += 4096 * 2
    outputs.append(block_bit_num.view(np.uint16))
    output_acculmulate_size += 64 * 4
    meta_info[8] = output_acculmulate_size
    compress = np.concatenate(outputs, axis=0)
    fixed[:compress.size] = compress
    return fixed


def rank_elements_with_index(arr):
    sorted_indices = np.argsort(-arr)
    rank = np.empty_like(sorted_indices)
    
    last_value = None
    last_rank = -1
    for idx, sorted_idx in enumerate(sorted_indices):
        if arr[sorted_idx] != last_value:
            last_rank = idx
        rank[sorted_idx] = last_rank
        last_value = arr[sorted_idx]
    sorted_with_original_indices = [(value, idx) for idx, value in enumerate(arr)]
    sorted_with_original_indices.sort(key=lambda x: (-x[0], x[1]))
    final_rank = np.empty(len(arr), dtype=int)
    for i, (_, original_idx) in enumerate(sorted_with_original_indices):
        final_rank[original_idx] = i
    return final_rank


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3, actually is ", len(sys.argv))
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
