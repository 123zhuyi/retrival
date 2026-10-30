#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(golden_file_lists, output_file_lists, d_type):
    if d_type == "float16":
        np_dtype = np.float16
        rtol, atol = 0.005, 0.005  # fp16 容忍度适当放宽
    elif d_type == "float32":
        np_dtype = np.float32
        rtol, atol = 1e-4, 1e-4
    elif d_type == "bfloat16":
        # 依赖于 ml_dtypes 或者 tensorflow 的 bfloat16
        try:
            from ml_dtypes import bfloat16
            np_dtype = bfloat16
        except ImportError:
            import tensorflow as tf
            np_dtype = tf.bfloat16.as_numpy_dtype
        rtol, atol = 0.005, 0.005
    else:
        raise ValueError("d_type must be float16 or float32 or bfloat16")
    
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, rtol, atol, True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print(f"File {os.path.basename(out)} PASSED!")
        else:
            print(f"File {os.path.basename(out)} FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    # 匹配 atan2 的 golden 和 output 文件
    golden_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*golden*.bin")))
    output_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*output*.bin")))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    if not golden_file_lists or not output_file_lists:
        print("Error: Could not find golden or output bin files.")
        return False
        
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result

if __name__ == '__main__':
    ret = process(sys.argv[1])
    exit(0 if ret else 1)