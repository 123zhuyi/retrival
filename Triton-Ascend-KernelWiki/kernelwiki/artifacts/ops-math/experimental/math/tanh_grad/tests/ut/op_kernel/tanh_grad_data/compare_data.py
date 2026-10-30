# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
    elif d_type == "float32":
        np_dtype = np.float32
    else:
        raise ValueError("Unsupported d_type")
    
    data_same = True
    for gold_path, out_path in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out_path, np_dtype)
        tmp_gold = np.fromfile(gold_path, np_dtype)

        # 使用 allclose 进行比较，可以更好地处理浮点数精度问题
        # equal_nan=True 使得 NaN == NaN 被认为是相等的
        if np.allclose(tmp_out, tmp_gold, equal_nan=True):
            print(f"PASSED: {os.path.basename(out_path)} matches {os.path.basename(gold_path)}")
        else:
            print(f"FAILED: {os.path.basename(out_path)} does not match {os.path.basename(gold_path)}")
            # 找出不匹配的索引
            diff_idx = np.where(~np.isclose(tmp_out, tmp_gold, equal_nan=True))[0]
            for idx in diff_idx[:5]: # 只打印前5个不匹配项
                print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    # 更新文件名匹配规则
    golden_file_lists = sorted(glob.glob(os.path.join(curr_dir, f"{dtype}_golden_*.bin")))
    output_file_lists = sorted(glob.glob(os.path.join(curr_dir, f"{dtype}_output_*.bin")))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    if not golden_file_lists or not output_file_lists:
        print(f"Error: No data files found for dtype '{d_type}'")
        return False
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("Compare result:", result)
    return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 compare_data.py <dtype>")
        exit(1)
    ret = process(sys.argv[1])
    exit(0 if ret else 1)