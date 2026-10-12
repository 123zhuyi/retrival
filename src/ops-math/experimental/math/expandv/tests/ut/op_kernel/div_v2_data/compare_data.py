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
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
def compare_data(golden_file_lists, output_file_lists, d_type):
    """比较golden数据和输出数据"""
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16
    }
    np_dtype = d_type_dict[d_type]
    
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        
        # 使用np.isclose比较（处理浮点数精度）
        # rtol=1e-5, atol=1e-8, equal_nan=True
        diff_res = np.isclose(tmp_out, tmp_gold, rtol=1e-5, atol=1e-8, equal_nan=True)
        diff_idx = np.where(diff_res != True)[0]
        
        if len(diff_idx) == 0:
            print(f"PASSED! {os.path.basename(out)} matches {os.path.basename(gold)}")
        else:
            print(f"FAILED! {os.path.basename(out)} does not match {os.path.basename(gold)}")
            # 打印前5个不匹配的索引和值
            for idx in diff_idx[:5]:
                print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    """获取golden和output文件列表"""
    golden_file_lists = sorted(glob.glob(curr_dir + f"/*{dtype}*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + f"/*{dtype}*output*.bin"))
    return golden_file_lists, output_file_lists

def process(d_type):
    """处理数据比较"""
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    
    if len(golden_file_lists) == 0:
        print(f"No golden files found for dtype {d_type}")
        return False
    
    if len(output_file_lists) == 0:
        print(f"No output files found for dtype {d_type}")
        return False
    
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print(f"Compare result for {d_type}: {'PASSED' if result else 'FAILED'}")
    return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 compare_data.py '<dtype>")
        print("Example: python3 compare_data.py 'float32'")
        exit(1)
    
    ret = process(sys.argv[1])
    exit(0 if ret else 1)