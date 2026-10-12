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
    if d_type == "uint8":
        np_dtype = np.uint8
    else:
        raise ValueError("d_type must be uint8")
    data_same = True
    
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np.uint8)
        tmp_gold = np.fromfile(gold, np.uint8)
        
        # 检查形状是否一致
        if tmp_out.shape != tmp_gold.shape:
            print(f"FAILED! 形状不匹配: output形状{tmp_out.shape}, golden形状{tmp_gold.shape}")
            data_same = False
            continue
        
        # 方法1：直接比较（推荐，uint8应该完全相等）
        if np.array_equal(tmp_out, tmp_gold):
            print(f"PASSED! 文件: {out}")
        else:
            print(f"FAILED! 文件: {out}")
            # 找出不相同的索引
            diff_idx = np.where(tmp_out != tmp_gold)[0]
            print(f"    不相同的数据数量: {len(diff_idx)}")
            
            # 显示前几个不同的值
            for i, idx in enumerate(diff_idx[:10]):
                print(f"    索引 {idx}: output={tmp_out[idx]}(0x{tmp_out[idx]:02x}), "
                      f"golden={tmp_gold[idx]}(0x{tmp_gold[idx]:02x}), "
                      f"差异={int(tmp_out[idx]) - int(tmp_gold[idx])}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result

if __name__ == '__main__':
    ret = process(sys.argv[1])
    exit(0 if ret else 1)