# /**
#  * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
#  * and is contributed to the CANN Open Software.
#  *
#  * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
#  * All Rights Reserved.
#  *
#  * Authors (accounts):
#  * - Liang Yanglin <@liang-yanglin>
#  * - Su Tonghua <@sutonghua>
#  *
#  * This program is free software: you can redistribute it and/or modify it.
#  * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#  * You may not use this file except in compliance with the License.
#  * See the LICENSE file at the root of the repository for the full text of the License.
#  *
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
#  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  */

import sys
import numpy as np
import glob
import os
import tensorflow as tf

curr_dir = os.path.dirname(os.path.realpath(__file__))


def get_np_dtype(d_type):
    # 将字符串类型转换为numpy对应的数据类型
    dtype_map = {
        # 浮点类型
        "float": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        # 整数类型
        "uint8": np.uint8,
        "int8": np.int8,
        "int32": np.int32,
        "int64": np.int64,
        "int16": np.int16
    }
    if d_type not in dtype_map:
        raise ValueError(f"Unsupported data type: {d_type}. Supported types: {list(dtype_map.keys())}")
    return dtype_map[d_type]


def compare_data(golden_file_lists, output_file_lists, input_d_type, output_d_type):

    # 比较golden和output文件的数据
    # :param golden_file_lists: golden文件列表
    # :param output_file_lists: output文件列表
    # :param input_d_type: 输入数据类型（对应golden文件的类型，可根据需求调换）
    # :param output_d_type: 输出数据类型（对应output文件的类型，可根据需求调换）
    # :return: 数据是否一致
    # 获取numpy数据类型
    np_input_dtype = get_np_dtype(input_d_type)
    np_output_dtype = get_np_dtype(output_d_type)
    
    data_same = True
    # 校验文件数量是否匹配
    if len(golden_file_lists) != len(output_file_lists):
        print(f"FAILED! Golden file count ({len(golden_file_lists)}) != Output file count ({len(output_file_lists)})")
        return False
    
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_gold = np.fromfile(out, np_output_dtype)
        tmp_out = np.fromfile(gold, np_output_dtype)

        # 统一转换为float32进行比较（避免不同类型直接比较的问题，如int32和float32）
        # 若需要严格按位比较，可注释此部分，直接比较（但不同类型位比较无意义）
        tmp_gold_cast = tmp_gold.astype(np.float32)
        tmp_out_cast = tmp_out.astype(np.float32)

        # 比较数据：rtol=相对误差, atol=绝对误差，可根据需求调整
        # 对于整数类型，建议设置rtol=0, atol=0；对于浮点类型，可设置合理的误差阈值
        if np.issubdtype(np_input_dtype, np.integer) and np.issubdtype(np_output_dtype, np.integer):
            diff_res = (tmp_gold == tmp_out)  # 整数类型直接等值比较
        else:
            # 浮点类型使用近似比较，可根据需求调整rtol和atol
            diff_res = np.isclose(tmp_gold_cast, tmp_out_cast, rtol=1e-5, atol=1e-8, equal_nan=False)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print(f"PASSED! File: {os.path.basename(gold)} vs {os.path.basename(out)}")
        else:
            print(f"FAILED! File: {os.path.basename(gold)} vs {os.path.basename(out)}")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists():
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(input_d_type, output_d_type):
    golden_file_lists, output_file_lists = get_file_lists()
    result = compare_data(golden_file_lists, output_file_lists, input_d_type, output_d_type)
    print("compare result:", result)
    return result

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 {} <input_data_type> <output_data_type>".format(sys.argv[0]))
        print("Example: python3 {} int32 float32".format(sys.argv[0]))
        sys.exit(1)
    # 获取输入和输出数据类型
    input_d_type = sys.argv[1].lower()  # 转为小写，兼容大小写输入
    output_d_type = sys.argv[2].lower()
    try:
        ret = process(input_d_type, output_d_type)
        sys.exit(0 if ret else 1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)