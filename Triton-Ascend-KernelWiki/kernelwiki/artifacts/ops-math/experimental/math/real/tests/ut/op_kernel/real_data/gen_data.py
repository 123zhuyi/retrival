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
import os
import re
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)

    # Real 算子从复数输入中提取实部
    if d_type == "complex32":
        # complex32 = 2 * half = 4 bytes, kernel用int32_t表示
        # 使用结构化数组生成正确的二进制格式
        real_part = np.random.uniform(-10, 10, shape).astype(np.float16)
        imag_part = np.random.uniform(-10, 10, shape).astype(np.float16)
        # 创建结构化数组：2个float16 (real, imag)
        complex32_dtype = np.dtype([('real', np.float16), ('imag', np.float16)])
        input_x = np.empty(shape, dtype=complex32_dtype)
        input_x['real'] = real_part
        input_x['imag'] = imag_part
        # Golden 数据是复数的实部 (float16)
        golden = real_part.astype(np.float16)
        input_x.tofile(f"{d_type}_input_real.bin")
        golden.tofile(f"{d_type}_golden_real.bin")
    elif d_type == "complex64":
        # complex64 = 2 * float32 = 8 bytes, kernel用int64_t表示
        # 使用结构化数组生成正确的二进制格式
        real_part = np.random.uniform(-10, 10, shape).astype(np.float32)
        imag_part = np.random.uniform(-10, 10, shape).astype(np.float32)
        # 创建结构化数组：2个float32 (real, imag)
        complex64_dtype = np.dtype([('real', np.float32), ('imag', np.float32)])
        input_x = np.empty(shape, dtype=complex64_dtype)
        input_x['real'] = real_part
        input_x['imag'] = imag_part
        # Golden 数据是复数的实部 (float32)
        golden = real_part.astype(np.float32)
        input_x.tofile(f"{d_type}_input_real.bin")
        golden.tofile(f"{d_type}_golden_real.bin")
    else:
        # 对于实数类型，Real 算子是恒等变换
        np_type = d_type_dict[d_type]
        input_x = np.random.uniform(-10, 10, shape).astype(np_type)
        golden = input_x.copy()
        input_x.astype(np_type).tofile(f"{d_type}_input_real.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_real.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
