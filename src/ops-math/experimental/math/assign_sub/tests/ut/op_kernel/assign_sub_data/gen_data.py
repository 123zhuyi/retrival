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

import os
import glob
import numpy as np
from ml_dtypes import bfloat16


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)

    # 从 JSON 第一个 case 获取参数
    d_type = "int8"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]

    # 生成输入数据
    input_var = np.ones((3, 5, 7, 9)).astype(np_type)
    input_value = np.ones((3, 5, 7, 9)).astype(np_type)

    # 计算 golden 数据: var - value
    if d_type in ("int8", "uint8"):
        result = input_var.astype(np.int16) - input_value.astype(np.int16)
        if d_type == "int8":
            golden = ((result + 128) % 256 - 128).astype(np.int8)
        else:
            golden = (result % 256).astype(np.uint8)
    else:
        golden = (input_var.astype(np.float64) - input_value.astype(np.float64)).astype(
            np_type
        )

    # 保存数据到文件
    input_var.tofile(f"{d_type}_input_assign_sub_var.bin")
    input_value.tofile(f"{d_type}_input_assign_sub_value.bin")
    if golden is not None:
        golden.tofile(f"{d_type}_golden_assign_sub.bin")

    print(f"生成完成: dtype={d_type}")
