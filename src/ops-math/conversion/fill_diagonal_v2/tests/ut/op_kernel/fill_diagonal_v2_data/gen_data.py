# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

#!/usr/bin/python3
"""
gen_data.py
"""
import sys
import numpy as np
import torch

def gen_golden_data_simple(shape, fill_value, wrap, dtype):
    fill_value = float(fill_value)
    wrap = bool(wrap)
    shape = eval(shape)
    type_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "int16": np.int16,
        "uint8": np.uint8,
        "bool" : np.bool8,
        "int64": np.int64
    }

    x = np.random.randn(*shape).astype(type_map[dtype])
    golden = x.copy()
    np.fill_diagonal(golden, fill_value, wrap)
    x.tofile("./input_self.bin")
    golden.tofile("./golden.bin")
    valArray = np.array([fill_value], dtype=type_map[dtype])
    valArray.tofile("./input_fill_value.bin")
    
if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])  # shape, fill_value, wrap, dtype
