#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np


def compare_data(golden_file, output_file, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    np_type = d_type_dict.get(d_type, np.float32)
    golden = np.fromfile(golden_file, dtype=np_type)
    output = np.fromfile(output_file, dtype=np_type)

    if golden.shape != output.shape:
        print(f"FAIL: shape mismatch golden={golden.shape} output={output.shape}")
        return False

    if np.issubdtype(np_type, np.integer):
        match = np.array_equal(golden, output)
    else:
        match = np.allclose(golden, output, atol=1e-4, rtol=1e-4)

    max_diff = np.max(np.abs(golden.astype(np.float64) - output.astype(np.float64)))
    print(f"{'PASS' if match else 'FAIL'}: max_diff={max_diff}")
    return match


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: compare_data.py <golden_file> <output_file> <dtype>")
        exit(1)
    ok = compare_data(sys.argv[1], sys.argv[2], sys.argv[3])
    exit(0 if ok else 1)
