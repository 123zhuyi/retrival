# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import os


def compare(dtype):
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    if dtype == 'int32':
        np_dtype = np.int32
        
    golden_file = f'{dtype}_golden_y.bin'
    output_file = f'{dtype}_output_y.bin'
    
    if not os.path.exists(golden_file) or not os.path.exists(output_file):
        print("Error: bin files not found in current directory.")
        return False

    golden = np.fromfile(golden_file, np_dtype)
    output = np.fromfile(output_file, np_dtype)
    
    if golden.shape != output.shape:
        print(f"Shape mismatch: golden {golden.shape} vs output {output.shape}")
        return False

    atol = 1e-3 if dtype == 'float16' else 1e-4
    rtol = 1e-3 if dtype == 'float16' else 1e-4
    
    # 比较
    if not np.allclose(golden, output, rtol, atol):
        diff = np.abs(golden - output)
        max_diff = np.max(diff)
        print(f"Mismatch! Max diff: {max_diff}")
        print("First 10 golden:", golden[:10])
        print("First 10 output:", output[:10])
        return False
        
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 compare_data.py 'dtype'")
        exit(1)
    # 返回值 0 表示成功，非 0 表示失败
    sys.exit(0 if compare(sys.argv[1]) else 1)