# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import numpy as np


def gen_data_and_golden(shape_str, d_type):
    shape_str = shape_str.strip()
    if shape_str.startswith('(') and shape_str.endswith(')'):
        shape_str = shape_str[1:-1]
    
    if not shape_str:
        shape = [1]
    else:
        shape = [int(x) for x in shape_str.split(',')]
    
    d_type_dict = {
        'float16': np.float16,
        'float32': np.float32,
        'int32': np.int32
    }
    
    if d_type not in d_type_dict:
        print(f"Unsupported dtype: {d_type}")
        return

    np_type = d_type_dict[d_type]
    
    # 生成数据
    if d_type == 'int32':
        x1 = np.random.randint(-100, 100, shape).astype(np_type)
        # 避免 x2 为 0，且包含正负数
        x2 = np.random.randint(1, 100, shape).astype(np_type) * \
             np.random.choice([1, -1], size=shape).astype(np_type)
    else:
        x1 = np.random.uniform(-100, 100, shape).astype(np_type)
        # 避免 x2 过于接近 0，且包含正负数
        x2 = np.random.uniform(0.1, 100, shape).astype(np_type) * \
             np.random.choice([1.0, -1.0], size=shape).astype(np_type)
    
    golden = np.mod(x1, x2).astype(np_type)
    
    x1.tofile(f'{d_type}_input_x1.bin')
    x2.tofile(f'{d_type}_input_x2.bin')
    golden.tofile(f'{d_type}_golden_y.bin')
    print(f"Generated data for {d_type} with shape {shape}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 gen_data.py '(shape)' 'dtype'")
        exit(1)
    gen_data_and_golden(sys.argv[1], sys.argv[2])