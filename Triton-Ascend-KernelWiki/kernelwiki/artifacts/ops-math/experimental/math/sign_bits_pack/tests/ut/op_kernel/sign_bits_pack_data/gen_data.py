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
import numpy as np
import re

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float"):
    d_type_dict = {
        "float16": np.float16,
        "float": np.float32
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    input_x = np.random.uniform(-10, 10, shape).astype(np_type)

    signs = (input_x >= 0).astype(np.uint8)
    n_elements = len(input_x)
    # 计算每个压缩单元包含多少元素
    n_packs = (n_elements + 7) // 8  # 每个uint8包含8个符号位

    # 位压缩
    packed = []
    for i in range(n_packs):
        packed_val = 0
        start_idx = i * 8
        end_idx = min((i + 1) * 8, n_elements)
        
        for j in range(start_idx, end_idx):
            bit_pos = j - start_idx
            if signs[j]:
                packed_val |= (1 << bit_pos)
        packed.append(packed_val)
    # 填充到total_packs长度    
    # golden = np.sign_bits_pack(input_x).astype(np_type)
    golden = np.array(packed, dtype=np.uint8)
    print(golden.shape)
    input_x.astype(np_type).tofile(f"{d_type}_input_t_sign_bits_pack.bin")
    golden.astype(np.uint8).tofile(f"uint8_golden_t_sign_bits_pack.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
