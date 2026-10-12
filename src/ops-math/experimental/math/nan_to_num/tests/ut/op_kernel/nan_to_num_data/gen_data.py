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
    input_x = np.random.uniform(-10, 0, shape).astype(np_type)
    nan = np.float32(0.0)
    posinf = np.float32(3.4028235e+38)
    neginf = np.float32(-3.4028235e+38)

    golden = np.nan_to_num(input_x, nan=nan, posinf=posinf, neginf=neginf).astype(np_type)

    input_x.astype(np_type).tofile(f"{d_type}_input_t_nan_to_num.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_t_nan_to_num.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
