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


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",") if x.strip() != ""]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="int16"):
    d_type_dict = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    if d_type == "bool":
        input_x = np.random.randint(0, 2, shape).astype(np.bool_)
        golden = np.invert(input_x).astype(np.int8)  # bool 逻辑非，落盘按 int8(0/1)
        input_x.astype(np.int8).tofile(f"{d_type}_input_bitwise_not.bin")
        golden.tofile(f"{d_type}_golden_bitwise_not.bin")
        return

    if d_type == "uint8":
        input_x = np.random.randint(0, 256, shape).astype(np_type)
    elif d_type == "int8":
        input_x = np.random.randint(-128, 128, shape).astype(np_type)
    else:
        info = np.iinfo(np_type)
        input_x = np.random.randint(-1000, 1000, shape).astype(np_type)

    golden = np.invert(input_x).astype(np_type)  # numpy.invert = ~x（按位补码）
    input_x.tofile(f"{d_type}_input_bitwise_not.bin")
    golden.tofile(f"{d_type}_golden_bitwise_not.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
