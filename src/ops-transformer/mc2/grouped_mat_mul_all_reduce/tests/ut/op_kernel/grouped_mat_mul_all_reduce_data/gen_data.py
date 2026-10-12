# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/usr/bin/python3
import sys
import os
import numpy as np
import re


def parse_str_to_shape_list(shape_str):
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
    return single_shape


def gen_data_and_golden(mListStr: str, kListStr: str, nListStr: str, d_type="float16"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32
    }
    np_type = d_type_dict[d_type]
    mList = parse_str_to_shape_list(mListStr)
    kList = parse_str_to_shape_list(kListStr)
    nList = parse_str_to_shape_list(nListStr)

    for index, m in enumerate(mList):
        tmp_x = (np.random.rand(m, kList[index]) - 0.5) * 1
        tmp_weight = (np.random.rand(kList[index], nList[index]) - 0.5) * 1
        tmp_bias = (np.random.rand(nList[index]) - 0.5) * 1

        tmp_x = tmp_x.astype(np_type)
        tmp_weight = tmp_weight.astype(np_type)
        tmp_bias = tmp_bias.astype(np_type)

        tmp_golden = tmp_x @ tmp_weight + np.expand_dims(tmp_bias, axis=0)

        tmp_x.tofile(f"{d_type}_input_x_{index}.bin")
        tmp_weight.tofile(f"{d_type}_input_weight_{index}.bin")
        tmp_bias.tofile(f"{d_type}_input_bias_{index}.bin")
        tmp_golden.tofile(f"{d_type}_golden_y_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Param num must be 5.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
