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
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",") if x.strip() != ""]
    return shape_list


def gen_data_and_golden(shape_str, d_type="float16"):
    d_type_dict = {
        "float16": np.float16,
        "float32": np.float32,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    if d_type in ("int16", "int32", "int64"):
        input_x1 = np.random.randint(-100, 100, shape).astype(np_type)
        # 除数不为 0
        input_x2 = np.random.randint(1, 10, shape).astype(np_type)
        golden = np.trunc(
            input_x1.astype(np.float64) / input_x2.astype(np.float64)
        ).astype(np_type)
    else:
        input_x1 = np.random.uniform(-10, 10, shape).astype(np_type)
        input_x2 = np.random.uniform(1, 5, shape).astype(np_type)
        golden = np.trunc(
            input_x1.astype(np.float32) / input_x2.astype(np.float32)
        ).astype(np_type)

    input_x1.astype(np_type).tofile(f"{d_type}_input_truncate_div_x1.bin")
    input_x2.astype(np_type).tofile(f"{d_type}_input_truncate_div_x2.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_truncate_div.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
