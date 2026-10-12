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
import numpy as np


def compare(d_type="int16"):
    np_type_dict = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bool": np.int8,  # bool 落盘为 int8(0/1)
    }
    np_type = np_type_dict[d_type]
    golden = np.fromfile(f"{d_type}_golden_bitwise_not.bin", dtype=np_type)
    output = np.fromfile(f"{d_type}_output_bitwise_not.bin", dtype=np_type)
    # 整数 / 逻辑算子：bitwise exact（按位严格相等，atol=0/rtol=0）
    if np.array_equal(golden, output):
        print(f"[bitwise_not] {d_type} compare PASS (bitwise exact)")
    else:
        diff = np.sum(golden != output)
        print(f"[bitwise_not] {d_type} compare FAILED, mismatch={diff}/{golden.size}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Param num must be 2.")
        sys.exit(1)
    compare(sys.argv[1])
