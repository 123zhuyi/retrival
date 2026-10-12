#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="uint8"):
    d_type_dict = {
        "bool": np.int8,
        "uint8": np.int8,
        "int8": np.int8,
        "int16": np.int16,
        "uint16": np.int16,
        "float16": np.int16,
        "bfloat16": np.int16,
        "float32": np.int32,
        "int32": np.int32,
        "uint64": np.int64,
        "int64": np.int64
        "complex64": np.int64
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    tmp_input = np.random.randint(low=2, high=11, size=size).astype(np_type)
    tmp_input = tmp_input.reshape(shape)
    tmp_golden = tf.matrix_diag(tmp_input).numpy()

    tmp_input.astype(np_type).tofile(f"{d_type}_input_t_is_finite.bin")
    tmp_golden.astype(bool).tofile(f"{d_type}_golden_t_is_finite.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])



