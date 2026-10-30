#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import re
import torch

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def exp_segsum_backward(gradout, output):
    T = gradout.size(-1)
    y = gradout * output
    mask = torch.tril(torch.ones(T, T, device=gradout.device, dtype=bool), diagonal=0)
    y = y.masked_fill(~mask, 0)
    y = torch.flip(y, [-2])
    y = torch.cumsum(y, dim = -2)
    y = torch.flip(y, [-2])
    mask = torch.tril(torch.ones(T, T, device=gradout.device, dtype=bool), diagonal=-1)
    y = y.masked_fill(~mask, 0)
    gradin = torch.sum(y, dim = -1, keepdim=False)
    return gradin

def gen_data_and_golden(input_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,

    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)

    size = np.prod(input_shape)
    tmp_input1 = np.random.random(size).reshape(input_shape).astype(np_type)
    tmp_input2 = np.random.random(size).reshape(input_shape).astype(np_type)
    x_tensor = torch.tensor(tmp_input1, dtype=torch.float32)
    y_tensor = torch.tensor(tmp_input2, dtype=torch.float32)
    z_golden = exp_segsum_backward(x_tensor)
    tmp_golden = np.array(z_golden).astype(np_type)

    tmp_input1.astype(np_type).tofile(f"{d_type}_input1_exp_segsum_grad.bin")
    tmp_input2.astype(np_type).tofile(f"{d_type}_input2_exp_segsum_grad.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_exp_segsum_grad.bin")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
