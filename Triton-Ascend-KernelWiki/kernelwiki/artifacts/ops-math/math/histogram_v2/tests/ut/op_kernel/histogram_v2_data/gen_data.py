#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
gen_data.py
"""
import sys
import numpy as np
import torch

def gen_golden_data_simple(shape, bins, min, max):
    bins = int(bins)
    min = float(min)
    max = float(max)
    shape = int(shape)

    if (min == max):
        input_np = np.random.uniform(0, 1024, size = shape).astype(float)
    else:
        input_np = np.random.uniform(min, max, size=shape).astype(float)
    input_x = torch.from_numpy(input_np).reshape(shape)
    input_x_fp32 = input_x.to(torch.float32)
    output_y = torch.histc(input_x_fp32, bins, min, max)
    golden = output_y.numpy()
    input_x_fp32.numpy().tofile("./input_self.bin")
    golden.tofile("./golden.bin")

    min_tensor = torch.tensor([min], dtype=torch.float32)
    max_tensor = torch.tensor([max], dtype=torch.float32)
    min_tensor.numpy().tofile("./min.bin")
    max_tensor.numpy().tofile("./max.bin")
    
if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])