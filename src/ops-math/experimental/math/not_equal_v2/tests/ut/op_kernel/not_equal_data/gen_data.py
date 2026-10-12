# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ast
import sys
import torch


def gen_data_and_golden(x1_type, x1_shape, x1_range, x2_type, x2_shape, x2_range):
    x1_type = getattr(torch, x1_type)
    x1_shape = ast.literal_eval(x1_shape)
    x1_range = ast.literal_eval(x1_range)
    x2_type = getattr(torch, x2_type)
    x2_shape = ast.literal_eval(x2_shape)
    x2_range = ast.literal_eval(x2_range)

    x1 = (x1_range[0] + (x1_range[1] - x1_range[0]) * torch.rand(x1_shape)).to(x1_type)
    x2 = (x2_range[0] + (x2_range[1] - x2_range[0]) * torch.rand(x2_shape)).to(x2_type)

    y = x1 != x2

    x1.view(torch.uint8).numpy().tofile('x1.bin')
    x2.view(torch.uint8).numpy().tofile('x2.bin')
    y.view(torch.uint8).numpy().tofile('golden.bin')


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('Usage: python gen_data.py x1_type x1_shape x1_range x2_type x2_shape x2_range')
        exit(1)
    gen_data_and_golden(*sys.argv[1:])
