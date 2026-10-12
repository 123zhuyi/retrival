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
import numpy
import sys
import torch


def verify_result(y_type, y_loss):
    y_type = getattr(torch, y_type)
    y_loss = ast.literal_eval(y_loss)

    y = torch.from_numpy(numpy.fromfile('y.bin', numpy.uint8)).view(y_type)
    golden = torch.from_numpy(numpy.fromfile('golden.bin', numpy.uint8)).view(y_type)

    if y_loss == 0:
        return torch.equal(y, golden)

    y = y.float()
    golden = golden.float()
    abs_err = (y - golden).abs()
    rel_err = abs_err / golden.abs()
    elem_bad = ~((abs_err <= y_loss) | (rel_err <= y_loss))
    bad_ratio = elem_bad.sum().item()
    return bad_ratio <= y_loss


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python compare_data.py y_type y_loss')
        exit(1)
    if not verify_result(*sys.argv[1:]):
        exit(1)
