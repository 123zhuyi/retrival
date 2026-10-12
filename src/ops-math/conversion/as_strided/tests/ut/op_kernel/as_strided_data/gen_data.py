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
import numpy as np
import torch

def gen_golden_data_simple():
    size = (1652720)
    low = 0
    high = 10
    # 创建原始数据
    original_data = (torch.rand(size) * (high - low) + low).to(torch.float16)
    #print(original_data)

    storage_offset = (0)
    size = (1024,64,8)
    stride = (53, 19, 32)
    size_array = np.array(size, dtype=np.int32)
    print(size_array)
    stride_array = np.array(stride, dtype=np.int32)
    print(stride_array)
    storage_offset_array = np.array(storage_offset, dtype=np.int32)

    y = torch.as_strided(original_data, size=size, stride=stride, storage_offset=storage_offset)
    #print(y)

    original_data.numpy().tofile("./x.bin")

    size_array.tofile("./size.bin")
    stride_array.tofile("./stride.bin")
    storage_offset_array.tofile("./offset.bin")
    y.numpy().tofile("./output.bin")

if __name__ == "__main__":
    gen_golden_data_simple()