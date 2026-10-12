# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import numpy as np
import stat
import torch

OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def write_file(shape, input):
    x = np.random.randint(1, 255, shape).astype(np.float32)
    x.tofile("./x.bin")


def gen_tiling():
    batch = 1
    channel = 1
    height = 64
    width = 64
    alignHeight = 64
    alignWidth = 64
    outHeight = 64
    outWidth = 62
    alignOutHeight = 64
    alignOutWidth = 64
    padTop = 0
    padBottom = 0
    padLeft = 1
    padRight = 1
    blockNum = 1
    ubFactorElement = 112
    ncPerCore = 1
    tailNC = 0
    tilingKey = 1101
    workspacePerCore = 0
    tiling = (np.array(i, dtype=np.uint32) for i in (batch, channel, height, width, alignHeight, alignWidth, 
                                                     outHeight, outWidth, alignOutHeight, alignOutWidth, 
                                                     padTop, padBottom, padLeft, padRight,
                                                     blockNum, ubFactorElement, ncPerCore, tailNC, tilingKey,
                                                     workspacePerCore
                                                     ))
    tiling_data = b''.join(x.tobytes() for x in tiling)

    with os.fdopen(os.open('./tiling.bin', WRITE_FILE_FLAGS, OPEN_FILE_MODES_640), 'wb') as f:
        f.write(tiling_data)


if __name__ == "__main__":
    x_shape = [1, 1, 64, 64]
    write_file(x_shape, "x_shape")
    gen_tiling()