#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
import torch
import sys


def gen_tiling(totalLength, dtype_str):
    sizeB8 = 1
    sizeB16 = 2
    sizeB32 = 4
    blockSize = 32
    BLOCK_DIM = 48
    SELECT_MODE_GE_ZERO_TMP_UB = 8000
    ubSizePlatForm = 192000
    dtype_dict = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "complex64": torch.complex64,
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64
    }
    bytesPerData = {
        "float32": 29,
        "float": 29,
        "float16": 15,
        "complex64": 66,
        "bool": 15,
        "uint8": 15,
        "int8": 21,
        "int16": 21,
        "int32": 25,
        "int64": 33
    }
    outDtype = dtype_dict[dtype_str]

    if outDtype == torch.float16:
        ALIGN_NUM = blockSize / sizeB16
    elif outDtype == torch.int8:
        ALIGN_NUM = blockSize / sizeB8
    elif outDtype == torch.int16:
        ALIGN_NUM = blockSize / sizeB16
    else:
        ALIGN_NUM = blockSize / sizeB32

    totalLengthAligned = int(((totalLength + ALIGN_NUM - 1) / ALIGN_NUM)) * ALIGN_NUM
    formerNum = int(totalLengthAligned / ALIGN_NUM) % BLOCK_DIM
    tailNum = BLOCK_DIM - formerNum
    formerLength = int((int(totalLengthAligned / BLOCK_DIM) + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM
    tailLength = int(int(totalLengthAligned / BLOCK_DIM) / ALIGN_NUM) * ALIGN_NUM

    dataPerRepeat = 256 / sizeB16 if outDtype == torch.float16 else 256 / sizeB32
    tileLength = int(int((ubSizePlatForm - SELECT_MODE_GE_ZERO_TMP_UB) / bytesPerData[dtype_str]) / dataPerRepeat) * dataPerRepeat
    variables_dict = {
        "totalLength": totalLength,
        "formerNum": formerNum,
        "tailNum": tailNum,
        "formerLength": int(formerLength),
        "tailLength": int(tailLength),
        "alignNum": int(ALIGN_NUM),
        "totalLengthAligned": int(totalLengthAligned),
        "tileLength": int(tileLength),
        "dataPerRepeat": int(dataPerRepeat)
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling(int(sys.argv[1]), (sys.argv[2]))  # python gen_tiling.py argv1 argv2
    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
