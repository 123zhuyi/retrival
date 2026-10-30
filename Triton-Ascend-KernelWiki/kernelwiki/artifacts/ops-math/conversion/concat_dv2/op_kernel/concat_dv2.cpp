/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_dv2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "concat_dv2.h"

extern "C" __global__ __aicore__ void concat_dv2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;

    GET_TILING_DATA(tiling_data, tiling);
    #if (ORIG_DTYPE_X == DT_INT8 || ORIG_DTYPE_X == DT_UINT8 || ORIG_DTYPE_X == DT_BOOL)
        ConcatDV2<int8_t> op(pipe);
        op.Init(x, y, tiling_data);
        op.Process();

    #elif (ORIG_DTYPE_X == DT_FLOAT16 || ORIG_DTYPE_X == DT_BF16 || ORIG_DTYPE_X == DT_INT16)
        ConcatDV2<half> op(pipe);
        op.Init(x, y, tiling_data);
        op.Process();

    #elif (ORIG_DTYPE_X == DT_FLOAT || ORIG_DTYPE_X == DT_INT32)
        ConcatDV2<float> op(pipe);
        op.Init(x, y, tiling_data);
        op.Process();

    #elif (ORIG_DTYPE_X == DT_INT64 || ORIG_DTYPE_X == DT_DOUBLE || ORIG_DTYPE_X == DT_COMPLEX64)
        ConcatDV2<int64_t> op(pipe);
        op.Init(x, y, tiling_data);
        op.Process();
    #endif
}