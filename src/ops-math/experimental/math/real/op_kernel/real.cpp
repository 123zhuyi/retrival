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
 * \file real.cpp
 * \brief real
 */

#include "real_tiling.h"
#include "real_kernel.h"

using namespace AscendC;
using namespace RealNs;

// Tiling key constants
#define COMPLEX32_MODE 1
#define COMPLEX64_MODE 2
#define FLOAT16_MODE 4
#define FLOAT_MODE 5

extern "C" __global__ __aicore__ void real(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RealTilingData);
    GET_TILING_DATA_WITH_STRUCT(RealTilingData, realTilingData, tiling);

    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(COMPLEX32_MODE)) { // COMPLEX32_MODE
        RealKernel<int32_t, half> op;
        op.Init(x, y, &realTilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(COMPLEX64_MODE)) { // COMPLEX64_MODE
        RealKernel<int64_t, float> op;
        op.Init(x, y, &realTilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FLOAT16_MODE)) { // FLOAT16_MODE
        RealKernel<half, half> op;
        op.Init(x, y, &realTilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FLOAT_MODE)) { // FLOAT_MODE
        RealKernel<float, float> op;
        op.Init(x, y, &realTilingData, &pipe);
        op.Process();
    }
}
