/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mul_addn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "mul_addn_align.h"
#include "mul_addn_align_bf16.h"

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelMulAddnAlign<float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMulAddnAlignF16<half, half> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(2)) {
        TPipe pipe;
        KernelMulAddnAlignF16<bfloat16_t, float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
#endif
    }
}