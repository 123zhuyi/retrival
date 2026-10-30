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
 * \file fused_mul_add_nl2loss_apt.cpp
 * \brief FusedMulAddNL2loss arch35 kernel entry（fp16/fp32 双二进制，DTYPE_X1 编译期分发）
 */

#include <type_traits>
#include "arch35/fused_mul_add_nl2loss.h"

using namespace FusedMulAddNL2lossOps;

extern "C" __global__ __aicore__ void fused_mul_add_nl2loss(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(FusedMulAddNL2lossTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (std::is_same<DTYPE_X1, half>::value) {
        FusedMulAddNL2lossKernel<half> op;
        op.Init(x1, x2, x3, y1, y2, workspace, &tilingData, &pipe);
        op.Process();
    } else {
        FusedMulAddNL2lossKernel<float> op;
        op.Init(x1, x2, x3, y1, y2, workspace, &tilingData, &pipe);
        op.Process();
    }
}
