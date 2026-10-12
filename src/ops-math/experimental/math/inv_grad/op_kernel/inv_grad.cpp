/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file inv_grad.cpp
 * \brief InvGrad kernel entry point (arch35 / Ascend950)
 *
 * Kernel signature dispatched by registry-invoke.  TilingKey selects D_T_Y:
 *   TilingKey 0: D_T_Y = float       (fp32 direct: Mul, Mul, Muls(-1))
 *   TilingKey 1: D_T_Y = half        (fp16 up-cast: Cast -> Mul -> Mul -> Muls(-1) -> Cast)
 *   TilingKey 2: D_T_Y = bfloat16_t  (bf16 up-cast: Cast -> Mul -> Mul -> Muls(-1) -> Cast)
 */

#include "inv_grad.h"

template <typename D_T_Y>
__global__ __aicore__ void inv_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(InvGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(InvGradTilingData, tilingData, tiling);
    NsInvGrad::InvGrad<D_T_Y> op;
    op.Init(y, dy, dx, &tilingData);
    op.Process();
}
