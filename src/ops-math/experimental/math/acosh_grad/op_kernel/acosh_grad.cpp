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
 * \file acosh_grad.cpp
 * \brief AcoshGrad 算子 kernel 入口
 *
 * schMode 0 → half  (fp16)
 * schMode 1 → float (fp32)
 * schMode 2 → bfloat16_t (bf16)
 */

#include "acosh_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void acosh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AcoshGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AcoshGradTilingData, tilingData, tiling);

    if constexpr (schMode == ACOSHGRAD_TPL_SCH_MODE_0) {
        NsAcoshGrad::AcoshGrad<half> op;
        op.Init(y, dy, dx, &tilingData);
        op.Process();
    }
    if constexpr (schMode == ACOSHGRAD_TPL_SCH_MODE_1) {
        NsAcoshGrad::AcoshGrad<float> op;
        op.Init(y, dy, dx, &tilingData);
        op.Process();
    }
    if constexpr (schMode == ACOSHGRAD_TPL_SCH_MODE_2) {
        NsAcoshGrad::AcoshGrad<bfloat16_t> op;
        op.Init(y, dy, dx, &tilingData);
        op.Process();
    }
}
