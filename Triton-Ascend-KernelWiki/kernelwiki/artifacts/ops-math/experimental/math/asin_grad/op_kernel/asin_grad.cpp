/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "asin_grad.h"

template <typename D_T_Y>
__global__ __aicore__ void asin_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AsinGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinGradTilingData, tilingData, tiling);
    if constexpr (std::is_same_v<D_T_Y, float>) {
        NsAsinGrad::KernelAsinGradFp32<D_T_Y> op;
        op.Init(y, dy, z, &tilingData);
        op.Process();
    } else {
        NsAsinGrad::KernelAsinGradCast<D_T_Y> op;
        op.Init(y, dy, z, &tilingData);
        op.Process();
    }
}
