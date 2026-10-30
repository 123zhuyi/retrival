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
 * \file stateless_bernoulli.cpp
 * \brief
 */

#include "arch35/stateless_bernoulli_simt.h"
#define STATELESS_BERNOULLI_DEFAULT_TILING_KEY 100

using namespace StatelessBernoulli;

__global__ __aicore__ void stateless_bernoulli(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(STATELESS_BERNOULLI_DEFAULT_TILING_KEY)) {
        using To = typename AscendC::Conditional<AscendC::IsSameType<DTYPE_Y, bool>::value, int8_t, DTYPE_Y>::type;
        if constexpr (AscendC::IsSameType<DTYPE_PROB, float>::value) {
            StatelessBernoulliKernel<float, To> op;
            op.Process(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, half>::value) {
            StatelessBernoulliKernel<half, To> op;
            op.Process(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, bfloat16_t>::value) {
            StatelessBernoulliKernel<bfloat16_t, To> op;
            op.Process(prob, y, &tilingData);
        }
    }
}
