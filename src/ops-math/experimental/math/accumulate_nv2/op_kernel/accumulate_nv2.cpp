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
 * \file accumulate_nv2.cpp
 * \brief
 */

#include "accumulate_nv2.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class AccumulateNv2TilingKey : uint32_t {
    TILING_KEY_ONE_INPUT = 0,
    TILING_KEY_INPUTS = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void accumulate_nv2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AccumulateNv2TilingData);
    GET_TILING_DATA_WITH_STRUCT(AccumulateNv2TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    // 场景1
    if constexpr (
        schMode == static_cast<uint32_t>(AccumulateNv2TilingKey::TILING_KEY_ONE_INPUT) ||
        schMode == static_cast<uint32_t>(AccumulateNv2TilingKey::TILING_KEY_INPUTS)) {
        if constexpr (AscendC::IsSameType<DTYPE_X, int8_t>::value || AscendC::IsSameType<DTYPE_X, uint8_t>::value) {
            MyAccumulateNv2::AccumulateNv2<DTYPE_X, half> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
        if constexpr (AscendC::IsSameType<DTYPE_X, half>::value) {
            MyAccumulateNv2::AccumulateNv2<DTYPE_X, float> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
        if constexpr (AscendC::IsSameType<DTYPE_X, float>::value || AscendC::IsSameType<DTYPE_X, int32_t>::value) {
            MyAccumulateNv2::AccumulateNv2<DTYPE_X, DTYPE_X> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
    }
}
