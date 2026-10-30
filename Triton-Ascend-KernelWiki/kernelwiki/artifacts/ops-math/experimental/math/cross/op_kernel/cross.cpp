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
 * \file cross.cpp
 * \brief
 */
#include "cross.h"

enum class CrossTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
    TILING_KEY_EXAMPLE_INT8 = 2,
    TILING_KEY_EXAMPLE_HALF = 3,
    TILING_KEY_EXAMPLE_UINT8 = 4,
    TILING_KEY_EXAMPLE_INT16 = 5
};

template <uint32_t schMode>
__global__ __aicore__ void cross(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CrossTilingData);
    GET_TILING_DATA_WITH_STRUCT(CrossTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsCross::Cross<float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsCross::Cross<int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_INT8)) {
        NsCross::Cross<int8_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_HALF)) {
        NsCross::Cross<half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_UINT8)) {
        NsCross::Cross<uint8_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(CrossTilingKey::TILING_KEY_EXAMPLE_INT16)) {
        NsCross::Cross<int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
}
