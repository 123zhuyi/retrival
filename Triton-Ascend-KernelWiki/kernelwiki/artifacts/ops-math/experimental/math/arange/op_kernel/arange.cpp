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
 * \file arange.cpp
 * \brief
 */

#include "arange.h"

enum class ArangeTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_OTHER = 0,
    TILING_KEY_EXAMPLE_FLOAT = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void arange(
    GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArangeTilingData, tilingData, tiling);

    // 场景1
    if constexpr (schMode == static_cast<uint32_t>(ArangeTilingKey::TILING_KEY_EXAMPLE_OTHER)) {
        NsArange::KernelArange_Cast<DTYPE_START, DTYPE_STEP, DTYPE_OUT> op;

        op.Init(
            start, end, step, out, tilingData.totalNum, tilingData.unitNum, tilingData.unitLoops, tilingData.tailNum);

        op.Process();
    }

    // 场景2
    if constexpr (schMode == static_cast<uint32_t>(ArangeTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsArange::KernelArange<float, float, float> op;

        op.Init(
            start, end, step, out, tilingData.totalNum, tilingData.unitNum, tilingData.unitLoops, tilingData.tailNum);

        op.Process();
    }
}
