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
 * \file erf.cpp
 * \brief
 */

#include "erf.h"

enum class ErfTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_ZERO = 0,
    TILING_KEY_EXAMPLE_ONE = 1,
};
template <uint32_t schMode>
__global__ __aicore__ void erf(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (schMode == static_cast<uint32_t>(ErfTilingKey::TILING_KEY_EXAMPLE_ONE)) {
        MyErf::KernelErf<DTYPE_SELF, DTYPE_OUT> op;
        op.Init(
            self, out, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.tileDataNum,
            tilingData.bigTailDataNum, tilingData.smallCoreDataNum, tilingData.finalSmallTileNum,
            tilingData.smallTailDataNum,
            &pipe);   // 算子kernel实例初始化
        op.Process(); // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ErfTilingKey::TILING_KEY_EXAMPLE_ZERO)) {
        MyErf::KernelErf_single_core<DTYPE_SELF, DTYPE_OUT> op;
        op.Init(self, out, tilingData.tileDataNum, tilingData.bigTailDataNum, &pipe);
    }
}