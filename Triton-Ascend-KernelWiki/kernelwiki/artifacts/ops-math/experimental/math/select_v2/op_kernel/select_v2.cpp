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
 * \file select_v2.cpp
 * \brief
 */

#include "select_v2.h"

enum class SelectV2TilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_OTHER = 1,
};
template <uint32_t schMode>
__global__ __aicore__ void select_v2(
    GM_ADDR condition, GM_ADDR self, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SelectV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SelectV2TilingData, tilingData, tiling);
    MySelectV2::KernelSelectV2<DTYPE_SELF, DTYPE_OUT> op;
    op.Init(
        condition, self, other, out, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum,
        tilingData.finalSmallTileNum, tilingData.tileDataNum, tilingData.smallTailDataNum, tilingData.bigTailDataNum,
        tilingData.tailBlockNum); // 算子kernel实例初始化
    op.Process();
}
