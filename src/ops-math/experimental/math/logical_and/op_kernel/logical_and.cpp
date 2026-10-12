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
 * \file logical_and.cpp
 * \brief
 */

#include "logical_and.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class LogicalAndTilingKey : uint32_t {
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void logical_and(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogicalAndTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogicalAndTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    // 场景1
    if constexpr (schMode == static_cast<uint32_t>(LogicalAndTilingKey::TILING_KEY_DB)) {
        MyLogicalAnd::LogicalAnd<DOUBLE_BUFFER_NUM> op; // 算子kernel实例获取
        op.Init(x1, x2, y, &tilingData, &pipe);         // 算子kernel实例初始化
        op.Process();                                   // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(LogicalAndTilingKey::TILING_KEY_NDB)) {
        MyLogicalAnd::LogicalAnd<SINGLE_BUFFER_NUM> op; // 算子kernel实例获取
        op.Init(x1, x2, y, &tilingData, &pipe);         // 算子kernel实例初始化
        op.Process();                                   // 算子kernel实例执行
    }
}
