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
 * \file log.cpp
 * \brief
 */

#include "log.h"

enum class LogTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_BIGCORE = 0,
    TILING_KEY_EXAMPLE_SMALLCORE = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void log(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogTilingData, tilingData, tiling);

    // 场景1
    if constexpr (schMode == static_cast<uint32_t>(LogTilingKey::TILING_KEY_EXAMPLE_BIGCORE)) {
        NsLog::LogKernel<DTYPE_X, DTYPE_Y, schMode> op; // 算子kernel实例获取
        op.Init(
            x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.base, tilingData.scale,
            tilingData.shift); // 算子kernel实例初始化
        op.Process();          // 算子kernel实例执行
    }

    // 场景2
    if constexpr (schMode == static_cast<uint32_t>(LogTilingKey::TILING_KEY_EXAMPLE_SMALLCORE)) {
        NsLog::LogKernel<DTYPE_X, DTYPE_Y, schMode> op; // 算子kernel实例获取
        op.Init(
            x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.base, tilingData.scale,
            tilingData.shift); // 算子kernel实例初始化
        op.Process();          // 算子kernel实例执行
    }
}
