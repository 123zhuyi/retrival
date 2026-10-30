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
 * \file real_div.cpp
 * \brief
 */

#include "real_div.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class RealDivTilingKey : uint32_t {
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void real_div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RealDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(RealDivTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(RealDivTilingKey::TILING_KEY_DB)) {
        MyRealDiv::KernelRealDiv<DTYPE_X1, DTYPE_Y, DOUBLE_BUFFER_NUM> op;
        op.Init(x1, x2, y, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(RealDivTilingKey::TILING_KEY_NDB)) {
        MyRealDiv::KernelRealDiv<DTYPE_X1, DTYPE_Y, SINGLE_BUFFER_NUM> op;
        op.Init(x1, x2, y, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();
    }
}