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
 * \file axpy_v2.cpp
 * \brief
 */

#include "axpy_v2.h"

enum class AxpyV2TilingKey : uint32_t {
    TILING_KEY_EXAMPLE_DB = 0,
    TILING_KEY_EXAMPLE_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void axpy_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AxpyV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(AxpyV2TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    if constexpr (schMode == static_cast<uint32_t>(AxpyV2TilingKey::TILING_KEY_EXAMPLE_DB)) {
        NsAxpyV2::AxpyV2<DTYPE_X1, DTYPE_X2, DTYPE_ALPHA, DTYPE_Y, 2> op; // 算子kernel实例获取
        op.Init(x1, x2, alpha, y, &tilingData, &pipe);                    // 算子kernel实例初始化
        op.Process();                                                     // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(AxpyV2TilingKey::TILING_KEY_EXAMPLE_NDB)) {
        NsAxpyV2::AxpyV2<DTYPE_X1, DTYPE_X2, DTYPE_ALPHA, DTYPE_Y, 1> op; // 算子kernel实例获取
        op.Init(x1, x2, alpha, y, &tilingData, &pipe);                    // 算子kernel实例初始化
        op.Process();                                                     // 算子kernel实例执行
    }
}
