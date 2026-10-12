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
 * \file ones_like.cpp
 * \brief
 */

#include "ones_like.h"

enum class OnesLikeTilingKey : uint32_t {
    TILING_KEY_0 = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void ones_like(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(OnesLikeTilingData);
    GET_TILING_DATA_WITH_STRUCT(OnesLikeTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (std::is_same_v<bool, DTYPE_X>) {
        NsOnesLike::OnesLike<int8_t> op;   // 算子kernel实例获取
        op.Init(x, y, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();                      // 算子kernel实例执行
    } else {
        NsOnesLike::OnesLike<DTYPE_X> op;  // 算子kernel实例获取
        op.Init(x, y, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();                      // 算子kernel实例执行
    }
}
