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
 * \file rsqrt.cpp
 * \brief
 */

#include "rsqrt.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class RsqrtTilingKey : uint32_t {
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void rsqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RsqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(RsqrtTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if constexpr (schMode == static_cast<uint32_t>(RsqrtTilingKey::TILING_KEY_DB)) {
        NsRsqrt::KernelRsqrt<DTYPE_X, DOUBLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(RsqrtTilingKey::TILING_KEY_NDB)) {
        NsRsqrt::KernelRsqrt<DTYPE_X, SINGLE_BUFFER_NUM> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
