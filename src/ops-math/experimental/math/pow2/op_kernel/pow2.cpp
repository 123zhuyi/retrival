/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pow2.cpp
 * \brief
 */

#include "pow2.h"

template <uint32_t schMode>
__global__ __aicore__ void pow2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Pow2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Pow2TilingData, tilingData, tiling);
    NsPow2::Pow2<DTYPE_X1, DTYPE_X2, DTYPE_Y> op; // 算子kernel实例获取
    op.Init(x1, x2, y, &tilingData);              // 算子kernel实例初始化
    op.Process();                                 // 算子kernel实例执行
}
