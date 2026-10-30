/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file col2_im.cpp
 * \brief
 */

#include "col2_im.h"

enum class Col2ImTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void col2_im(GM_ADDR col, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Col2ImTilingData);
    GET_TILING_DATA_WITH_STRUCT(Col2ImTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(Col2ImTilingKey::TILING_KEY_FLOAT)) {
        NsCol2Im::Col2Im<float> op;  // 算子kernel实例获取
        op.Init(col, x, tilingData); // 算子kernel实例初始化
        op.Process();                // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(Col2ImTilingKey::TILING_KEY_FLOAT16)) {
        NsCol2Im::Col2Im<half> op;   // 算子kernel实例获取
        op.Init(col, x, tilingData); // 算子kernel实例初始化
        op.Process();                // 算子kernel实例执行
    }
}
