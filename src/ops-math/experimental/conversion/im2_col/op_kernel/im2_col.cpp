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
 * \file im2_col.cpp
 * \brief
 */

#include "im2_col.h"

enum class Im2ColTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_FLOAT16 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void im2_col(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Im2ColTilingData);
    GET_TILING_DATA_WITH_STRUCT(Im2ColTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(Im2ColTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsIm2Col::Im2Col<float> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);  // 算子kernel实例初始化
        op.Process();               // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(Im2ColTilingKey::TILING_KEY_EXAMPLE_FLOAT16)) {
        NsIm2Col::Im2Col<half> op; // 算子kernel实例获取
        op.Init(x, z, tilingData); // 算子kernel实例初始化
        op.Process();              // 算子kernel实例执行
    }
}
