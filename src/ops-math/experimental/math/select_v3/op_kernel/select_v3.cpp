/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file select_v3.cpp
 */

#include "select_v3.h"

enum class SelectV3TilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_HALF = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void select_v3(GM_ADDR x, GM_ADDR y, GM_ADDR b, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SelectV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(SelectV3TilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SelectV3TilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsSelectV3::SelectV3<float> op;   // 算子kernel实例获取
        op.Init(x, y, b, z, &tilingData); // 算子kernel实例初始化
        op.Process();                     // 算子kernel实例执行
    } else if constexpr (schMode == static_cast<uint32_t>(SelectV3TilingKey::TILING_KEY_EXAMPLE_HALF)) {
        NsSelectV3::SelectV3<half> op;    // 算子kernel实例获取
        op.Init(x, y, b, z, &tilingData); // 算子kernel实例初始化
        op.Process();                     // 算子kernel实例执行
    }
}