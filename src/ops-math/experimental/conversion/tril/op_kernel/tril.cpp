/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file tril.cpp
 * \brief
 */

#include "tril.h"
enum class AddExampleTilingKey : uint32_t {
    TILING_KEY_NaivePath = 1,
    TILING_KEY_SheerDup = 2,
    TILING_KEY_SheerZero = 3,
    TILING_KEY_FastPath = 4,
};
template <uint32_t schMode>
__global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TrilTilingData);
    GET_TILING_DATA_WITH_STRUCT(TrilTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_NaivePath)) {
        NsTril::Tril<DTYPE_X> op;      // 算子kernel实例获取
        op.Init(x, y, &tilingData, 1); // 算子kernel实例初始化
        op.Process();                  // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_SheerDup)) {
        NsTril::Tril<DTYPE_X> op;      // 算子kernel实例获取
        op.Init(x, y, &tilingData, 2); // 算子kernel实例初始化
        op.Process();                  // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_SheerZero)) {
        NsTril::Tril<DTYPE_X> op;      // 算子kernel实例获取
        op.Init(x, y, &tilingData, 3); // 算子kernel实例初始化
        op.Process();                  // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_FastPath)) {
        NsTril::Tril<DTYPE_X> op;      // 算子kernel实例获取
        op.Init(x, y, &tilingData, 4); // 算子kernel实例初始化
        op.Process();                  // 算子kernel实例执行
    }
}