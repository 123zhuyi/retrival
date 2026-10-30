/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file expandv.cpp
 * \brief
 */

#include "expandv.h"

enum class ExpandvTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
    TILING_KEY_EXAMPLE_FLOAT16 = 2,
    TILING_KEY_EXAMPLE_BF16 = 3,
    TILING_KEY_EXAMPLE_INT8 = 4,
    TILING_KEY_EXAMPLE_UINT8 = 5,
    TILING_KEY_EXAMPLE_BOOL = 6,
    TILING_KEY_EXAMPLE_INT16 = 7,
    TILING_KEY_EXAMPLE_UINT16 = 8,
    TILING_KEY_EXAMPLE_UINT32 = 9,
};

template <uint32_t schMode>
__global__ __aicore__ void expandv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpandvTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpandvTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsExpandv::Expandv<float> op; // 算子kernel实例获取
        op.Init(x, y, &tilingData);   // 算子kernel实例初始化
        op.Process();                 // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsExpandv::Expandv<int32_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_FLOAT16)) {
        NsExpandv::Expandv<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_BF16)) {
        NsExpandv::Expandv<bfloat16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_INT8)) {
        NsExpandv::Expandv<int8_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_UINT8)) {
        NsExpandv::Expandv<uint8_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_BOOL)) {
        NsExpandv::Expandv<bool> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_INT16)) {
        NsExpandv::Expandv<int16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_UINT16)) {
        NsExpandv::Expandv<uint16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpandvTilingKey::TILING_KEY_EXAMPLE_UINT32)) {
        NsExpandv::Expandv<uint32_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
