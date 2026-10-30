/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file reflection_pad1d_v2.cpp
 * \brief
 */
#include "reflection_pad1d_v2.h"
using namespace NsReflectionPad1dV2;

enum class ReflectionPad1dV2TilingKey : uint32_t {
    TILING_KEY_TPL_BF16 = 27,
    TILING_KEY_TPL_UINT8 = 4,
    TILING_KEY_TPL_FP16 = 1,
    TILING_KEY_TPL_FP32 = 0,
    TILING_KEY_TPL_INT8 = 2,
    TILING_KEY_TPL_INT16 = 6,
    TILING_KEY_TPL_INT32 = 3,
    TILING_KEY_TPL_FP64 = 11,
    TILING_KEY_TPL_UINT16 = 7,
    TILING_KEY_TPL_UINT32 = 8,
    TILING_KEY_TPL_UINT64 = 10,
    TILING_KEY_TPL_INT64 = 9,
};

using BF16 = bfloat16_t;
using UINT8 = uint8_t;
using FP16 = half;
using FP32 = float;
using INT8 = int8_t;
using INT16 = int16_t;
using INT32 = int32_t;
using FP64 = double;
using UINT16 = uint16_t;
using UINT32 = uint32_t;
using UINT64 = uint64_t;
using INT64 = int64_t;

// kernel function
template <uint32_t TYPE_X>
__global__ __aicore__ void reflection_pad1d_v2(GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(ReflectionPad1dV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReflectionPad1dV2TilingData, tilingData, tiling);

    if constexpr (TYPE_X == static_cast<uint32_t>(ReflectionPad1dV2TilingKey::TILING_KEY_TPL_BF16)) {
        ReflectionPad1dV2<BF16> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(ReflectionPad1dV2TilingKey::TILING_KEY_TPL_FP16)) {
        ReflectionPad1dV2<FP16> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(ReflectionPad1dV2TilingKey::TILING_KEY_TPL_FP32)) {
        ReflectionPad1dV2<FP32> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}