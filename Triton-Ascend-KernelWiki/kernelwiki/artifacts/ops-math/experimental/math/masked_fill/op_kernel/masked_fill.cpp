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
 * \file masked_fill.cpp
 * \brief
 */
#include "masked_fill.h"
using namespace NsMaskedFill;

enum class MaskedFillTilingKey : uint32_t {
    TILING_KEY_TPL_BF16 = 27,   // bfloat16
    TILING_KEY_TPL_UINT8 = 4,   // uint8
    TILING_KEY_TPL_FP16 = 1,    // float16
    TILING_KEY_TPL_FP32 = 0,    // float32
    TILING_KEY_TPL_INT8 = 2,    // int8
    TILING_KEY_TPL_INT16 = 6,   // int16
    TILING_KEY_TPL_INT32 = 3,   // int32
    TILING_KEY_TPL_FP64 = 11,   // float64
    TILING_KEY_TPL_UINT16 = 7,  // uint16
    TILING_KEY_TPL_UINT32 = 8,  // uint32
    TILING_KEY_TPL_UINT64 = 10, // uint64
    TILING_KEY_TPL_INT64 = 9,   // int64
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
__global__ __aicore__ void masked_fill(
    GM_ADDR x, GM_ADDR mask, GM_ADDR value, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MaskedFillTilingData);
    GET_TILING_DATA_WITH_STRUCT(MaskedFillTilingData, tilingData, tiling);

    if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_BF16)) {
        MaskedFill<BF16> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_UINT8)) {
        MaskedFill<UINT8> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_FP16)) {
        MaskedFill<FP16> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_FP32)) {
        MaskedFill<FP32> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_INT8)) {
        MaskedFill<INT8> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_INT16)) {
        MaskedFill<INT16> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_INT32)) {
        MaskedFill<INT32> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_FP64)) {
        MaskedFill<FP64> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_UINT16)) {
        MaskedFill<UINT16> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_UINT32)) {
        MaskedFill<UINT32> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_UINT64)) {
        MaskedFill<UINT64> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    } else if constexpr (TYPE_X == static_cast<uint32_t>(MaskedFillTilingKey::TILING_KEY_TPL_INT64)) {
        MaskedFill<INT64> op;
        op.Init(x, mask, value, z, &tilingData);
        op.Process();
    }
}