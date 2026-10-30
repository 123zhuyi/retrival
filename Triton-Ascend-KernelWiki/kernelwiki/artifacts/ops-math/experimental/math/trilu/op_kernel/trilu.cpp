/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/trilu_simt.h"

enum class TriluTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_INT32 = 2,
    TILING_KEY_INT64 = 3,
    TILING_KEY_INT8 = 4,
    TILING_KEY_INT16 = 5,
    TILING_KEY_UINT8 = 6,
    TILING_KEY_UINT16 = 7,
};

template <uint32_t schMode>
__global__ __aicore__ void trilu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TriluTilingData);
    GET_TILING_DATA_WITH_STRUCT(TriluTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_FLOAT)) {
        NsTrilu::Process<float>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_FLOAT16)) {
        NsTrilu::Process<half>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_INT32)) {
        NsTrilu::Process<int32_t>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_INT64)) {
        NsTrilu::Process<int64_t>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_INT8)) {
        NsTrilu::Process<int8_t>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_INT16)) {
        NsTrilu::Process<int16_t>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_UINT8)) {
        NsTrilu::Process<uint8_t>(x, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(TriluTilingKey::TILING_KEY_UINT16)) {
        NsTrilu::Process<uint16_t>(x, y, &tilingData);
    }
}
