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
 * \file kl_div_v2.cpp
 * \brief
 */

#include "kl_div_v2.h"

enum class KLDivV2TilingKey : uint32_t {
    TILING_KEY_KLDIVV2_MODE_0 = 0,
    TILING_KEY_KLDIVV2_MODE_1 = 1,
    TILING_KEY_KLDIVV2_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void kl_div_v2(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KLDivV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(KLDivV2TilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(KLDivV2TilingKey::TILING_KEY_KLDIVV2_MODE_0)) {
        NsKLDivV2::KLDivV2<half> op;
        op.Init(x, target, y, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(KLDivV2TilingKey::TILING_KEY_KLDIVV2_MODE_1)) {
        NsKLDivV2::KLDivV2<float> op;
        op.Init(x, target, y, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(KLDivV2TilingKey::TILING_KEY_KLDIVV2_MODE_2)) {
        NsKLDivV2::KLDivV2<bfloat16_t> op;
        op.Init(x, target, y, workspace, &tilingData);
        op.Process();
    }
}
