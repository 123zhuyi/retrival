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
 * \file softsign_apt.cpp
 * \brief softsign kernel entry
 */

#include "arch35/softsign_simt.h"

enum class SoftsignTilingKey : uint32_t {
    TILING_KEY_FP32 = 0,
    TILING_KEY_FP16 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void softsign(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftsignTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftsignTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SoftsignTilingKey::TILING_KEY_FP32)) {
        NsSoftsign::Process<float>(x, y, &tilingData);
    }

    if constexpr (schMode == static_cast<uint32_t>(SoftsignTilingKey::TILING_KEY_FP16)) {
        NsSoftsign::Process<half>(x, y, &tilingData);
    }
}
