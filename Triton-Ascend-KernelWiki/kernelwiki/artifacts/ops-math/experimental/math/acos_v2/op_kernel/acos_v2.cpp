/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "acos_v2.h"

enum class AcosV2TilingKey : uint32_t {
    TILING_KEY_FP32 = 0,
    TILING_KEY_FP16 = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t CALC_MODE>
__global__ __aicore__ void acos_v2(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AcosV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(AcosV2TilingData, tilingData, tiling);

    if constexpr (CALC_MODE == static_cast<uint32_t>(AcosV2TilingKey::TILING_KEY_FP32)) {
        NsAcos::AcosV2Kernel<float, ACOS_MODE_FP32> op;
        op.Init(self, out, &tilingData);
        op.Process();
    }
    if constexpr (CALC_MODE == static_cast<uint32_t>(AcosV2TilingKey::TILING_KEY_FP16)) {
        NsAcos::AcosV2Kernel<half, ACOS_MODE_FP16> op;
        op.Init(self, out, &tilingData);
        op.Process();
    }
    if constexpr (CALC_MODE == static_cast<uint32_t>(AcosV2TilingKey::TILING_KEY_BF16)) {
        NsAcos::AcosV2Kernel<bfloat16_t, ACOS_MODE_BF16> op;
        op.Init(self, out, &tilingData);
        op.Process();
    }
}
