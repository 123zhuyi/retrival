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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file tan_v3_arch32.cpp
 * \brief TanV3 kernel entry point (arch32 architecture - Ascend910B)
 */

#include "tan_v3.h"

enum class TanV3TilingKey : uint32_t {
    TILING_KEY_FLOAT32 = 0,
    TILING_KEY_FLOAT16 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void tan_v3(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(TanV3TilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(TanV3TilingKey::TILING_KEY_FLOAT32)) {
        NsTanV3::TanV3<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TanV3TilingKey::TILING_KEY_FLOAT16)) {
        NsTanV3::TanV3<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
