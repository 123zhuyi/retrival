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

/**
 * \file log_add_exp_arch32.cpp
 * \brief LogAddExp kernel entry (arch32)
 */

#include "log_add_exp.h"

enum class LogAddExpTilingKey : uint32_t {
    TILING_KEY_FP32 = 0,
    TILING_KEY_FP16 = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void log_add_exp(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogAddExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogAddExpTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(LogAddExpTilingKey::TILING_KEY_FP32)) {
        NsLogAddExp::LogAddExp<float, float> op;
        op.Init(x, y, out, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LogAddExpTilingKey::TILING_KEY_FP16)) {
        NsLogAddExp::LogAddExp<half, float> op;
        op.Init(x, y, out, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LogAddExpTilingKey::TILING_KEY_BF16)) {
        NsLogAddExp::LogAddExp<bfloat16_t, float> op;
        op.Init(x, y, out, workspace, &tilingData);
        op.Process();
    }
}
