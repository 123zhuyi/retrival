/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumulative_logsumexp_simt.h"
#include "cumulative_logsumexp_tiling_key.h"

enum class CumulativeLogsumexpTilingKey : uint32_t {
    TILING_KEY_FP32 = CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT,
    TILING_KEY_FP16 = CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16,
};

template <uint32_t schMode>
__global__ __aicore__ void cumulative_logsumexp(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)axis;
    (void)workspace;
    REGISTER_TILING_DEFAULT(CumulativeLogsumexpTilingData);
    GET_TILING_DATA_WITH_STRUCT(CumulativeLogsumexpTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(CumulativeLogsumexpTilingKey::TILING_KEY_FP32)) {
        NsCumulativeLogsumexp::CumulativeLogsumexpSimt::Process<float>(x, y, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(CumulativeLogsumexpTilingKey::TILING_KEY_FP16)) {
        NsCumulativeLogsumexp::CumulativeLogsumexpSimt::Process<half>(x, y, &tilingData);
    }
}
