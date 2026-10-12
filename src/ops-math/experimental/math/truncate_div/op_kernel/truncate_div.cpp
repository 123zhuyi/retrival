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
 * \file truncate_div.cpp
 * \brief TruncateDiv kernel entry. schMode selects the (x1, x2, y) dtype combo.
 */

#include "truncate_div.h"

#if defined(ASCENDC_CPU_DEBUG)
#include <cstring>
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(structType, name, tilingArg) \
    structType name;                                             \
    (void)memcpy_s(&(name), sizeof(structType), (tilingArg), sizeof(structType))
#endif
#else
#include "truncate_div_tiling_key.h"
#endif

template <uint32_t schMode>
__global__ __aicore__ void truncate_div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateDivTilingData, tilingData, tiling);
    (void)workspace;

    if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_0) {
        NsTruncateDiv::Run<bfloat16_t, bfloat16_t, bfloat16_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_1) {
        NsTruncateDiv::Run<half, half, half>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_2) {
        NsTruncateDiv::Run<half, float, float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_3) {
        NsTruncateDiv::Run<float, half, float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_4) {
        NsTruncateDiv::Run<float, float, float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_5) {
        NsTruncateDiv::Run<float, int32_t, float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_6) {
        NsTruncateDiv::Run<int32_t, int32_t, int32_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_7) {
        NsTruncateDiv::Run<int32_t, float, float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_8) {
        NsTruncateDiv::Run<uint8_t, uint8_t, uint8_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_9) {
        NsTruncateDiv::Run<int8_t, int8_t, int8_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_10) {
        NsTruncateDiv::Run<int64_t, int64_t, int64_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATEDIV_TPL_SCH_MODE_11) {
        NsTruncateDiv::Run<int16_t, int16_t, int16_t>(x1, x2, y, &tilingData);
    }
}
