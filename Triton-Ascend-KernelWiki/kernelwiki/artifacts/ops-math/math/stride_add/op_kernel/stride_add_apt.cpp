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
 * \file stride_add_apt.cpp
 * \brief StrideAdd operator entry point
 */

#include "arch35/stride_add_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void stride_add(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StrideAddTilingData);
    GET_TILING_DATA_WITH_STRUCT(StrideAddTilingData, tilingData, tiling);

    // DTYPE_X1 宏自动实例化所有 dtype 组合 (fp16/fp32/bf16)
    using T = DTYPE_X1;

    // R003+R006: 根据索引参数位宽选择实例化
    // 判断条件：所有 VF 内索引参数（totalElements + x1NStride + x2NStride + hwC0Size + c1Len + perCoreElements）
    // 都必须在 uint32_t 范围内，否则地址计算会截断
    bool use32Bit = (tilingData.totalElements <= static_cast<int64_t>(UINT32_MAX))
        && (tilingData.x1NStride <= static_cast<int64_t>(UINT32_MAX))
        && (tilingData.x2NStride <= static_cast<int64_t>(UINT32_MAX))
        && (tilingData.hwC0Size <= static_cast<int64_t>(UINT32_MAX))
        && (tilingData.c1Len <= static_cast<int64_t>(UINT32_MAX))
        && (tilingData.perCoreElements <= static_cast<int64_t>(UINT32_MAX));

    if (use32Bit) {
        if constexpr (schMode == STRIDE_ADD_TPL_SCH_MODE_0) {
            NsStrideAdd::Process<T, uint32_t>(x1, x2, y, workspace, tiling, &tilingData);
        }
    } else {
        if constexpr (schMode == STRIDE_ADD_TPL_SCH_MODE_0) {
            NsStrideAdd::Process<T, uint64_t>(x1, x2, y, workspace, tiling, &tilingData);
        }
    }
}