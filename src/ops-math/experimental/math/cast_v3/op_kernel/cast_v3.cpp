/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_bf16.h"
#include "cast_copy.h"
#include "cast_generic.h"
#include "cast_expand.h"
#include "cast_tiling_data.h"
#include "cast_tiling_key.h"
#include "kernel_tiling/kernel_tiling.h"

// bfloat16_t is not defined in AscendC AICore kernel headers, but the build system
// passes -DDTYPE_X=bfloat16_t/-DDTYPE_Y=bfloat16_t for BF16 type kernels.
// Use typedef (bit-compatible with uint16_t) instead of #define to avoid macro pollution.
#ifndef bfloat16_t
typedef uint16_t bfloat16_t;
#endif

template <uint32_t schMode>
__global__ __aicore__ void cast_v3(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(CastTilingData);
    GET_TILING_DATA_WITH_STRUCT(CastTilingData, tiling_data, tiling);
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        // BF16 input always goes through CastBf16<uint16_t, ...> (bit-compatible with bfloat16)
        AscendC::CastBf16<uint16_t, DTYPE_Y> op(x, y, workspace, tiling_data);
        op.Process();
    } else if constexpr (std::is_same_v<DTYPE_Y, bfloat16_t>) {
        // BF16 output always goes through CastBf16 for ComputeToBF16 path
        AscendC::CastBf16<DTYPE_X, uint16_t> op(x, y, workspace, tiling_data);
        op.Process();
    } else if (tiling_data.tilingKey == 1) {
        if constexpr (std::is_same_v<DTYPE_X, half>) {
            AscendC::CastBf16<half, int16_t> op(x, y, workspace, tiling_data);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, int64_t>) {
            AscendC::CastBf16<int64_t, DTYPE_Y> op(x, y, workspace, tiling_data);
            op.Process();
        } else {
            AscendC::CastBf16<int16_t, DTYPE_Y> op(x, y, workspace, tiling_data);
            op.Process();
        }
    } else if (tiling_data.tilingKey == 4) {
        AscendC::CastCopy op(x, y, workspace, tiling_data);
        op.Process();
    } else if (tiling_data.tilingKey == 5) {
        if constexpr (std::is_same_v<DTYPE_X, bool>) {
            AscendC::CastExpand<uint8_t, DTYPE_Y> op(x, y, workspace, tiling_data);
            op.Process();
        } else {
            AscendC::CastExpand<DTYPE_X, DTYPE_Y> op(x, y, workspace, tiling_data);
            op.Process();
        }
    } else {
        AscendC::CastGeneric<DTYPE_X, DTYPE_Y> op(x, y, workspace, tiling_data);
        op.Process();
    }
}
