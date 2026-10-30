/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "../split_v/arch35/split_v_pure_copy_same_len.h"
#include "../split_v/arch35/split_v_ub_split_same_len.h"
#include "../split_v/arch35/split_v_ub_split_same_len_small_g.h"
#include "../split_v/arch35/split_v_ub_split_same_len_deinterleave.h"
#include "kernel_operator.h"
using namespace AscendC;
using namespace Ops::Base;

#define TILING_KEY_PURE_MOVE_SAME_LEN 100
#define TILING_KEY_UB_SPLIT_SAME_LEN 101
#define TILING_KEY_UB_SPLIT_SAME_LEN_SMALL_G 111
#define TILING_KEY_UB_SPLIT_SAME_LEN_DEINTERLEAVE 112
extern "C" __global__ __aicore__ void split(GM_ADDR split_dim, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_PURE_MOVE_SAME_LEN)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint8_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint64_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLen<uint8_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLen<uint16_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLen<uint32_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLen<uint64_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN_SMALL_G)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint8_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint16_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint32_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint64_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN_DEINTERLEAVE)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint8_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint64_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    }
}
