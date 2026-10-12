/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bias_add.cpp
 * \brief bias_add kernel
 */

#include "kernel_operator.h"
#include "bias_add.h"
#include "bias_add_tiling_data.h"
#include "bias_add_tiling_key.h"

using namespace AscendC;

// Dependent-false helper for the unreachable if-constexpr branch below. DTYPE_X is a build
// macro (a concrete type), so sizeof(DTYPE_X)==0 would be a non-dependent constant and fire
// at parse time; keying the assert on the schMode template parameter defers it to actual
// instantiation, so it only triggers if an unsupported configuration is ever instantiated.
template <uint32_t Mode>
struct BiasAddAlwaysFalse {
    static constexpr bool value = false;
};

template <typename T>
__aicore__ inline bool ShouldUseThinSmallRuntimeInplace(const BiasAddTilingData& tilingData)
{
    const uint32_t total = static_cast<uint32_t>(tilingData.totalElements);
    const uint32_t channel = static_cast<uint32_t>(tilingData.channelSize);
    constexpr uint32_t kSmallInplaceBytes = 4096U;
    const bool smallInplace = total * sizeof(T) <= kSmallInplaceBytes;
    const bool narrowChannel = channel <= 16U;
    if constexpr (std::is_same<T, int32_t>::value) {
        // int32 always uses the in-place ThinTiny variant: no cast chain (unlike bf16),
        // and the single-buffer in-place GatherMask broadcast-add is valid at any size on
        // this path, so there is no correctness or measured-perf reason to size-gate it the
        // way fp16/fp32 are. Intentional, not a leftover WA.
        return true;
    }
    if constexpr (std::is_same<T, half>::value) {
        return smallInplace;
    }
    if constexpr (std::is_same<T, float>::value) {
        return smallInplace && narrowChannel;
    }
    return false;
}

template <uint32_t schMode>
__global__ __aicore__ void bias_add(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (schMode == BIAS_ADD_TPL_SCH_MODE_BASE) {
        REGISTER_TILING_DEFAULT(BiasAddTilingData);
        GET_TILING_DATA_WITH_STRUCT(BiasAddTilingData, tilingData, tiling);
        NsBiasAdd::KernelBiasAdd<DTYPE_X> op;
        op.Init(x, bias, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE) {
        REGISTER_TILING_DEFAULT(BiasAddTilingData);
        GET_TILING_DATA_WITH_STRUCT(BiasAddTilingData, tilingData, tiling);
        NsBiasAdd::KernelBiasAddTinyNoQueue<DTYPE_X> op;
        op.Init(x, bias, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST) {
        REGISTER_TILING_DEFAULT(BiasAddTilingData);
        GET_TILING_DATA_WITH_STRUCT(BiasAddTilingData, tilingData, tiling);
        if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
            NsBiasAdd::KernelBiasAddThinTinyNhwcBf16Runtime op;
            op.Init(x, bias, y, &tilingData);
            op.Process();
        } else {
            if (ShouldUseThinSmallRuntimeInplace<DTYPE_X>(tilingData)) {
                NsBiasAdd::KernelBiasAddThinTinyNhwcVectorBroadcast<DTYPE_X> op;
                op.Init(x, bias, y, &tilingData);
                op.Process();
            } else {
                NsBiasAdd::KernelBiasAddThinTinyNhwcVectorBroadcastOutplace<DTYPE_X> op;
                op.Init(x, bias, y, &tilingData);
                op.Process();
            }
        }
    } else if constexpr (schMode == BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE) {
        REGISTER_TILING_DEFAULT(BiasAddTilingData);
        GET_TILING_DATA_WITH_STRUCT(BiasAddTilingData, tilingData, tiling);
        if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
            NsBiasAdd::KernelBiasAddBf16BroadcastUbCastTile op;
            op.Init(x, bias, y, &tilingData);
            op.Process();
        } else if constexpr (std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, float>::value ||
                             std::is_same<DTYPE_X, half>::value) {
            NsBiasAdd::KernelBiasAddBroadcastUbTile<DTYPE_X> op;
            op.Init(x, bias, y, &tilingData);
            op.Process();
        } else {
            // Unsupported DTYPE_X. Host GetShapeAttrsInfo restricts dtype to
            // float/float16/bf16/int32, so this branch is unreachable in valid builds. Fail at
            // compile time (dependent-false static_assert) instead of silently falling back to the
            // generic KernelBiasAdd, which lacks this path's vector-broadcast / bf16-cast handling.
            static_assert(BiasAddAlwaysFalse<schMode>::value,
                          "Unsupported DTYPE_X for BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE");
        }
    }
}
