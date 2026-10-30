/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_tiling_key.h"

#include "moe_distribute_combine.h"
#if __has_include("../moe_distribute_combine_v2/moe_distribute_combine_tiling.h")
#include "../moe_distribute_combine_v2/moe_distribute_combine_tiling.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_a2.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_a2_layered.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_a2_layered_aicpu.h"
#else
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_a2.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_a2_layered.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_a2_layered_aicpu.h"
#endif
using namespace MoeDistributeCombineImpl;
using namespace MoeDistributeCombineA2Impl;

using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                GM_ADDR xActiveMask, GM_ADDR activationScale,
                                                GM_ADDR weightScale, GM_ADDR groupList,
                                                GM_ADDR expandScales, GM_ADDR XOut, GM_ADDR workspaceGM,
                                                GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineA2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeDistributeCombineTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", MoeDistributeCombineTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeCombineA2TilingData);

    TPipe pipe;
    
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
        MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#else
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
        MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }

    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                (LayeredMode == TILINGKEY_TPL_MTE) && (QuantMode == TILINGKEY_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, scales, xActiveMask,
          nullptr, nullptr, nullptr, nullptr, nullptr, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_NO_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeCombineA2LayeredAicpu<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
              contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
            op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, nullptr, 
                    XOut, workspaceGM, &pipe, &tilingData,
              contextGM0);
            op.Process();
        }
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_INT8_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
            if (dataplaneMode == DataplaneMode::AIV) {
                MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
                op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, nullptr, 
                        XOut, workspaceGM, &pipe, &tilingData,
                  contextGM0);
                op.Process();
            }
    }
#endif
#endif
}