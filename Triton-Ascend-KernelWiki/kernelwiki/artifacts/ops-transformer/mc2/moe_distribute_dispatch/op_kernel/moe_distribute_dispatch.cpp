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
 * \file moe_distribute_dispatch.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_dispatch_tiling_key.h"

#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_dispatch_tiling.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_tiling.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_a2.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_a2_layered.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_a2_layered_aicpu.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2_layered.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2_layered_aicpu.h"
#endif
#include "moe_distribute_dispatch.h"

using namespace MoeDistributeDispatchImpl;
using namespace MoeDistributeDispatchA2Impl;

using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch( 
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchA2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeDistributeDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", MoeDistributeDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeDispatchA2TilingData);

    TPipe pipe;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, HasTp> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
            MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, HasTp> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
            MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, ScaleMode, HasTp> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }
#endif
#else
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, HasTp> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }

    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE)) { 
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, nullptr, expandXOut, dynamicScalesOut, expandIdxOut, 
                expertTokenNumsOut, epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();

    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, nullptr, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
            MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, HasTp> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
            MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, ScaleMode, HasTp> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }

    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE) && (ScaleMode == false)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, nullptr, expandXOut, dynamicScalesOut, expandIdxOut, 
                expertTokenNumsOut, epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE) && (ScaleMode == true)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, nullptr, expandXOut, dynamicScalesOut, expandIdxOut, 
                expertTokenNumsOut, epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (CommMode == TILINGKEY_TPL_AICPU) && (ScaleMode == false)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, nullptr, expandXOut, dynamicScalesOut, expandIdxOut,
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU) && (ScaleMode == true)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, nullptr, expandXOut, dynamicScalesOut, expandIdxOut, 
                    expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#endif
#endif
}