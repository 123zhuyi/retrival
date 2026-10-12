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
 * \file moe_distribute_dispatch_v3.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_full_mesh.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling_key.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#endif

using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v3(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        } else if constexpr (FullMesh == TILINGKEY_NO_FULLMESH) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, HasTp> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
            if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, false, false> op;
                op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, false> op;
                op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            }
        } else if constexpr (FullMesh == TILINGKEY_NO_FULLMESH) {
            if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, false, HasTp> op;
                op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, HasTp> op;
                op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            }
        }
    }
#endif
}