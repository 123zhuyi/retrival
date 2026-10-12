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
 * \file moe_finalize_routing.cpp
 * \brief
 */
#include "moe_finalize_routing_fp_db.h"
#include "moe_finalize_routing_bf16.h"
#include "moe_finalize_routing_fp_cuth.h"
#include "moe_finalize_routing_fp_cuth_k2.h"
#include "moe_finalize_routing_fp_cuth_k4.h"
#include "moe_finalize_routing_fp_cutk.h"
#include "moe_finalize_routing_bf16_cutk.h"
#include "moe_finalize_routing_bf16_cuth.h"
#include "moe_finalize_routing_bf16_cuth_k2.h"
#include "moe_finalize_routing_bf16_cuth_k4.h"
#include "moe_finalize_routing_fp_db_all_bias.h"
#include "moe_finalize_routing_bf16_all_bias.h"
#include "moe_finalize_routing_fp_cuth_k2_optimized.h"

using namespace MoeFinalizeRouting;

extern "C" __global__ __aicore__ void moe_finalize_routing(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2,
                                                           GM_ADDR bias, GM_ADDR scales, GM_ADDR expandedSrcToDstRow,
                                                           GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(20000)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCutK<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20001)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCutK<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20002)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(20003)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        MoeFinalizeRouting::MoeFinalizeRoutingBF16<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(20004)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDb<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20005)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDb<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20006)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20007)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20008)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CuthK2<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20009)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK4<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20010)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK4<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20011)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CuthK4<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20012)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuth<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20013)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuth<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20014)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16Cuth<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20015)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDbAllBias<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20016)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDbAllBias<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20017)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBF16AllBias<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20018)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2Optimized<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    }
}
