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
 * \file attention_to_ffn.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "attention_to_ffn.h"
#include "attention_to_ffn_tiling.h"
#include "attention_to_ffn_tiling_key.h"

using namespace AscendC;
using namespace AttentionToFFNImpl;
using namespace Mc2Tiling;

template<bool isQuant, bool isSync, bool isActiveMask, uint8_t ArchTag>
__global__ __aicore__ void attention_to_ffn(GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId,
    GM_ADDR layerId, GM_ADDR expertIds, GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttentionToFFNTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", AttentionToFFNTilingData);
    TPipe pipe;
    
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(AttentionToFFNTilingData, tilingData, tilingGM);
        AttentionToFFN<DTYPE_X, isQuant, isSync, isActiveMask> op;
        op.Init(x, sessionId, microBatchId, layerId, expertIds, expertRankTable, scales, active_mask,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}