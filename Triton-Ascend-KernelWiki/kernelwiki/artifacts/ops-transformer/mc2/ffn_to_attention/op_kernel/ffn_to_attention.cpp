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
* \file ffn_to_attention.cpp
* \brief
*/

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "ffn_to_attention.h"
#include "ffn_to_attention_tiling.h"
#include "ffn_to_attention_tilling_key.h"

using namespace AscendC;
using namespace FFNToAttentionImpl;
using namespace Mc2Tiling;

/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：是否输入attnRankTable:
*     0: 不输入, 1: 输入
* 第2位（十位）：无实际意义:
* 第3位（百位）：无实际意义:
* 第4位（千位）：无实际意义:
* 第5位（万位）：无实际意义
*/

template<bool RankTableMode, uint8_t ArchTag>  
__global__ __aicore__ void ffn_to_attention(GM_ADDR x, GM_ADDR sessionIds,
    GM_ADDR microBatchIds, GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum,
    GM_ADDR attnRankTable, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(FFNToAttentionTilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(FFNToAttentionTilingData, tilingData, tilingGM);
        FFNToAttention<DTYPE_X, RankTableMode> op;
        op.Init(x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
                attnRankTable, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}