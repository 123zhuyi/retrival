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
 * \file moe_token_unpermute_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokensNum);    // tokens 轴大小
TILING_DATA_FIELD_DEF(int64_t, topK);         // topK 轴大小
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);   // hiddenSize 轴大小
TILING_DATA_FIELD_DEF(int64_t, numOutTokens); // input的第0维输入

TILING_DATA_FIELD_DEF(int64_t, formerCoreNum);    // 前core_num - tailCoreNum轴
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);      // 尾轴数
TILING_DATA_FIELD_DEF(int64_t, tokenNumEachCore); // 前core_num - tailCoreNum轴计算的token_num数据量
TILING_DATA_FIELD_DEF(int64_t, tokenNumTailCore); // 后tailCoreNum轴计算的token_num数据量
TILING_DATA_FIELD_DEF(int64_t, rowIdMapEachCore); // 前core_num - tailCoreNum轴计算的sorted_indices数据量
TILING_DATA_FIELD_DEF(int64_t, rowIdMapTailCore); // 后tailCoreNum轴计算的sorted_indices数据量

TILING_DATA_FIELD_DEF(int64_t, hiddenSizeAlign);        // hiddenSize对齐
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeLoopTimes);    // hiddenSize循环次数
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeTail);         // hiddenSize尾块
TILING_DATA_FIELD_DEF(int64_t, inputReserveNum);        // input一次搬入的数量
TILING_DATA_FIELD_DEF(int64_t, indicesReserveNum);      // sorted_indices一次搬入的数量
TILING_DATA_FIELD_DEF(int64_t, indicesReserveNumAlign); // sorted_indices一次搬入的数量对齐

TILING_DATA_FIELD_DEF(int64_t, totalUbSize); // ub空间总大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteGrad, MoeTokenUnpermuteGradTilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_GRAD_TILING_H
