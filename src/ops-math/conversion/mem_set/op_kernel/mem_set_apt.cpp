/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mem_set.cpp
 * \brief mem_set kernel
 */

#include "kernel_operator.h"
#include "arch35/mem_set.h"

using namespace AscendC;
using namespace MemSetSpc;

template <uint16_t inputCount>
__global__ __aicore__ void mem_set(GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR* param_base = (GM_ADDR*)get_para_base();
    REGISTER_TILING_DEFAULT(MemSetTilingData<256>);
    // 256为最大支持tensor数量
    if constexpr (inputCount <= 16) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<inputCount>, tilingData, param_base[inputCount]);
        MemSet<inputCount> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    } else if constexpr (inputCount <= 32) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<32>, tilingData, param_base[inputCount]);
        MemSet<32> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    } else if constexpr (inputCount <= 64) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<64>, tilingData, param_base[inputCount]);
        MemSet<64> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    } else if constexpr (inputCount <= 128) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<128>, tilingData, param_base[inputCount]);
        MemSet<128> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    } else if constexpr (inputCount <= 192) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<192>, tilingData, param_base[inputCount]);
        MemSet<192> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    } else if constexpr (inputCount <= 256) {
        GET_TILING_DATA_WITH_STRUCT(MemSetTilingData<256>, tilingData, param_base[inputCount]);
        MemSet<256> op(tilingData, pipe, param_base);
        op.Init();
        op.Process();
    }
}