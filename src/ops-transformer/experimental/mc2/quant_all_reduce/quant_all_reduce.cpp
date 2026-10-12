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
 * \file quant_all_reduce.cpp
 * \brief
 */
#include <cstdio>
#include <cstring>
#include "kernel_operator.h"
#include "./kernel/quant_all_reduce_tiling_data.h"
#include "./kernel/quant_all_reduce_mte_one_shot.h"

using namespace AscendC;
using namespace QuantAllReduceImpl;

#define MTE_ONE_SHOT 1

template<uint32_t quantAllReduceCommMode, typename DTYPE_X, typename DTYPE_SCALES, typename DTYPE_OUT_PUT>
__attribute__((always_inline)) __aicore__ __inline__ void quant_all_reduce_with_mc2context(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR workspaceGM,
                                                        GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (quantAllReduceCommMode == MTE_ONE_SHOT) {
        QuantAllReduceMteOneShot<DTYPE_X, DTYPE_SCALES, DTYPE_OUT_PUT> op;
        op.InitWithMc2Context(x, scales, output, mc2Context, tilingGM, &pipe);
        op.Process();
    }
}
// 核函数入口实现
extern "C" __global__ __aicore__ void QuantAllReduce_MTE_ONE_SHOT_Generic(
    int8_t type,
    GM_ADDR x, GM_ADDR scales, GM_ADDR output, 
    GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    // 根据不同的数据类型调用不同的模板
    switch (type) {
        case 1: 
            quant_all_reduce_with_mc2context<MTE_ONE_SHOT, int8_t, float32_t, float16_t>(
                x, scales, output, workspaceGM, mc2Context, tilingGM);
            break;
        case 2: 
            quant_all_reduce_with_mc2context<MTE_ONE_SHOT, fp8_e5m2_t, fp8_e8m0_t, bfloat16_t>(
                x, scales, output, workspaceGM, mc2Context, tilingGM);
            break;
        case 3: 
            quant_all_reduce_with_mc2context<MTE_ONE_SHOT, fp8_e5m2_t, fp8_e8m0_t, float16_t>(
                x, scales, output, workspaceGM, mc2Context, tilingGM);
            break;
        default:
            AscendC::PRINTF("QuantAllReduce Error: invalid type = %d\n", type);
            return;
    }
}
// <<<>>>调用函数
void quant_all_reduce_demo(int8_t type, uint32_t blockDim, void* stream, uint8_t* x, uint8_t* scales, uint8_t* output, 
                           uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
    QuantAllReduce_MTE_ONE_SHOT_Generic<<<blockDim, nullptr, stream>>>(
        type, 
        x, scales, output, workspaceGM, mc2Context, tilingGM
    );
}