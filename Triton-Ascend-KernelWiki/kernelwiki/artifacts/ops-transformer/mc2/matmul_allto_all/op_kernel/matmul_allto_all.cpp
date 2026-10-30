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
 * \file matmul_allto_all.cpp
 * \brief kernel内核实现
 */
#include <lib/matmul_intf.h>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch32/matmul_allto_all_tiling_key.h"
#include "arch32/matmul_allto_all_tiling.h"
#include "arch32/matmul_allto_all.h"

using namespace AscendC;
using namespace matmul_allto_all_910b_tiling_key;
using namespace MatmulAlltoAllImpl;

template<bool MM_ALLTO_ALL_TRANS_X2, bool MM_ALLTO_ALL_HAS_BIAS, bool MM_ALLTO_ALL_QUANT_BF16>
__global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, 
                                            GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, 
                                            GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (MM_ALLTO_ALL_QUANT_BF16) {
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, bfloat16_t, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2> op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
        op.Process();
    } else {
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2> op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
        op.Process();
    }
}