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
 * \file stateless_randperm_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/stateless_randperm.h"
#include "stateless_randperm_struct.h"
#include "arch35/stateless_randperm_tiling_key.h"


using namespace AscendC;

template <uint64_t randomType, uint64_t nIsInt32, uint64_t schId, uint64_t isInt32, uint64_t isDescend>
__global__ __aicore__ void stateless_randperm(
    GM_ADDR n, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{   
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(StatelessRandpermTilingData);
    GET_TILING_DATA_WITH_STRUCT(StatelessRandpermTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (nIsInt32 == 0) {
        if constexpr (randomType == 0) {
            StatelessRandperm::StatelessRandperm<int64_t, uint8_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else if constexpr (randomType == 1) {
            StatelessRandperm::StatelessRandperm<int64_t, uint16_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else if constexpr (randomType == 2) {
            StatelessRandperm::StatelessRandperm<int64_t, int32_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else {
            StatelessRandperm::StatelessRandperm<int64_t, int64_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        }
    } else {
        if constexpr (randomType == 0) {
            StatelessRandperm::StatelessRandperm<int32_t, uint8_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else if (randomType == 1) {
            StatelessRandperm::StatelessRandperm<int32_t, uint16_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else if (randomType == 2) {
            StatelessRandperm::StatelessRandperm<int32_t, int32_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        } else{
            StatelessRandperm::StatelessRandperm<int32_t, int64_t, DTYPE_Y, schId, isInt32, isDescend> op(&pipe, &tilingData);
            op.Init(n, seed, offset, y, workspace);
            op.Process();
        }
    }
}
 
