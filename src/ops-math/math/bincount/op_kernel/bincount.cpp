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
 * \file bincount.cpp
 * \brief The kernel function of bincount and the entrance for using bincount.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/bincount_simt_base.h"
#include "arch35/bincount_simt_full_load.h"
#include "arch35/bincount_simt_not_full_load_ub.h"
#include "arch35/bincount_simt_not_full_load_gm.h"
#include "arch35/bincount_determine.h"
#include "arch35/bincount_tiling_key.h"
#include "arch35/bincount_tiling_data.h"

using namespace AscendC;

#define TPL_SCH_ID_FULL_LOAD 0     // full load to UB
#define TPL_SCH_ID_BATCH_LOAD 1    // batch load to UB
#define TPL_SCH_ID_NOT_FULL_LOAD 2 // not load to UB
#define TPL_SCH_ID_DETERMIN 3      // determin

#define TPL_OUTPUT_DTYPE_FLOAT 0
#define TPL_OUTPUT_DTYPE_INT32 1
#define TPL_OUTPUT_DTYPE_INT64 2

#define TPL_EMPTY_WEIGHT 0
#define TPL_HAS_WEIGHT 1

template <uint64_t schId, uint64_t outputDtype, uint64_t isWeight>
__global__ __aicore__ void bincount(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    // Init the TPipe to manage the memory. The TPipe init not in the class will be good for compiler (constant folding)
    // and improve performance.
    AscendC::TPipe tPipe;

    // Get tiling data in local memory from 'tiling' of the pointer in global memory and get the pointer of
    // 'tilingDataIn' in local memory. The pointer of 'tilingDataIn' will be set restriced to imporve the performance.
    // Of course, the restrictd pointer will covered when you use wrong. if you use the default tiling struct, you can
    // use the define of GET_TILING_DATA(tilingData, tiling).
    REGISTER_TILING_DEFAULT(BincountTilingData);
    GET_TILING_DATA_WITH_STRUCT(BincountTilingData, tilingDataIn, tiling);
    const BincountTilingData* __restrict tilingData = &tilingDataIn;

    bool isWeightEmpty = isWeight == TPL_EMPTY_WEIGHT ? true : false;

    // The weight is not empty and type is float and the input can be loaded in UB.
    if (schId == TPL_SCH_ID_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
        BincountSimt::BincountSimtFullLoad<float> op;
        op.Init(array, size, weights, bins, tilingData, &tPipe, isWeightEmpty);
        op.Process();
    }
    // The weight is not empty and type is float and the input can not be loaded in UB.
    if (schId == TPL_SCH_ID_BATCH_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
        BincountSimt::BincountSimtNotFullLoadUb<float> op;
        op.Init(array, size, weights, bins, tilingData, &tPipe, isWeightEmpty);
        op.Process();
    }
    // The weight is not empty and type is float and the input will be loaded in GM.
    if (schId == TPL_SCH_ID_NOT_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
        BincountSimt::BincountSimtNotFullLoadGm<float> op;
        op.Init(array, size, weights, bins, tilingData, &tPipe, isWeightEmpty);
        op.Process();
    }
    // determine realiztion
    if (schId == TPL_SCH_ID_DETERMIN && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
        BincountSimt::BincountDetermine<float> op;
        op.Init(array, size, weights, bins, tilingData, &tPipe, isWeightEmpty);
        op.Process();
    }
}