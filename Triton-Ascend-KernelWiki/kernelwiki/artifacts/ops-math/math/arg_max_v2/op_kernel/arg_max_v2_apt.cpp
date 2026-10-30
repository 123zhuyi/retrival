/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_v2_apt.cpp
 * \brief arg_max_v2_apt
 */
#define TILING_KEY_AR 10001
#define TILING_KEY_ARA 10002
#define TILING_KEY_COPY 10003
#define TILING_KEY_ARA_CUT_A_AND_NEXT_A 10012
#define TILING_KEY_AR_GATHER 20001
#define TILING_KEY_ARA_GATHER 20002
#define TILING_KEY_RA 20003
#define TILING_KEY_GROUP_REDUCE 30001

#include "../arg_max_with_value/arch35/arg_max_with_value_ar.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_ara.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_copy.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_ar_gather.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_ara_gather.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_ra.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_group_reduce.h"
#include "../arg_max_with_value/arch35/arg_max_with_value_ara_cut_a_and_next_a.h"

using namespace ArgMaxWithValue;

extern "C" __global__ __aicore__ void arg_max_v2(GM_ADDR x, GM_ADDR dimension, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); 
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if constexpr(ORIG_DTYPE_X == DT_INT64) {
        if (TILING_KEY_IS(TILING_KEY_AR)) {
            ArgMaxWithValue::ArgMaxWithValueAr<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_ARA)) {
            ArgMaxWithValue::ArgMaxWithValueArA<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_COPY)) {
            ArgMaxWithValue::ArgMaxWithValueCopy<DTYPE_X, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if(TILING_KEY_IS(TILING_KEY_ARA_CUT_A_AND_NEXT_A)){
            ArgMaxWithValue::ArgMaxWithValueArACutAAndNextA<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_RA)) {
            ArgMaxWithValue::ArgMaxWithValueRa<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_ARA_GATHER)) {
            ArgMaxWithValue::ArgMaxWithValueAraGather<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_AR_GATHER)) {
            ArgMaxWithValue::ArgMaxWithValueArGather<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_GROUP_REDUCE)) {
            ArgMaxWithValue::ArgMaxWithValueGroupReduce<DTYPE_X, int64_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        }
    } else {
        if (TILING_KEY_IS(TILING_KEY_AR)) {
            ArgMaxWithValue::ArgMaxWithValueAr<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_ARA)) {
            ArgMaxWithValue::ArgMaxWithValueArA<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_COPY)) {
            ArgMaxWithValue::ArgMaxWithValueCopy<DTYPE_X, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if(TILING_KEY_IS(TILING_KEY_ARA_CUT_A_AND_NEXT_A)){
            ArgMaxWithValue::ArgMaxWithValueArACutAAndNextA<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_AR_GATHER)) {
            ArgMaxWithValue::ArgMaxWithValueArGather<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_RA)) {
            ArgMaxWithValue::ArgMaxWithValueRa<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_ARA_GATHER)) {
            ArgMaxWithValue::ArgMaxWithValueAraGather<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_GROUP_REDUCE)) {
            ArgMaxWithValue::ArgMaxWithValueGroupReduce<DTYPE_X, int32_t, DTYPE_Y, false, false> op;
            op.Init(x, y, dimension, workspace, &tilingData, &pipe);
            op.Process();
        }
    }
}