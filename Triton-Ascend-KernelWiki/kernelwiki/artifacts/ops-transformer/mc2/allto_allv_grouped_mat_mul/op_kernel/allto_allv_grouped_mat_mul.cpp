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
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "allto_allv_grouped_mat_mul_coarse_grained.h"
#include "allto_allv_grouped_mat_mul_tiling_key.h"

using namespace AscendC;

#define INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL()                                                                 \
    do {                                                                                                          \
        op.Init(                                                                                                  \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,           \
            mmweightOptionalGM, gmmyGM, mmyOptionalGM, permuteOutOptionalGM, workspaceGM, contextGM, &tilingData, \
            hcclInitTiling, alltoAllvCcTiling, &pipe);                                                            \
        op.Process();                                                                                             \
    } while (0)

template <
    int D_T_MM, bool TILINGKEY_MM, bool TILINGKEY_GMM_WEIGHT_TRANSPOSE, 
    bool TILINGKEY_MM_WEIGHT_TRANSPOSE>
__global__ __aicore__ void allto_allv_grouped_mat_mul(GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR gmmxOffsetGM,
    GM_ADDR gmmWOffsetGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR mmxOffsetGM, GM_ADDR mmWOffsetGM,
    GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    auto tiling = (__gm__ AlltoAllvGmmTilingData*)tilingGM;
    __gm__ void* hcclInitTiling = (__gm__ void*)(&(tiling->hcclInitTiling));
    __gm__ void* alltoAllvCcTiling = (__gm__ void*)(&(tiling->alltoAllvCcTiling));
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

if (D_T_MM == ADD_TPL_BP16) {
    AlltoAllvGmmCoarseGrained<bfloat16_t, TILINGKEY_MM, TILINGKEY_GMM_WEIGHT_TRANSPOSE,
                                TILINGKEY_MM_WEIGHT_TRANSPOSE> op;
    INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
    return;
}

if (D_T_MM == ADD_TPL_FP16) {
    AlltoAllvGmmCoarseGrained<half, TILINGKEY_MM, TILINGKEY_GMM_WEIGHT_TRANSPOSE,
                                TILINGKEY_MM_WEIGHT_TRANSPOSE> op;
    INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
    return;
}
}