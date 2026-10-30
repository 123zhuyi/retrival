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
 * \file grouped_mat_mul_allto_allv.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "grouped_mat_mul_allto_allv.h"
#include "grouped_mat_mul_allto_allv_tiling_key.h"

using namespace AscendC;

template <typename X_T, const bool IS_OPT_MM, const bool IS_GMM_WEIGHT_TRANS, const bool IS_OPT_WEIGHT_TRANS>
struct GMMATAVType { // Grouped_Mat_Mul_All_To_Allv_Type
    using xType = X_T;
    static constexpr bool isOptionalMm = IS_OPT_MM;
    static constexpr bool isGmmWeightTrans = IS_GMM_WEIGHT_TRANS;
    static constexpr bool isOptWeightTrans = IS_OPT_WEIGHT_TRANS;
};

#define INVOKE_GMMATAV_OP_IMPL_A5(templateClass, ...)                                                   \
    do {                                                                                                \
        TPipe pipe;                                                                                     \
        templateClass<GMMATAVType<__VA_ARGS__>> op;                                                     \
        op.Init(                                                                                        \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM, \
            mmweightOptionalGM, yGM, mmyOptionalGM, workspaceGM, contextGM, &tilingData, &pipe);        \
        op.Process();                                                                                   \
    } while (0)

#define INVOKE_GMMATAV_OP_IMPL(templateClass, ...)                                                       \
    do {                                                                                                 \
        TPipe pipe;                                                                                      \
        templateClass<GMMATAVType<__VA_ARGS__>> op;                                                      \
        op.Init(                                                                                         \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,  \
            mmweightOptionalGM, yGM, mmyOptionalGM, workspaceGM, contextGM, &tilingData, hcclInitTiling, \
            alltoAllvCcTiling, &pipe);                                                                   \
        op.Process();                                                                                    \
    } while (0)

template <
    bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GROUPED_MATMUL_TRANS, 
    bool TILINGKEY_MATMUL_TRANS>
__global__ __aicore__ void grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR yGM, GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspaceGM == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(GroupedMatMulAlltoAllvTilingData); 
    auto tiling = (__gm__ GroupedMatMulAlltoAllvTilingData*)tilingGM;
    __gm__ void* hcclInitTiling = (__gm__ void*)(&(tiling->hcclInitTiling));
    __gm__ void* alltoAllvCcTiling = (__gm__ void*)(&(tiling->alltoAllvCcTiling));
    GET_TILING_DATA(tilingData, tilingGM);
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

#if (ORIG_DTYPE_GMM_X == DT_FLOAT16)
    using X_TYPE = half;
    INVOKE_GMMATAV_OP_IMPL(GroupedMatmulAlltoAllv, X_TYPE, TILINGKEY_COMPUTE_MATMUL,
                            TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS);
#endif

#if (ORIG_DTYPE_GMM_X == DT_BF16)
    using X_TYPE = bfloat16_t;
    INVOKE_GMMATAV_OP_IMPL(GroupedMatmulAlltoAllv, X_TYPE, TILINGKEY_COMPUTE_MATMUL,
                            TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS);
#endif
}