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
 * \file add_mat_mat_elements.cpp
 * \brief AddMatMatElements Kernel entry (arch35, Ascend950)
 */

#include "arch35/add_mat_mat_elements.h"

#ifdef __CCE_KT_TEST__
// UT path: extern "C", DTYPE_C from -D flag, default RANK
extern "C" __global__ __aicore__ void add_mat_mat_elements(GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta, GM_ADDR alpha,
                                                           GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AddMatMatElementsTilingData, tilingData, tiling);
    NsAddMatMatElements::AddMatMatElementsKernel<DTYPE_C, ADD_MAT_MAT_ELEMENTS_RANK_MAX> op;
    op.Init(c, a, b, beta, alpha, cOut, tilingData);
    op.Process();
}
#else
// Kernel path: template<typename D_T, int RANK> for TPL instantiation
template <typename D_T, int RANK>
__global__ __aicore__ void add_mat_mat_elements(GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta, GM_ADDR alpha,
                                                GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddMatMatElementsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddMatMatElementsTilingData, tilingData, tiling);
    NsAddMatMatElements::AddMatMatElementsKernel<D_T, RANK> op;
    op.Init(c, a, b, beta, alpha, cOut, tilingData);
    op.Process();
}
#endif
