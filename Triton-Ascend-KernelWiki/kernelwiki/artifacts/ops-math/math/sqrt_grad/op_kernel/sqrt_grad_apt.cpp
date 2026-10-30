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
 * \file sqrt_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/sqrt_grad_dag.h"
#include "arch35/sqrt_grad_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/sqrt_grad_tiling_struct.h"

using namespace AscendC;
using namespace SqrtGradNs;
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void sqrt_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SqrtGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtGradTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, SqrtGradDag::SqrtGradCustom<half>::OpDag> sch(&(tilingData.basetiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, SqrtGradDag::SqrtGradCustom<bfloat16_t>::OpDag> sch(&(tilingData.basetiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, SqrtGradDag::SqrtGradCustom<float>::OpDag> sch(&(tilingData.basetiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    }
    return;
}