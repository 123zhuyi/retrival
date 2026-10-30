/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rsqrt_grad.cpp
 * \brief z = -0.5 * y**3 * dy
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/rsqrt_grad_dag.h"
#include "arch35/rsqrt_grad_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/rsqrt_grad_tilingdata.h"

using namespace RsqrtGradOp;

namespace AscendC {
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void rsqrt_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RsqrtGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(RsqrtGradTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, RsqrtGradDAG<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, RsqrtGradDAG<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, RsqrtGradDAG<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_INT8) {
        ElementwiseSch<schMode, RsqrtGradInt8<int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    } else if constexpr (dType == TPL_INT32) {
        ElementwiseSch<schMode, RsqrtGradWithDiv<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y, dy, z);
        sch.Process();
    }
    return;
}
}