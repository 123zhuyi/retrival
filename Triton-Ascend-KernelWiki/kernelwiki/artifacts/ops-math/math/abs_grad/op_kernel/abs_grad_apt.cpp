/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file abs_grad_apt.cpp
 * \brief z = dy * sign(y)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/abs_grad_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "abs_grad_struct.h"

using namespace AscendC;
using namespace AbsGradNs;
using namespace AbsGradOp;

extern "C" __global__ __aicore__ void abs_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AbsGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsGradTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        if constexpr (std::is_same<DTYPE_Y, half>::value) {
            ElementwiseSch<0UL, AbsGradDag<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(y, dy, z);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_Y, float>::value) {
            ElementwiseSch<0UL, AbsGradDag<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(y, dy, z);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_Y, bfloat16_t>::value) {
            ElementwiseSch<0UL, AbsGradDag<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(y, dy, z);
            sch.Process();
        }
    }
    return;
}
