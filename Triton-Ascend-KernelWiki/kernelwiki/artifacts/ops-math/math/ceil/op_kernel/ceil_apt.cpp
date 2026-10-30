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
 * \file ceil.cpp
 * \brief z = ceil(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/ceil_dag.h"
#include "arch35/ceil_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "arch35/ceil_tiling_struct.h"

using namespace AscendC;
using namespace Ops::Base;
using namespace CeilNs;
using namespace CeilOp;
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void ceil(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CeilTilingData);
    GET_TILING_DATA_WITH_STRUCT(CeilTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, CeilDAG<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(input_x, output_y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, CeilDAG<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(input_x, output_y);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, CeilDAG<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(input_x, output_y);
        sch.Process();
    }
    return;
}