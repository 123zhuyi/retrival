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
 * \file fills.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/fills_dag.h"
#include "arch35/fills_tiling_key.h"
#include "arch35/fills_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void fills(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FillsTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    ElementwiseSch<schMode, FillsDAG<DTYPE_X>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.template SetVar<DTYPE_X, 0>(*(DTYPE_X*)(&tilingData.value));
    sch.Init(y);
    sch.Process();
}