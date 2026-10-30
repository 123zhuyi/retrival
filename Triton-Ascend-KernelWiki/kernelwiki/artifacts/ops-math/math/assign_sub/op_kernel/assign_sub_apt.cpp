/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/assign_sub_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/assign_sub_tiling_struct.h"

using namespace AssignSubNs;
using namespace Ops::Base;

extern "C" __global__ __aicore__ void assign_sub(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AssignSubTilingData);
    GET_TILING_DATA_WITH_STRUCT(AssignSubTilingData, tilingData, tiling);

    TPipe pipe;

    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch<0UL, AssignSubOp<half>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(102UL)) {
        ElementwiseSch<0UL, AssignSubOp<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(103UL)) {
        ElementwiseSch<0UL, AssignSubOp<float>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(104UL)) {
        ElementwiseSch<0UL, AssignSubOp<int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(105UL)) {
        ElementwiseSch<0UL, AssignSubOp<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(106UL)) {
        ElementwiseSch<0UL, AssignSubOp<int64_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    } else if (TILING_KEY_IS(107UL)) {
        ElementwiseSch<0UL, AssignSubOp<uint8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); // 获取Schedule
        sch.Init(x1, x2, y);
        sch.Process();
    }
    return;
}
