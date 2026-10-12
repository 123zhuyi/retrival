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
 * \file square.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/square_dag.h"
#include "atvoss/elewise/elewise_sch_16b.h"
#include "atvoss/util/dfx.h"

using namespace Ops::Base;

extern "C" __global__ __aicore__ void square(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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
    REGISTER_TILING_DEFAULT(EleBaseTilingData16B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData16B, tilingData, tiling);
    if (TILING_KEY_IS(1UL)) {
        ElementwiseSch16B<0UL, SquareOp<half>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(2UL)) {
        ElementwiseSch16B<0UL, SquareOp<bfloat16_t>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(3UL)) {
        ElementwiseSch16B<0UL, SquareOp<float>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(4UL)) {
        ElementwiseSch16B<0UL, SquareOp<int32_t>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(5UL)) {
        ElementwiseSch16B<0UL, SquareOp<int64_t>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
        return;
    }
    return;
}
