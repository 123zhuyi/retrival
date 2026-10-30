/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "diag_part.h"

template <uint32_t schMode, uint32_t dtype>
__global__ __aicore__ void diag_part(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DiagPartTilingData);
    GET_TILING_DATA_WITH_STRUCT(DiagPartTilingData, tilingData, tiling);

    if constexpr (schMode == 0) {
        if constexpr (dtype == DIAG_PART_TPL_DTYPE_FLOAT16) {
            NsDiagPart::KernelDiagPart<half> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else if constexpr (dtype == DIAG_PART_TPL_DTYPE_FLOAT) {
            NsDiagPart::KernelDiagPart<float> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else if constexpr (dtype == DIAG_PART_TPL_DTYPE_INT32) {
            NsDiagPart::KernelDiagPart<int32_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
    }
}