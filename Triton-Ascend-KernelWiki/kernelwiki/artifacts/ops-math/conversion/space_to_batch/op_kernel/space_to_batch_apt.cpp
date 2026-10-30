/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/space_to_batch_tiling_data.h"
#include "arch35/space_to_batch_tiling_key.h"
#include "arch35/space_to_batch.h"

using namespace AscendC;
using namespace STB;

template <int UbAxis>
__aicore__ inline void SpaceToBatchDispatch(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(SpaceToBatchTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        SpaceToBatchKernel<int8_t, UbAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        SpaceToBatchKernel<int16_t, UbAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        SpaceToBatchKernel<int32_t, UbAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        SpaceToBatchKernel<int64_t, UbAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

template <int UbAxis>
__global__ __aicore__ void space_to_batch(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if constexpr (UbAxis == TPL_UB_AXIS_N) {
        SpaceToBatchDispatch<TPL_UB_AXIS_N>(x, y, tiling);
    } else if constexpr (UbAxis == TPL_UB_AXIS_H) {
        SpaceToBatchDispatch<TPL_UB_AXIS_H>(x, y, tiling);
    } else if constexpr (UbAxis == TPL_UB_AXIS_W) {
        SpaceToBatchDispatch<TPL_UB_AXIS_W>(x, y, tiling);
    } else if constexpr (UbAxis == TPL_UB_AXIS_C) {
        SpaceToBatchDispatch<TPL_UB_AXIS_C>(x, y, tiling);
    }
}
