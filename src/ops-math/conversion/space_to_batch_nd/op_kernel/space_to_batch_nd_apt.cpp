/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/space_to_batch_nd.h"
#include "arch35/space_to_batch_nd_tiling_data.h"
#include "arch35/space_to_batch_nd_tiling_key.h"

template <int TilingKey>
__aicore__ inline void SpaceToBatchNDDispatch(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(SpaceToBatchNDTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        SpaceToBatchND<int8_t, TilingKey> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        SpaceToBatchND<int16_t, TilingKey> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        SpaceToBatchND<int32_t, TilingKey> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        SpaceToBatchND<int64_t, TilingKey> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

template <int TilingKey>
__global__ __aicore__ void space_to_batch_nd(GM_ADDR x, GM_ADDR block_shape, GM_ADDR paddings, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpaceToBatchNDTilingData);
    SpaceToBatchNDDispatch<TilingKey>(x, y, tiling);
}
