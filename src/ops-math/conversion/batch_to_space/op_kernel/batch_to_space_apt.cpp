/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/batch_to_space.h"
#include "arch35/batch_to_space_tiling_data.h"
#include "arch35/batch_to_space_tiling_key.h"

using namespace NsBatchToSpace;

template <uint8_t UbAxis>
__aicore__ inline void BatchToSpaceDispatch(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        BatchToSpace<int8_t, UbAxis> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        BatchToSpace<int16_t, UbAxis> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        BatchToSpace<int32_t, UbAxis> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        BatchToSpace<int64_t, UbAxis> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

template <uint8_t UbAxis>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR crops, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    BatchToSpaceDispatch<UbAxis>(x, y, tiling);
}
