/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>
#include "arch35/batch_to_space_nd_tiling_data.h"
#include "arch35/batch_to_space_nd_tiling_key.h"
#include "arch35/batch_to_space_nd_simt.h"
#include "arch35/batch_to_space_nd_large_c.h"
#include "arch35/batch_to_space_nd_small_c.h"

using namespace AscendC;
using namespace B2SND;

template <bool isBigShape>
__aicore__ inline void BatchToSpaceND4Simt(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_X);
    using DTYPE_X_ = std::conditional_t<
        tSize != b32,
        std::conditional_t<
            tSize == b8, uint8_t,
            std::conditional_t<tSize == b16, uint16_t, std::conditional_t<tSize == b64, uint64_t, DTYPE_X>>>,
        uint32_t>;

    SetSysWorkspace(workspace);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(B2SNDSimtTilingData, tilingData, tiling);
    if constexpr (isBigShape == false) {
        BatchToSpaceNDSIMT<DTYPE_X_, uint32_t> op;
        op.Init(x, block_shape, crops, y, &tilingData, &pipe);
        op.Process(tiling);
    } else if constexpr (isBigShape == true) {
        BatchToSpaceNDSIMT<DTYPE_X_, uint64_t> op;
        op.Init(x, block_shape, crops, y, &tilingData, &pipe);
        op.Process(tiling);
    }
}

__aicore__ inline void BatchToSpaceND4LargeC(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(B2SNDLargeCTilingData, tilingData, tiling);
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        BatchToSpaceLargeC<int8_t> op(&pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        BatchToSpaceLargeC<int16_t> op(&pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        BatchToSpaceLargeC<int32_t> op(&pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        BatchToSpaceLargeC<int64_t> op(&pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

template <uint8_t blockShapeDimNum>
__aicore__ inline void BatchToSpaceND4SmallC(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(B2SNDSmallCTilingData, tilingData, tiling);
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        BatchToSpaceSmallC<uint8_t, blockShapeDimNum> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        BatchToSpaceSmallC<uint16_t, blockShapeDimNum> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        BatchToSpaceSmallC<uint32_t, blockShapeDimNum> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        BatchToSpaceSmallC<uint64_t, blockShapeDimNum> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

template <uint8_t mode, uint8_t blockShapeDimNum, bool IsBigShape>
__global__ __aicore__ void batch_to_space_nd(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if constexpr (mode == TPL_MODE_SIMT) {
        BatchToSpaceND4Simt<IsBigShape>(x, block_shape, crops, y, workspace, tiling);
        return;
    }
    if constexpr (mode == TPL_MODE_LARGE_C) {
        BatchToSpaceND4LargeC(x, block_shape, crops, y, workspace, tiling);
        return;
    }
    if constexpr (mode == TPL_MODE_SMALL_C) {
        BatchToSpaceND4SmallC<blockShapeDimNum>(x, block_shape, crops, y, workspace, tiling);
        return;
    }
}
