/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to.h
 * \brief expose broadcastTo implementation
 */

#ifndef BROADCAST_TO_H_
#define BROADCAST_TO_H_

#include <type_traits>

#include "broadcast_to_with_copypad.h"
#include "broadcast_to_with_dma.h"
#include "broadcast_to_with_tailAxis.h"
#include "broadcast_to_with_ub.h"

using namespace AscendC;
using namespace BrcTo;

#define TILING_MODE_NDDMA 11000
#define TILING_MODE_UB_BRC 11001
#define TILING_MODE_LAST_DIM_LARGE_A 11002
#define TILING_MODE_LAST_DIM_LARGE_B 11003
#define TILING_MODE_FULL_NDDMA 11004
#define TILING_MODE_LAST_DIM_SMALL_A 11005

// for other operator(ex: tile) to use
__aicore__ void inline broadcast_to_impl(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

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
        DTYPE_X>;
    TPipe pipe;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(TILING_MODE_NDDMA)) {
        BrcToWithNDDMA<DTYPE_X_, BroadcastToTilingData> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_MODE_UB_BRC)) {
        BroadcastToUb<DTYPE_X_, BroadcastToTilingData> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_MODE_LAST_DIM_LARGE_A)) {
        BrcToDataCopyPad<DTYPE_X_, BroadcastToTilingData> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_MODE_LAST_DIM_LARGE_B)) {
        BrcToWithTailAxis<DTYPE_X_, BroadcastToTilingData> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_MODE_FULL_NDDMA)) {
        BrcToWithNDDMA<DTYPE_X_, BroadcastToTilingData, 5U> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_MODE_LAST_DIM_SMALL_A)) {
        BroadcastToUb<DTYPE_X_, BroadcastToTilingData, true> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

#endif // BROADCAST_TO_H_
