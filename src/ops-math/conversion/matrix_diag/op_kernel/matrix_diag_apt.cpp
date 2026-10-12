/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_diag_apt.cpp
 * \brief MatrixDiag implementation
 */

#include <type_traits>
#include "arch35/matrix_diag_simt.h"
#include "arch35/matrix_diag_tensor_move.h"
#include "arch35/matrix_diag_tiling_data_struct.h"
#include "arch35/matrix_diag_scatter_lower.h"
#include "arch35/matrix_diag_scatter_high_slice_yUB.h"
#include "arch35/matrix_diag_scatter_high_slice_batch.h"

using namespace AscendC;
using namespace MatrixDiagAsc;

extern "C" __aicore__ inline void matrixdiagSliceYUb(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
{
    GET_TILING_DATA_WITH_STRUCT(MatrixDiagScatterLargeTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
        MatrixDiagSliceYUb<uint8_t, uint16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
        MatrixDiagSliceYUb<uint16_t, uint16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
        MatrixDiagSliceYUb<uint32_t, uint32_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
        MatrixDiagSliceYUb<uint64_t, uint64_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void matrixdiagSliceBatch(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
{
    GET_TILING_DATA_WITH_STRUCT(MatrixDiagScatterTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
        MatrixDiagSliceBatch<uint8_t, uint16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
        MatrixDiagSliceBatch<uint16_t, uint16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
        MatrixDiagSliceBatch<uint32_t, uint32_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
        MatrixDiagSliceBatch<uint64_t, uint64_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void matrixdiagScatterLow(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
{
    GET_TILING_DATA_WITH_STRUCT(MatrixDiagScatterTilingData, tilingData, tiling);

    if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
        MatrixDiagScatterLower<uint8_t, uint16_t, int16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
        MatrixDiagScatterLower<uint16_t, uint16_t, int16_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
        MatrixDiagScatterLower<uint32_t, uint32_t, int32_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
        MatrixDiagScatterLower<uint64_t, uint64_t, int64_t> op;
        op.Init(x, y, &tilingData, pipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void matrix_diag(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_X);
    using DTYPE_X_ =
        std::conditional_t<tSize != b32,
                           std::conditional_t<tSize == b8, uint8_t,
                                              std::conditional_t<tSize == b16, uint16_t,
                                                                 std::conditional_t<tSize == b64, uint64_t, DTYPE_X>>>,
                           uint32_t>;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MatrixDiagSimtTilingData);

    SetSysWorkspace(workspace);

    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(MatrixDiagSimtTilingData, tilingData, tiling);
        MatrixDiagSIMT<DTYPE_X_> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(TILING_KEY_PURE_COPY)) {
        GET_TILING_DATA_WITH_STRUCT(MatrixDiagPureCopyTilingData, tilingData, tiling);
        MatrixDiagTensorMove<DTYPE_X_> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(TILING_KEY_SCATTER_LOW)){
        matrixdiagScatterLow(x, y, tiling, &pipe);
    }
    else if (TILING_KEY_IS(TILING_KEY_SCATTER_HIGH)) {
        matrixdiagSliceBatch(x, y, tiling, &pipe);
    }
    else if (TILING_KEY_IS(TILING_KEY_SCATTER_LARGE)) {
        matrixdiagSliceYUb(x, y, tiling, &pipe);
    }
}
