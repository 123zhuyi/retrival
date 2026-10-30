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
 * \file transpose.cpp
 * \brief
 */

#include "arch35/transpose_big_dim.h"
#include "arch35/transpose_cut_one_axis.h"
#include "arch35/transpose_cut_two_axis.h"
#include "arch35/transpose_n_last.h"
#include "arch35/transpose_small_shape.h"
#include "arch35/transpose_tensor_move.h"
#include "arch35/transpose_with_gather.h"
#include "arch35/transpose_transdata_5hd.h"
#include "arch35/transpose_transdata_5hd_021.h"

#define TENSOR_MOVE 10000
#define SMALL_SHAPE 10001
#define CUT_ONCE 10002
#define CUT_TWICE 10003
#define N_LAST_TRANSPOSE 10004
#define BIG_DIM 10005
#define GATHER_TRANSPOSE 10006
#define VCONV_TRANSPOSE 10007
#define VCONV_021_TRANSPOSE 10008

using namespace Transpose;

extern "C" __aicore__ inline void TransposeTensorMoveProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeTensorMove<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeTensorMove<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeTensorMove<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeTensorMove<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeTensorMove<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeSmallShapeProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeSmallShape<int8_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeSmallShape<int16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeSmallShape<int32_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeSmallShape<int64_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        Transpose::TransposeSmallShape<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeCutOneAxisProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeCutOneAxis<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeCutOneAxis<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeCutOneAxis<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeCutOneAxis<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeCutOneAxis<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeCutTwoAxisProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeCutTwoAxis<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeCutTwoAxis<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeCutTwoAxis<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeCutTwoAxis<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeCutTwoAxis<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeBigDimProcess(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                         TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeBigDim<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeBigDim<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeBigDim<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeBigDim<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeBigDim<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeNLastProcess(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                        TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeNLast<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeNLast<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeNLast<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeNLast<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeNLast<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeGatherProcess(GM_ADDR x, GM_ADDR y,
                                                         const GatherTransposeTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeWithGather<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeWithGather<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeWithGather<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeWithGather<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void TransposeVconvProcess(GM_ADDR x, GM_ADDR y,
                                                        const TransposeVCONVTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::KernelTransDataTo5HD<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void Transpose021VconvProcess(GM_ADDR x, GM_ADDR y,
                                                           const Transpose021VCONVTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::KernelTransDataTo5HD021<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::KernelTransDataTo5HD021<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::KernelTransDataTo5HD021<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TENSOR_MOVE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeTensorMoveProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(SMALL_SHAPE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeSmallShapeProcess(x, y, &tilingData.transposeOpTiling);
    } else if (TILING_KEY_IS(CUT_ONCE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeCutOneAxisProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(CUT_TWICE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeCutTwoAxisProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(N_LAST_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeNLastProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(BIG_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeBigDimProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(GATHER_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(GatherTransposeTilingData, tilingData, tiling);
        TransposeGatherProcess(x, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(VCONV_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(TransposeVCONVTilingData, tilingData, tiling);
        TransposeVconvProcess(x, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(VCONV_021_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(Transpose021VCONVTilingData, tilingData, tiling);
        Transpose021VconvProcess(x, y, &tilingData, &pipe);
    }
}
