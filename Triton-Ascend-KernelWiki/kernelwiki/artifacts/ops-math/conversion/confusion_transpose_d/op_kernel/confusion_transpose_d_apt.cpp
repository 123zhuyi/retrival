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
 * \file confusion_transpose_d.cpp
 * \brief confusion_transpose_d
 */
#include "../transpose/arch35/transpose_tensor_move.h"
#include "../transpose/arch35/transpose_cut_one_axis.h"
#include "../transpose/arch35/transpose_cut_two_axis.h"
#include "../transpose/arch35/transpose_small_shape.h"
#include "../transpose/arch35/transpose_big_dim.h"
#include "../transpose/arch35/transpose_n_last.h"

#define TENSOR_MOVE 10000
#define SMALL_SHAPE 10001
#define CUT_ONCE 10002
#define CUT_TWICE 10003
#define N_LAST_TRANSPOSE 10004
#define BIG_DIM 10005

using namespace Transpose;

__global__ __aicore__ void confusion_transpose_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{   
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TENSOR_MOVE)) {
        Transpose::TransposeTensorMove<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(SMALL_SHAPE)) {
        Transpose::TransposeSmallShape<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling);
        op.Process();
    } else if (TILING_KEY_IS(CUT_ONCE)) {
        Transpose::TransposeCutOneAxis<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(CUT_TWICE)) {
        Transpose::TransposeCutTwoAxis<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(BIG_DIM)) {
        Transpose::TransposeBigDim<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(N_LAST_TRANSPOSE)) {
        Transpose::TransposeNLast<DTYPE_X> op;
        op.Init(x, y, &tilingData.transposeOpTiling, &pipe);
        op.Process();
    }
}
