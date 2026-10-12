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
 * \file muls_apt.cpp
 * \brief muls kernel
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/muls_dag.h"
#include "arch35/muls_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;

__global__ __aicore__ void muls(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MulsTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if constexpr (std::is_same<DTYPE_X, int16_t>::value) {
        ElementwiseSch<0UL, MulsDag::MulsInt16Op::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, int64_t>::value) {
        ElementwiseSch<0UL, MulsDag::MulsOp<DTYPE_X, MulsDag::CAST_MODE_ROUND, MulsDag::CAST_MODE_ROUND>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, int32_t>::value) {
        ElementwiseSch<0UL, MulsDag::MulsOp<DTYPE_X, MulsDag::CAST_MODE_ROUND, MulsDag::CAST_MODE_TRUNC>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, float>::value || std::is_same<DTYPE_X, half>::value ||
                         std::is_same<DTYPE_X, bfloat16_t>::value) {
        ElementwiseSch<0UL, MulsDag::MulsOp<DTYPE_X, MulsDag::CAST_MODE_NONE, MulsDag::CAST_MODE_RINT>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, complex32>::value) {
        ElementwiseSch<0UL, MulsDag::MulsComplex32Op<complex32, complex64>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, complex64>::value) {
        ElementwiseSch<0UL, MulsDag::MulsComplex64Op<complex64>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.SetVar<float, 0>(tilingData.value);
        sch.Init(x, y);
        sch.Process();
    }
}
