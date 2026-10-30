/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file abs.cpp
 * \brief y =|x|
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/abs_dag.h"
#include "arch35/abs_complex_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "abs_struct.h"

using namespace AscendC;
using namespace AbsNs;
using namespace AbsOp;

extern "C" __global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch<0UL, AbsDag<bfloat16_t, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(102UL)) {
        if constexpr (std::is_same<DTYPE_X, half>::value) {
            ElementwiseSch<0UL, AbsDag<half, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, float>::value) {
            ElementwiseSch<0UL, AbsDag<float, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int8_t>::value) {
            ElementwiseSch<0UL, AbsDag<int8_t, int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int16_t>::value) {
            ElementwiseSch<0UL, AbsDag<int16_t, int16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int32_t>::value) {
            ElementwiseSch<0UL, AbsDag<int32_t, int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int64_t>::value) {
            ElementwiseSch<0UL, AbsDag<int64_t, int64_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        }
    } else if (TILING_KEY_IS(103UL)) {
        if constexpr (std::is_same<DTYPE_X, complex64>::value) {
            ElementwiseSch<0UL, AbscomplexDag<complex64, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, complex32>::value) {
            ElementwiseSch<0UL, AbscomplexDag<complex32, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(x, y);
            sch.Process();
        }
    }
    return;
}