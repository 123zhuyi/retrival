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
 * \file axpy_v2_apt.cpp
 * \brief axpy_v2 kernel
 */

#include "kernel_operator.h"
#include "arch35/axpy_v2_dag.h"
#include "arch35/axpy_v2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t Mode, typename T>
__global__ __aicore__ void axpy_v2_op(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    constexpr bool isFp16Bf16 = (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value);
    constexpr bool isInt32Int64 = (std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value);
    if constexpr (isFp16Bf16) {
        using OpDag =
            typename AxpyV2Op::AxpyV2ComputeFp16Bf16<T, AxpyV2Op::CAST_NONE_MODE, AxpyV2Op::CAST_RINT_MODE>::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    } else if constexpr (std::is_same<T, float>::value) {
        using OpDag = typename AxpyV2Op::AxpyV2ComputeFloat::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    } else if constexpr (isInt32Int64) {
        using OpDag = typename AxpyV2Op::AxpyV2ComputeInt32Int64<T>::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    } else if constexpr (std::is_same<T, int8_t>::value) {
        using OpDag = typename AxpyV2Op::AxpyV2ComputeUint8Int8<T, int32_t>::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        using OpDag = typename AxpyV2Op::AxpyV2ComputeUint8Int8<T, uint32_t>::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    } else if constexpr (std::is_same<T, bool>::value) {
        using OpDag = typename AxpyV2Op::AxpyV2ComputeBool::OpDag;
        BroadcastSch<Mode, OpDag> sch(tiling);
        sch.Process(x1, x2, alpha, y);
    }
}

template <uint64_t schMode>
__global__ __aicore__ void axpy_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    axpy_v2_op<schMode, DTYPE_X1>(x1, x2, alpha, y, workspace, tiling);
}