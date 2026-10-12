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
 * \file logical_or.cpp
 * \brief logical_or kernel
 */

#include "arch35/logical_or_dag.h"
#include "kernel_operator.h"
#include "arch35/logical_or_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void logical_or(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, bool>::value) {
        using OpDag = LogicalOrOp::LogicalOrCompute<uint8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrIntegralCompute<int8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrIntegralCompute<uint8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int16_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrIntegralCompute<int16_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int32_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrIntegralCompute<int32_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int64_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrIntegralCompute<int64_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, half>::value) {
        using OpDag = LogicalOrOp::LogicalOrFloatCompute<half, uint16_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
        using OpDag = LogicalOrOp::LogicalOrFloatCompute<bfloat16_t, uint16_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, float>::value) {
        using OpDag = LogicalOrOp::LogicalOrFloatCompute<float, uint32_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}