/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file right_shift_apt.cpp
 * \brief right_shift_apt
 */

#include "kernel_operator.h"
#include "arch35/right_shift_dag.h"
#include "arch35/right_shift_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace RightShiftOp;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void right_shift(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (dType == TPL_INT8) {
        using OpDag = RightShiftDag8<int8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_UINT8) {
        using OpDag = RightShiftDag8<uint8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_INT16) {
        using OpDag = RightShiftDag16<int16_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_UINT16) {
        using OpDag = RightShiftDag16<uint16_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_INT32) {
        using OpDag = RightShiftDag32<int32_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_UINT32) {
        using OpDag = RightShiftDag32<uint32_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_INT64) {
        using OpDag = RightShiftDag64<int64_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    } else if constexpr (dType == TPL_UINT64) {
        using OpDag = RightShiftDag64<uint64_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, y, z);
    }
}
