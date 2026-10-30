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
 * \file floor_mod.cpp
 * \brief floor_mod kernel
 */

#include "kernel_operator.h"
#include "arch35/floor_mod_dag.h"
#include "arch35/floor_mod_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace Ops::Base;
using namespace AscendC;

template <uint64_t schMode>
__global__ __aicore__ void floor_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value) {
        using OpDag = FloorModOp::FloorModFloatWithCastOp<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, float>::value) {
        using OpDag = FloorModOp::FloorModFloatOp<float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int32_t>::value) {
        using OpDag = FloorModOp::FloorModInt32Op<int32_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int64_t>::value) {
        using OpDag = FloorModOp::FloorModInt64Op<int64_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}
