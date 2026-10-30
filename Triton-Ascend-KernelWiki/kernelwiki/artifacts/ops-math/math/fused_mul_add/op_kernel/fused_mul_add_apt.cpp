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
 * \file fused_mul_add_apt.cpp
 * \brief fused_mul_add kernel: y = x1 * x2 + x3, element-wise with broadcast
 */

#include "kernel_operator.h"
#include "arch35/fused_mul_add_dag.h"
#include "arch35/fused_mul_add_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

template <uint64_t schMode>
__global__ __aicore__ void fused_mul_add(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (std::is_same<DTYPE_X1, int32_t>::value) {
        using OpDag = FusedMulAddOp::FusedMulAddInt32Op<int32_t>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    } else {
        using OpDag = typename FusedMulAddOp::FusedMulAddFloatOp<DTYPE_X1>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    }
}
