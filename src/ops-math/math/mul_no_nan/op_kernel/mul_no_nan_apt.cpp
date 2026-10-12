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
 * \file mul_no_nan_apt.cpp
 * \brief mul_no_nan kernel: y = (x2 == 0) ? 0 : x1 * x2, element-wise with broadcast
 */

#include "kernel_operator.h"
#include "arch35/mul_no_nan_dag.h"
#include "arch35/mul_no_nan_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void mul_no_nan(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value) {
        using OpDag = typename MulNoNanOp::MulNoNanFloatCast<DTYPE_X1, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else {
        using OpDag = typename MulNoNanOp::MulNoNan<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}
