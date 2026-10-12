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
 * \file is_close.cpp
 * \brief is_close
 */

#include "kernel_operator.h"
#include "arch35/is_close_dag.h"
#include "arch35/is_close_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode, uint32_t equalNan>
__global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (static_cast<uint32_t>(equalNan) == 0) {
        using OpDag = IsCloseOp::IsCloseDag<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else {
        using OpDag = IsCloseOp::IsCloseEqualNanDag<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}