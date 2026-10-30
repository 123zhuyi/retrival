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
 * \file masked_fill.cpp
 * \brief masked_fill kernel
 */

#include "kernel_operator.h"
#include "arch35/masked_fill_dag.h"
#include "arch35/masked_fill_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void masked_fill(GM_ADDR x, GM_ADDR mask, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X, bool>::value) {
        using OpDag = MaskedFill<int8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, mask, value, y);
    } else {
        using OpDag = MaskedFill<DTYPE_X>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x, mask, value, y);
    }
}
