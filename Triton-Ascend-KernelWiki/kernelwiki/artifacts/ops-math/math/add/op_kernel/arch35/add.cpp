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
 * \file add.cpp
 * \brief add kernel
 */

#include "kernel_operator.h"
#include "add_dag.h"
#include "add_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace Ops::Base;
using namespace AscendC;

template <uint64_t schMode>
__global__ __aicore__ void add(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr bool isMixDtype = (std::is_same<DTYPE_X1, half>::value && std::is_same<DTYPE_X2, float>::value) ||
                                (std::is_same<DTYPE_X1, float>::value && std::is_same<DTYPE_X2, half>::value) ||
                                (std::is_same<DTYPE_X1, bfloat16_t>::value && std::is_same<DTYPE_X2, float>::value) ||
                                (std::is_same<DTYPE_X1, float>::value && std::is_same<DTYPE_X2, bfloat16_t>::value);
    constexpr bool isNeedCast = (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value ||
                                 std::is_same<DTYPE_X1, float>::value) &&
                                (!isMixDtype);

    if constexpr (std::is_same<DTYPE_X1, bool>::value) {
        using OpDag = AddOp::AddBoolCompute<int8_t>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (isMixDtype) {
        using OpDag = AddOp::AddMixDtypeCompute<DTYPE_X1, DTYPE_X2>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (isNeedCast) {
        using OpDag = AddOp::AddWithCastCompute<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else {
        using OpDag = AddOp::AddWithoutCastCompute<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}
