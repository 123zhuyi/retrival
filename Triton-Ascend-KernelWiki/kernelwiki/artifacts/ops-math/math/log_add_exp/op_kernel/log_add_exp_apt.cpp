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
 * \file log_add_exp_apt.cpp
 * \brief log_add_exp kernel
 */

#include "kernel_operator.h"
#include "arch35/log_add_exp_dag.h"
#include "arch35/log_add_exp_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace Ops::Base;
using namespace AscendC;

// formulaType: 0=simplified, 1=full
template <uint64_t schMode, uint64_t formulaType>
__global__ __aicore__ void log_add_exp(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr bool useFullFormula = (formulaType == 1);

    if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
        if constexpr (useFullFormula) {
            using OpDag = LogAddExpOp::LogAddExpFullWithCastCompute<DTYPE_X1>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else {
            using OpDag = LogAddExpOp::LogAddExpSimplifiedWithCastCompute<DTYPE_X1>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    } else {
        if constexpr (useFullFormula) {
            using OpDag = LogAddExpOp::LogAddExpFullCompute<DTYPE_X1>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else {
            using OpDag = LogAddExpOp::LogAddExpSimplifiedCompute<DTYPE_X1>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    }
}
