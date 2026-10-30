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
 * \file reduce_std_v2.cpp
 * \brief reduce_std_v2
 */
#include <cmath>
#include "atvoss/reduce/reduce_sch.h"
#include "atvoss/reduce/reduce_operator.h"
#include "../reduce_var/arch35/reduce_var_empty.h"
#include "../reduce_var/arch35/reduce_var_pure_move.h"
#include "../reduce_var/arch35/reduce_var_sch.h"
#include "arch35/reduce_std_v2_tiling_key.h"

using namespace ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_std_v2(GM_ADDR x, GM_ADDR std, GM_ADDR mean, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    using PromoteType = Ops::Base::ReduceOpTmpl::__reduceType::GetPromoteType<DTYPE_X>::T;
    if constexpr (PatternID == 0 && LoopARCount == 0 && LoopInnerARCount == 0) {
        using Op = ReduceVarEmpty<DTYPE_X>;
        Op op(static_cast<DTYPE_X>(NAN));
        op.InitTiling(&tilingData);
        op.Init(&pipe, x, std, mean);
        op.Process();
    } else if constexpr (PatternID / Ops::Base::ReduceOpTmpl::CONST10 == Ops::Base::ReduceOpTmpl::CONST10) {
        using Op = ReduceVarPureMove<DTYPE_X>;
        Op op;
        op.InitTiling(&tilingData);
        op.Init(&pipe, x, std, mean);
        op.Process();
    } else {
        using Op = ReduceVarSch<DTYPE_X, PromoteType, REDUCE_TPL_VALUE, true>;
        Op op(&tilingData);
        op.Init(&pipe, x, std, mean, workspace);
        op.Process();
    }
}