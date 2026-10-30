/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file round.cpp
 * \brief 
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/round_dag.h"
#include "arch35/round_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/round_tiling_struct.h"

using namespace AscendC;
using namespace RoundOp;
using namespace Ops::Base;

template <uint64_t schMode, uint64_t dType, typename DtypeX>
__global__ __aicore__ void RoundKernelI(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RoundTilingData);
    GET_TILING_DATA_WITH_STRUCT(RoundTilingData, tilingData, tiling);
    TPipe pipe;

    if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32)) {
        ElementwiseSch<schMode, typename RoundDag::RoundInt<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32_CONST)) {
        ElementwiseSch<schMode, typename RoundDag::RoundIntConst<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<int, 0>(tilingData.num);
        sch.Init(y);
        sch.Process();
    }  else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32_NEG_NINE)) {
        ElementwiseSch<schMode, typename RoundDag::RoundIntNegativeDecimalsNine<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32_NEGINF)) {
        ElementwiseSch<schMode, typename RoundDag::RoundIntNegativeDecimalsInf<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<int, 0>(tilingData.power);
        sch.template SetVar<int, 1>(tilingData.num);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32_NEG)) {
        ElementwiseSch<schMode, typename RoundDag::RoundIntNegativeDecimals<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<int, 0>(tilingData.power);
        sch.template SetVar<int, 1>(tilingData.num);
        sch.Init(x, y);
        sch.Process();
    }

    return;
}

template <uint64_t schMode, uint64_t dType, typename DtypeX>
__global__ __aicore__ void RoundKernelF(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RoundTilingData);
    GET_TILING_DATA_WITH_STRUCT(RoundTilingData, tilingData, tiling);
    TPipe pipe;

    if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_ZERO)) {
        ElementwiseSch<schMode, typename RoundDag::RoundZero<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_POSITIVE_DECIMALS)) {
        ElementwiseSch<schMode, typename RoundDag::RoundPositiveDecimals<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.decimals);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_NEGATIVE_DECIMALS)) {
        ElementwiseSch<schMode, typename RoundDag::RoundNegativeDecimals<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.decimals);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_NAN_DECIMALS)) {
        ElementwiseSch<schMode, typename RoundDag::RoundNan<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    }

    return;
}

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void round(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (std::is_same<DTYPE_X, int32_t>::value) {
        RoundKernelI<schMode, dType, DTYPE_X>(x, y, workspace, tiling);
    } else {
        RoundKernelF<schMode, dType, DTYPE_X>(x, y, workspace, tiling);
    }
    return;
}