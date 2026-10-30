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
 * \file power_apt.cpp
 * \brief y = exp(power * log(x * scale + shift)) elementwise kernel
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/power_dag.h"
#include "arch35/power_struct.h"
#include "arch35/power_tiling_struct.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace Ops::Base;
using namespace PowerOp;

template <uint64_t schMode, uint64_t culType, typename DtypeX>
__global__ __aicore__ void PowerKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PowerOp::PowerTilingData);
    GET_TILING_DATA_WITH_STRUCT(PowerOp::PowerTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_ALL_ZEROS)) {
        ElementwiseSch<schMode, typename PowerAllZerosDag<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_BROADCAST_SCALAR)) {
        ElementwiseSch<schMode, typename PowerBcastScalarDag<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.Init(y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_LINEAR)) {
        ElementwiseSch<schMode, typename PowerLinearDag<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.template SetVar<float, POWER_VAR_IDX_1>(tilingData.shift);
        sch.Init(x, y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_SQUARE)) {
        ElementwiseSch<schMode, typename PowerSquareDag<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.template SetVar<float, POWER_VAR_IDX_1>(tilingData.shift);
        sch.Init(x, y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_CUBE)) {
        ElementwiseSch<schMode, typename PowerCubeDag<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.template SetVar<float, POWER_VAR_IDX_1>(tilingData.shift);
        sch.Init(x, y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_GENERIC_POW_POS)) {
        ElementwiseSch<schMode, typename PowerGenericDag<DtypeX, 1>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.template SetVar<float, POWER_VAR_IDX_1>(tilingData.shift);
        sch.template SetVar<float, POWER_VAR_IDX_2>(tilingData.power);
        sch.template SetVar<float, POWER_VAR_IDX_3>(tilingData.negScalar);
        sch.Init(x, y); 
        sch.Process();
    } else if constexpr (culType == static_cast<uint64_t>(POWER_TPL_CUL_GENERIC_POW_NEG)) {
        ElementwiseSch<schMode, typename PowerGenericDag<DtypeX, 0>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, POWER_VAR_IDX_0>(tilingData.scale);
        sch.template SetVar<float, POWER_VAR_IDX_1>(tilingData.shift);
        sch.template SetVar<float, POWER_VAR_IDX_2>(tilingData.power);
        sch.template SetVar<float, POWER_VAR_IDX_3>(tilingData.negScalar);
        sch.Init(x, y); 
        sch.Process();
    }
}

template <uint64_t schMode, uint64_t culType, uint64_t dType>
__global__ __aicore__ void power(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (dType == POWER_TPL_DTYPE_FP16) {
        PowerKernel<schMode, culType, half>(x, y, workspace, tiling);
    } else if constexpr (dType == POWER_TPL_DTYPE_BF16) {
        PowerKernel<schMode, culType, bfloat16_t>(x, y, workspace, tiling);
    } else if constexpr (dType == POWER_TPL_DTYPE_FP32) {
        PowerKernel<schMode, culType, float>(x, y, workspace, tiling);
    }
}
