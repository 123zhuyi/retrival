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
 * \file signbit_apt.cpp
 * \brief signbit. kernel
 */

#include "kernel_operator.h"
#include "arch35/signbit_float_dag.h"
#include "arch35/signbit_integral_dag.h"
#include "arch35/signbit_tiling_struct.h"
#include "arch35/signbit_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace Ops::Base;
using namespace AscendC;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void signbit(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SignbitTilingData);
    GET_TILING_DATA_WITH_STRUCT(SignbitTilingData, tilingData, tiling);
    TPipe pipe;

    if constexpr (dType == TPL_FP16) {
        Ops::Base::ElementwiseSch<schMode, SignbitFloatCompute<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        Ops::Base::ElementwiseSch<schMode, SignbitFloatCompute<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        Ops::Base::ElementwiseSch<schMode, SignbitFloatCompute<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT64) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<int64_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_UINT64) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<uint64_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT32) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<int32_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT8) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<int8_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_UINT8) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<uint8_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_BOOL) {
        Ops::Base::ElementwiseSch<schMode, SignbitIntegralOp::SignbitIntegralCompute<int8_t>::OpDag> sch(
            &(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_DOUBLE) {
        Ops::Base::ElementwiseSch<schMode, SignbitDoubleCompute<double>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}