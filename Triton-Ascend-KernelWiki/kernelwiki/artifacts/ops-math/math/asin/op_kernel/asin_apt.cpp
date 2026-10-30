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
 * \file asin_apt.cpp
 * \brief Asin kernel - 使用自定义 AsinCustom Vf
 *
 * \note 超越函数统一在 float 下计算，half/bfloat16 通过 DAG 中的 Cast 节点转换
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/asin_dag.h"
#include "arch35/asin_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;

__global__ __aicore__ void asin(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AsinTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    if constexpr (std::is_same<DTYPE_X, half>::value) {
        // half: cast to float -> compute -> cast back
        ElementwiseSch<0UL, AsinDag::AsinOpWithCast<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, float>::value) {
        // float: direct compute
        ElementwiseSch<0UL, AsinDag::AsinOpDirect<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
        // bfloat16: cast to float -> compute -> cast back
        ElementwiseSch<0UL, AsinDag::AsinOpWithCast<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    }
}
