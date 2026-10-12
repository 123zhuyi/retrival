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
 * \file nan_to_num_apt.cpp
 * \brief z = nan_to_num(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/nan_to_num_dag.h"
#include "arch35/nan_to_num_struct.h"
#include "arch35/nan_to_num_tiling_struct.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace NanToNumOp;
using namespace NanToNum;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void nan_to_num(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NanToNumTilingData);
    GET_TILING_DATA_WITH_STRUCT(NanToNumTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, NanToNumOp::NanToNumDAG<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_0>(tilingData.nan);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_1>(tilingData.posinf);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_2>(tilingData.neginf);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, NanToNumOp::NanToNumDAG<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_0>(tilingData.nan);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_1>(tilingData.posinf);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_2>(tilingData.neginf);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, NanToNumOp::NanToNumDAG<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_0>(tilingData.nan);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_1>(tilingData.posinf);
        sch.template SetVar<float, NanToNumOp::PLACEHOLDER_INDEX_2>(tilingData.neginf);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}
