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
 * \file zeros_like.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/zeros_like_dag.h"
#include "arch35/zeros_like_tiling_key.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "zeros_like_struct.h"

using namespace AscendC;
using namespace ZerosLikeNs;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void zeros_like(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ZerosLikeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ZerosLikeTilingData, tilingData, tiling);

    TPipe pipe;
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        ElementwiseSch<schMode, ZerosLikeOp::ZerosLikeDAG<int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        ElementwiseSch<schMode, ZerosLikeOp::ZerosLikeDAG<int16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        ElementwiseSch<schMode, ZerosLikeOp::ZerosLikeDAG<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    } else {
        // sizeof(DTYPE_X) == sizeof(int64_t)
        ElementwiseSch<schMode, ZerosLikeOp::ZerosLikeDAG<int64_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    }
}