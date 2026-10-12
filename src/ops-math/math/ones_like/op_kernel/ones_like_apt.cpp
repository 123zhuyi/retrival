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
 * \file fills.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "arch35/ones_like_dag.h"
#include "arch35/ones_like_tiling_key.h"
#include "ones_like_struct.h"

using namespace AscendC;
using namespace Ops::Base;
using namespace OnesLikeNs;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void ones_like(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(OnesLikeTilingData);
    GET_TILING_DATA_WITH_STRUCT(OnesLikeTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (std::is_same<DTYPE_X, bool>::value) {
        ElementwiseSch<schMode, OnesLikeDAG<int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    } else {
        ElementwiseSch<schMode, OnesLikeDAG<DTYPE_X>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    }
}