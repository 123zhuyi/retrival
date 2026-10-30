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
 * \file is_finite_apt.cpp
 * \brief y = abs(x) < infinity
 */
#include "arch35/is_finite_dag.h"
#include "arch35/is_finite_struct_arch35.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace IsFiniteOp;

template <uint64_t extraMode, uint64_t schMode, uint64_t dType>
__global__ __aicore__ void is_finite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (dType == TPL_FP16) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<schMode, IsFiniteDag<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<schMode, IsFiniteDag<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<schMode, IsFiniteDag<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}
// fixme: this comment is added for incremental build opp_kernel, please delete before submit to master.
