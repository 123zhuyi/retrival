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
 * \file erfc_apt.cpp
 * \brief z = erfc(x)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "erfc_dag.h"
#include "erfc_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/elewise/elewise_base_struct.h"

using namespace Ops::Base;
namespace AscendC {

template <uint64_t schMode>
__global__ __aicore__ void erfc(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    constexpr bool isNeedCast = std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, bfloat16_t>::value;

    if constexpr (isNeedCast) {
        ElementwiseSch<schMode, ErfcOp::ErfcWithCast<DTYPE_X>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else {
        ElementwiseSch<schMode, ErfcOp::ErfcWithoutCast<DTYPE_X>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }

    return;
}

} // namespace AscendC
