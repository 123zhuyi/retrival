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
 * \file assign_add.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/assign_add_dag.h"
#include "arch35/assign_add_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/elewise/elewise_base_struct.h"

using namespace Ops::Base;
namespace AscendC {

template <uint64_t schMode, uint64_t dType, typename DtypeRef, typename DtypeValue>
__global__ __aicore__ void AssignAddKernel(GM_ADDR ref, GM_ADDR value, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Ops::Base::EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(Ops::Base::EleBaseTilingDataV2, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    ElementwiseSch<schMode, typename AssignAddDag::AssignAddDAG<DtypeRef, DtypeValue>::OpDag> sch(
        &tilingData, &pipe); // 获取Schedule
    sch.Init(ref, value, out);
    sch.Process();

    return;
}

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void assign_add(GM_ADDR ref, GM_ADDR value, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    return AssignAddKernel<schMode, dType, DTYPE_REF, DTYPE_VALUE>(ref, value, out, workspace, tiling);
}

} // namespace AscendC