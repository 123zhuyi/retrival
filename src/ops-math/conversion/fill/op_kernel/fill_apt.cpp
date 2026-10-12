/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill.cpp
 * \brief fill kernel
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/fill_dag.h"
#include "arch35/fill_struct.h"
#include "atvoss/elewise/elewise_sch_16b.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace FillOp;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingData16B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData16B, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (dType == TPL_FP16) {
        ElementwiseSch16B<schMode, FillDag<half>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch16B<schMode, FillDag<float>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch16B<schMode, FillDag<bfloat16_t>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT8) {
        ElementwiseSch16B<schMode, FillDag<int8_t>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT32) {
        ElementwiseSch16B<schMode, FillDag<int32_t>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_INT64) {
        ElementwiseSch16B<schMode, FillDag<int64_t>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    } else if constexpr (dType == TPL_BOOL) {
        ElementwiseSch16B<schMode, FillDag<int8_t>::OpDag> sch(tilingData);
        sch.Init(value, y);
        sch.Process();
    }
    return;
}
