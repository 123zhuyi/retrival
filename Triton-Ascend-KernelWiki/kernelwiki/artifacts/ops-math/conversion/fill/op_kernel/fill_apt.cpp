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

#include "fill_dag.h"
#include "fill_struct.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
__global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FillStruct::FillTilingDataStruct);
    GET_TILING_DATA_WITH_STRUCT(FillStruct::FillTilingDataStruct, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        if constexpr (std::is_same<DTYPE_VALUE, bool>::value) {
            Ops::Base::ElementwiseSch<0UL, FillDag<int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(value, y);
            sch.Process();
        } else {
            Ops::Base::ElementwiseSch<0UL, FillDag<DTYPE_VALUE>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(value, y);
            sch.Process();
        }
    }
    return;
}