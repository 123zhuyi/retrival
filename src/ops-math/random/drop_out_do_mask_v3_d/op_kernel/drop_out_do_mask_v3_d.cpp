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
 * \file drop_out_do_mask_v3_d.cpp
 * \brief
 */
#define DROP_OUT_DO_MASK_V3_D_TILING_KEY 100
#include "../drop_out_do_mask_v3/arch35/drop_out_do_mask_v3.h"

using namespace AscendC;
using namespace DropOutDoMaskV3;

extern "C" __global__ __aicore__ void drop_out_do_mask_v3_d(
    GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedTilingDataStruct);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    if (TILING_KEY_IS(DROP_OUT_DO_MASK_V3_D_TILING_KEY)) {
        DropOutDoMaskV3::DropOutDoMaskV3Op<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, mask, y);
        op.Process();
    }
}