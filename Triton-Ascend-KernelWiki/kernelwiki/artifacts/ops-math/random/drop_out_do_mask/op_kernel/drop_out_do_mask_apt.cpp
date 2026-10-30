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
 * \file drop_out_do_mask.cpp
 * \brief
 */

#define DO_MASK_TILING_KEY 100

#include "arch35/drop_out_do_mask.h"

using namespace DropOutDoMask;

extern "C" __global__ __aicore__ void drop_out_do_mask(
    GM_ADDR x, GM_ADDR mask, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(DO_MASK_TILING_KEY)) {
        DropOutDoMask::DropOutDoMaskImpl<DTYPE_X> op;
        op.Init(x, mask, prob, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}