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
 * \file sinkhorn.cpp
 * \brief
 */
#include "sinkhorn.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 这两句都不要删除，有些调用框架还不能适配，需要保留
    SetSysWorkspace(workspace);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_IS(0)) {
        // ge::DT_FLOAT
        AscendC::KernelSinkhorn<float, float> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // ge::DT_FLOAT16
        AscendC::KernelSinkhorn<half, half> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(27)) {
        // ge::DT_BFLOAT16
        AscendC::KernelSinkhorn<float, bfloat16_t> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    }
}
