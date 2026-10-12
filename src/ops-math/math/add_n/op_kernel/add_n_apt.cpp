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
 * \file add_n.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/add_n_regbase_fullload.h"
#include "arch35/add_n_regbase.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void add_n(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (TILING_KEY_IS(100ULL)) {
        AddNRegbase::AddNRegbase<DTYPE_X> op(tilingData, &pipe);
        op.Init(x, y);
        op.Process();
        return;
    } else if (TILING_KEY_IS(101ULL)) {
        AddNRegbaseFullLoad::AddNRegbaseFullLoad<DTYPE_X> op(tilingData, &pipe);
        op.Init(x, y);
        op.Process();
        return;
    }
    return;
}