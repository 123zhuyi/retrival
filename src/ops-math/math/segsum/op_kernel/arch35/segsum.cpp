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
 * \file segsum.cpp
 * \brief Segsum kernel entry for arch35 / Ascend950.
 */

#include "segsum_arch35.h"

using namespace SegsumArch35;

extern "C" __global__ __aicore__ void segsum(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    REGISTER_TILING_DEFAULT(SegsumTilingDataArch35);
    GET_TILING_DATA_WITH_STRUCT(SegsumTilingDataArch35, tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        Segsum<DTYPE_X, 0> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        Segsum<DTYPE_X, 1> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        return;
    }
}
