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
 * \file mod.cpp
 * \brief
 */

#include "mod.h"
#include "mod_tiling_data.h"
#include "mod_tiling_key.h"

using namespace ModNs;

// kernel function
template <int D_T_X1, int D_T_X2, int D_T_Y>
__global__ __aicore__ void mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(ModTilingData);
    GET_TILING_DATA_WITH_STRUCT(ModTilingData, tilingData, tiling);

    ModKernelImpl<D_T_X1, D_T_X2, D_T_Y>(x1, x2, y, &tilingData);
}