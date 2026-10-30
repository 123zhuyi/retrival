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
 * \file is_finite.cpp
 * \brief
 */

#include "is_finite.h"
#include "is_finite_struct.h"

using namespace IsFiniteNs;

// kernel function
template<int D_T_X, int D_T_Y>
__global__ __aicore__ void is_finite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(IsFiniteTilingData);
    GET_TILING_DATA_WITH_STRUCT(IsFiniteTilingData, tilingData, tiling);

    IsFiniteKernelImpl<D_T_X, D_T_Y>(x, y, &tilingData);
}