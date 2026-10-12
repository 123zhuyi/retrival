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
 * \file strided_slice_assign_v2.cpp
 * \brief
 */

#include "strided_slice_assign_v2.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin,
    GM_ADDR end, GM_ADDR strides, GM_ADDR axes, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelStridedSliceAssignV2<DTYPE_VAR> op(&pipe);
        op.Init(var, input_value, begin, end, strides, axes, var_out, &tilingData);
        op.Process();
    }
}
