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
 * \file tensor_redirect_apt.cpp
 * \brief tensor_redirect kernel entry
 */

#include "arch35/tensor_redirect.h"
#include "arch35/tensor_redirect_tiling_key.h"

using namespace AscendC;

// 核函数参数顺序（固定）：输入 x -> 输出 output_x -> workspace -> tiling
template <uint64_t schMode>
__global__ __aicore__ void tensor_redirect(GM_ADDR x, GM_ADDR output_x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector，不使用 Cube
    REGISTER_TILING_DEFAULT(TensorRedirectTilingData);
    GET_TILING_DATA_WITH_STRUCT(TensorRedirectTilingData, tilingData, tiling);

    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    TPipe pipe;

    // 按元素字节宽静态分发
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        NsTensorRedirect::TensorRedirectKernel<int8_t> op;
        op.Init(x, output_x, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        NsTensorRedirect::TensorRedirectKernel<int16_t> op;
        op.Init(x, output_x, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        NsTensorRedirect::TensorRedirectKernel<int32_t> op;
        op.Init(x, output_x, &tilingData, &pipe);
        op.Process();
    } else { // sizeof(DTYPE_X) == sizeof(int64_t)
        NsTensorRedirect::TensorRedirectKernel<int64_t> op;
        op.Init(x, output_x, &tilingData, &pipe);
        op.Process();
    }
}
