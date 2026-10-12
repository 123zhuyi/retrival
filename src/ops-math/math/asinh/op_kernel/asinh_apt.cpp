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
 * \file asinh_arch35.cpp
 * \brief Asinh 算子 Kernel 入口（arch35 / Ascend950）
 *
 * dtype 由 op def 的 DataType profile 展开，kernel 通过 DTYPE_X 获取输入类型。
 */

#include "arch35/asinh.h"
__global__ __aicore__ void asinh(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(AsinhTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinhTilingData, tilingData, tiling);

    NsAsinh::Asinh<DTYPE_X> op;
    op.Init(input, out, &tilingData);
    op.Process();
}
