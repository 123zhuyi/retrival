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
 * \file reciprocal_grad.cpp
 * \brief ReciprocalGrad 算子 Kernel 入口（atvoss 框架 - Elewise 模式）
 */
#include "kernel_operator.h"
#include "arch35/reciprocal_grad_dag.h"
#include "arch35/reciprocal_grad_struct.h"
#include "arch35/reciprocal_grad_tiling_data.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void reciprocal_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 自定义包装结构体，需使用 REGISTER_TILING_DEFAULT 注册并配合 GET_TILING_DATA_WITH_STRUCT 获取
    REGISTER_TILING_DEFAULT(ReciprocalGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReciprocalGradTilingData, tilingData, tiling);

    TPipe pipe;

    // dtype 由构建系统根据 reciprocal_grad_def.cpp 中注册的 dtype 通过编译期宏 DTYPE_Y 注入
    using OpDag = NsReciprocalGrad::ReciprocalGradCompute<DTYPE_Y>::OpDag;
    ElementwiseSch<schMode, OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.Init(y, dy, z);
    sch.Process();
}
