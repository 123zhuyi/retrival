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
 * \file acos_grad_v2.cpp
 * \brief AcosGradV2 算子 Kernel 入口（arch22 / Ascend910B）
 *
 * Inputs (对齐 def.cpp / README):
 *   y  : 前向 Acos 的输入张量
 *   dy : 上游梯度
 *   z  : 输出梯度
 *
 * 公式: z = -dy / sqrt(1 - y^2)
 */

#include "acos_grad_v2.h"

template <typename D_T>
__global__ __aicore__ void acos_grad_v2(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AcosGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(AcosGradV2TilingData, tilingData, tiling);
    NsAcosGradV2::KernelAcosGradV2<D_T> op;
    op.Init(y, dy, z, &tilingData);
    op.Process();
}
