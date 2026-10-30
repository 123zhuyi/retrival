/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file atan_grad.cpp
 * \brief AtanGrad Kernel 入口（arch35）
 *
 * 模板参数（与 atan_grad_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 对应）：
 *   - D_T_X:       数据类型（half / float / bfloat16_t）
 *   - BUFFER_MODE: 缓冲策略（0=单缓冲, 1=双缓冲）
 */

#include "atan_grad.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void atan_grad(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AtanGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AtanGradTilingData, tilingData, tiling);
    NsAtanGrad::AtanGrad<D_T_X, BUFFER_MODE> op;
    op.Init(x, dy, dx, &tilingData);
    op.Process();
}
