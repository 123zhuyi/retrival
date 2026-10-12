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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_with_agent.cpp
 * \brief AsinhWithAgent 算子 Kernel 入口
 *
 * 模板参数（与 asinh_with_agent_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X: 数据类型（half=float16 / float=float32），由 ASCENDC_TPL_DATATYPE_DECL 定义
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲），由 ASCENDC_TPL_UINT_DECL 定义
 *
 * 核函数参数顺序（固定）：输入 → 输出 → workspace → tiling
 */

#include "asinh_with_agent.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void asinh_with_agent(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AsinhWithAgentTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinhWithAgentTilingData, tilingData, tiling);
    NsAsinhWithAgent::AsinhWithAgent<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
