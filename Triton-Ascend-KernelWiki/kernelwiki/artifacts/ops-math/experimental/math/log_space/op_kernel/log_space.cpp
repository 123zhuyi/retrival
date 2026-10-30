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
 * \file log_space.cpp
 * \brief LogSpace Kernel 入口（arch35 / ascend950）
 *
 * 模板参数（与 log_space_tiling_key.h 对应）：
 *   - D_T_Y: 输出 dtype
 *   - MODE:  0 NORMAL / 1 SINGLE
 *
 * 核函数参数顺序（registry-invoke 固定）：
 *   (GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling)
 *
 * LogSpace 无输入 tensor，仅 1 个输出；workspace/tiling 位置固定。
 */

#include "log_space.h"

template <typename D_T_Y, int MODE>
__global__ __aicore__ void log_space(GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogSpaceTilingData, tilingData, tiling);
    NsLogSpace::LogSpace<D_T_Y, MODE> op;
    op.Init(result, &tilingData);
    op.Process();
}
