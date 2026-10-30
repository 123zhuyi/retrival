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
 * \file ndtri.cpp
 * \brief Ndtri Kernel 入口（arch35 / Ascend950）
 *
 * 模板参数：
 *   - D_T: 输入 Tensor 数据类型（float / half / bfloat16_t）
 *   - K_ALIGN: 32B 对齐标记（0=非对齐, 1=对齐）
 *
 * 迭代一仅 fp32 + 对齐真正验证；其他分支共用相同实现路径（后续替换）。
 *
 * 核函数参数顺序：1 输入 + 1 输出 + workspace + tiling
 */

#include "ndtri_kernel.h"

template <typename D_T, int K_ALIGN>
__global__ __aicore__ void ndtri(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NdtriTilingData);
    GET_TILING_DATA_WITH_STRUCT(NdtriTilingData, tilingData, tiling);
    NsNdtri::Ndtri<D_T, K_ALIGN> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
