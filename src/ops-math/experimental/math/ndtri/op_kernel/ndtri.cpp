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
 * \file ndtri.cpp
 * \brief Ndtri Kernel 入口（arch35 / Ascend950）
 *
 * dtype 由 def 驱动：DTYPE_SELF=self dtype（T），在入口处实例化 op 类的 T 模板实参。
 * 模板参数 K_ALIGN: 32B 对齐标记（0=非对齐, 1=对齐）
 */

#include "ndtri_kernel.h"

template <int K_ALIGN>
__global__ __aicore__ void ndtri(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NdtriTilingData);
    GET_TILING_DATA_WITH_STRUCT(NdtriTilingData, tilingData, tiling);
    NsNdtri::Ndtri<DTYPE_SELF, K_ALIGN> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
