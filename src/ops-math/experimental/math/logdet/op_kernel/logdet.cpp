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
 * \file logdet.cpp
 * \brief Logdet Kernel 入口（现代规范：模板 + if constexpr 分发，禁用 TILING_KEY_IS）。
 *
 * 模板参数（与 logdet_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一一对应）：
 *   - D_T:          数据类型（仅 fp32 / float）
 *   - MEM_STRATEGY: 0=FULL_RESIDENT（全驻留）, 1=BLOCKED（核内分块）
 *
 * Kernel parameter order: input(self) -> output(out) -> workspace -> tiling.
 */

#include "logdet.h"
#include "logdet_tiling_data.h"
#include "logdet_tiling_key.h"  // ASCENDC_TPL_ARGS_DECL/SEL：kernel 编译期模板参数声明（codegen 据此实例化模板）

template <typename D_T, int MEM_STRATEGY>
__global__ __aicore__ void logdet(
    GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(LogdetTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogdetTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    NsLogdet::LogdetKernel<D_T, MEM_STRATEGY> op;
    op.Init(self, out, workspace, &tilingData, &pipe);
    op.Process();
}
