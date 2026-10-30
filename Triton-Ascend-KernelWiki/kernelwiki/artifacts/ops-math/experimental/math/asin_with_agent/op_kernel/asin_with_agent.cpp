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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_arch32.cpp
 * \brief AsinWithAgent Kernel 入口（arch32: Ascend910B）
 *
 * 模板参数 D_T 由 TilingKey 决定（通过 ASCENDC_TPL_SEL_PARAM 设置）：
 *   D_T=float   -> TilingKey=0（Group A fp32）
 *   D_T=half    -> TilingKey=1（Group A fp16）
 *   D_T=float   -> TilingKey=2（Group B DOUBLE：op_api 层已将 fp64 转为 fp32，Kernel 接收 fp32）
 *   D_T=int8_t  -> TilingKey=3（Group C INT8）
 *   D_T=int16_t -> TilingKey=4（Group C INT16）
 *   D_T=int32_t -> TilingKey=5（Group C INT32）
 *   D_T=int64_t -> TilingKey=6（Group C INT64）
 *   D_T=uint8_t -> TilingKey=7（Group C UINT8）
 *   D_T=uint8_t -> TilingKey=8（Group C BOOL，与 UINT8 路径相同）
 *
 * 注意：DOUBLE（TilingKey=2）在 Kernel 侧使用 D_T=float，因为 op_api 层已完成 fp64->fp32 转换。
 *       BOOL（TilingKey=8）在 Kernel 侧使用 D_T=uint8_t（bool 底层为 uint8_t）。
 *
 * Kernel 参数顺序（固定）：input, output, workspace, tiling
 */

#include "asin_with_agent_impl.h"

template <typename D_T>
__global__ __aicore__ void asin_with_agent(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AsinWithAgentTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinWithAgentTilingData, tilingData, tiling);

    NsAsinWithAgent::AsinWithAgent<D_T> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();
}
