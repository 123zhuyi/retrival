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
 * \file add_mat_mat_elements_apt.cpp
 * \brief AddMatMatElements Kernel 入口（arch35 架构，Ascend950）
 *
 * 模板参数（与 arch35/add_mat_mat_elements_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 对应）：
 *   D_T: 数据类型（half / float / bfloat16_t）
 *
 * TilingKey 映射：
 *   TilingKey_0: D_T = half        (fp16 直接路径)
 *   TilingKey_1: D_T = float       (fp32 直接路径)
 *   TilingKey_2: D_T = bfloat16_t  (bf16 Cast 绕行路径)
 *
 * 注意：alpha/beta 已定义为 Attr（非 tensor Input），不在 GM_ADDR 参数列表中出现。
 *       alpha/beta 通过 TilingData 传入 Kernel。
 */

#include "arch35/add_mat_mat_elements.h"

template <typename D_T>
__global__ __aicore__ void add_mat_mat_elements(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddMatMatElementsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddMatMatElementsTilingData, tilingData, tiling);

    NsAddMatMatElements::KernelAddMatMatElements<D_T> op;
    op.Init(a, b, c, cOut, tilingData);
    op.Process();
}
