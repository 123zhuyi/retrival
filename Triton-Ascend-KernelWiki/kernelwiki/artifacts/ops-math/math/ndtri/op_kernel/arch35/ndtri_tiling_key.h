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
 * \file ndtri_tiling_key.h
 * \brief Ndtri Tiling 模板参数定义
 *
 * 模板参数维度：
 *   - D_T: 输入/输出 Tensor 的数据类型（C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *   - K_ALIGN: 32B 对齐标记（1=对齐, 0=非对齐）
 *
 * 6 个 TilingKey：{fp32, fp16, bf16} × {对齐, 非对齐}
 * 全部通过 Cast(fp16↔fp32) / Cast(bf16↔fp32) 链路实现；fp32 走 Adds(*, 0.0f) 等价 Copy。
 */

#ifndef NDTRI_TILING_KEY_H_
#define NDTRI_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Ndtri,
    ASCENDC_TPL_DATATYPE_DECL(D_T,
        C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16,
        ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(K_ALIGN, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

// 6 个 TilingKey：{fp32, fp16, bf16} × {对齐, 非对齐}
// 全部通过 Cast(fp16↔fp32) / Cast(bf16↔fp32) 链路实现；fp32 走 Adds(*, 0.0f) 等价 Copy。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif  // NDTRI_TILING_KEY_H_
