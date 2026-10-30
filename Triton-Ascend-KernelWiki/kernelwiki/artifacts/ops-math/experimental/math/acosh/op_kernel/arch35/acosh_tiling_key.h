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
 * \file acosh_tiling_key.h
 * \brief Acosh 算子 TilingKey 定义
 *
 * 模板参数：
 *   D_T_X      — 数据类型 (C_DT_FLOAT16 / C_DT_FLOAT / C_DT_BF16)
 *   BUFFER_MODE — 缓冲模式 (0=单缓冲, 1=双缓冲)
 *
 * 共 6 个 TilingKey 组合（3 dtype × 2 buffer_mode）：
 *   TilingKey_A: D_T_X=C_DT_FLOAT16, BUFFER_MODE=0  (fp16 单缓冲)
 *   TilingKey_B: D_T_X=C_DT_FLOAT16, BUFFER_MODE=1  (fp16 双缓冲)
 *   TilingKey_C: D_T_X=C_DT_FLOAT,   BUFFER_MODE=0  (fp32 单缓冲)
 *   TilingKey_D: D_T_X=C_DT_FLOAT,   BUFFER_MODE=1  (fp32 双缓冲)
 *   TilingKey_E: D_T_X=C_DT_BF16,    BUFFER_MODE=0  (bf16 单缓冲)
 *   TilingKey_F: D_T_X=C_DT_BF16,    BUFFER_MODE=1  (bf16 双缓冲)
 */

#ifndef __ACOSH_TILING_KEY_H__
#define __ACOSH_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Acosh, ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // fp16 单缓冲 (TilingKey_A)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16), ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
    // fp32 单/双缓冲 (TilingKey_C / TilingKey_D)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
    // bf16 单/双缓冲 (TilingKey_E / TilingKey_F)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16), ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif // __ACOSH_TILING_KEY_H__
