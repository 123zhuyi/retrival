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
 * \file add_mat_mat_elements_tiling_key.h
 * \brief AddMatMatElements TilingKey 模板参数定义
 *
 * 模板参数 D_T：数据类型（C_DT_FLOAT16 / C_DT_FLOAT / C_DT_BF16）
 *   - TilingKey_0: D_T = C_DT_FLOAT16 (fp16 直接路径)
 *   - TilingKey_1: D_T = C_DT_FLOAT   (fp32 直接路径)
 *   - TilingKey_2: D_T = C_DT_BF16    (bf16 Cast 绕行路径)
 */

#ifndef ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_
#define ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// 声明模板参数：D_T 从输入 tensor 0 的 dtype 中选取
ASCENDC_TPL_ARGS_DECL(
    AddMatMatElements, ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)), );

// 显式枚举所有合法 TilingKey 组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16)), );

#endif // ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_
