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
 * \file slogdet_tiling_key.h
 * \brief Slogdet TilingKey 模板参数声明（现代规范：ASCENDC_TPL_*，禁用整型常量宏 + TILING_KEY_IS）。
 *
 * 模板参数:
 *   D_T:          输入/输出数据类型（仅 fp32）
 *   MEM_STRATEGY: 单矩阵核内驻留策略 (0=FULL_RESIDENT 全驻留, 1=BLOCKED 核内分块)
 */

#ifndef SLOGDET_TILING_KEY_H
#define SLOGDET_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Slogdet,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(MEM_STRATEGY, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MEM_STRATEGY, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);

#endif // SLOGDET_TILING_KEY_H
