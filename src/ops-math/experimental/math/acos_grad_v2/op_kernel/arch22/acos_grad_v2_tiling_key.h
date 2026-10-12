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
 * \file acos_grad_v2_tiling_key.h
 * \brief AcosGradV2 TilingKey 定义（arch22 / Ascend910B）
 *
 * 模板参数 D_T 由输入 0 (y) 的 dtype 决定：FP32 / FP16 / BF16。
 */

#ifndef ACOS_GRAD_V2_TILING_KEY_H
#define ACOS_GRAD_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(AcosGradV2,
                      ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16)), );

#endif // ACOS_GRAD_V2_TILING_KEY_H
