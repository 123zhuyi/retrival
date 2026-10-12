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

/**
 * \file inv_grad_tiling_key.h
 * \brief InvGrad TilingKey definition (arch35)
 *
 * TilingKey mapping (based on input dtype of y, which must match dy):
 *   TilingKey 0: D_T_Y = C_DT_FLOAT     (float32 direct)
 *   TilingKey 1: D_T_Y = C_DT_FLOAT16   (float16 up-cast via fp32)
 *   TilingKey 2: D_T_Y = C_DT_BF16      (bfloat16 up-cast via fp32)
 */

#ifndef INV_GRAD_TILING_KEY_H
#define INV_GRAD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    InvGrad, ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_BF16)), );

#endif // INV_GRAD_TILING_KEY_H
