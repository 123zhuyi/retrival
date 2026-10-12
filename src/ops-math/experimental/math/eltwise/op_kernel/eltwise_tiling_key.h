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
 * \file eltwise_tiling_key.h
 * \brief Eltwise TilingKey definition (arch35)
 *
 * Template parameters:
 *   D_T  - DataType: C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT
 *   MODE - UINT(8): 0=PRODUCT, 1=SUM, 2=MAX
 *
 * 9 TilingKey combinations: 3 dtype x 3 mode
 */

#ifndef ELTWISE_TILING_KEY_H
#define ELTWISE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Eltwise, ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2));

ASCENDC_TPL_SEL(
    // float16 x 3 modes
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16), ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2)),
    // bfloat16 x 3 modes
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16), ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2)),
    // float32 x 3 modes
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2)), );

#endif // ELTWISE_TILING_KEY_H
