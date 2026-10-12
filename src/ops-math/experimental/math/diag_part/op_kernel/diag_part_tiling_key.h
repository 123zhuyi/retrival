/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DIAG_PART_TILING_KEY_H_
#define DIAG_PART_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0

#define DIAG_PART_TPL_DTYPE_FLOAT16 1
#define DIAG_PART_TPL_DTYPE_FLOAT 0
#define DIAG_PART_TPL_DTYPE_INT32 3

ASCENDC_TPL_ARGS_DECL(
    DiagPart, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0),
    ASCENDC_TPL_UINT_DECL(
        dtype, 2, ASCENDC_TPL_UI_LIST, DIAG_PART_TPL_DTYPE_FLOAT, DIAG_PART_TPL_DTYPE_FLOAT16,
        DIAG_PART_TPL_DTYPE_INT32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0),
    ASCENDC_TPL_UINT_SEL(
        dtype, ASCENDC_TPL_UI_LIST, DIAG_PART_TPL_DTYPE_FLOAT, DIAG_PART_TPL_DTYPE_FLOAT16,
        DIAG_PART_TPL_DTYPE_INT32)));

#endif