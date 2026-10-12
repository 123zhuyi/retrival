/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SLICE_LAST_DIM_TILING_KEY_H_
#define _SLICE_LAST_DIM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define SLICE_LAST_DIM_COPY_MODE_CONTINUOUS 0
#define SLICE_LAST_DIM_COPY_MODE_STRIDED 1
#define SLICE_LAST_DIM_UB_AXIS_0 0
#define SLICE_LAST_DIM_UB_AXIS_1 1

ASCENDC_TPL_ARGS_DECL(
    SliceLastDim,
    ASCENDC_TPL_UINT_DECL(
        copyMode, 8, ASCENDC_TPL_UI_LIST, SLICE_LAST_DIM_COPY_MODE_CONTINUOUS, SLICE_LAST_DIM_COPY_MODE_STRIDED),
    ASCENDC_TPL_UINT_DECL(ubAxis, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 1), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(
            copyMode, ASCENDC_TPL_UI_LIST, SLICE_LAST_DIM_COPY_MODE_CONTINUOUS, SLICE_LAST_DIM_COPY_MODE_STRIDED),
        ASCENDC_TPL_UINT_SEL(ubAxis, ASCENDC_TPL_UI_RANGE, 1, 0, 1), ), );

#endif
