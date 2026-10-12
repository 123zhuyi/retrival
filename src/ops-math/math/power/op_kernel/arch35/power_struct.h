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
 * \file power_struct.h
 * \brief Power op template-argument declaration. Decides which DAG variant to instantiate
 *        in the kernel via two tilingKey fields: culType and dType.
 */

#ifndef OPS_MATH_POWER_STRUCT_H_
#define OPS_MATH_POWER_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace PowerOp {
// dtype encoding
#define POWER_TPL_DTYPE_FP16 1
#define POWER_TPL_DTYPE_BF16 2
#define POWER_TPL_DTYPE_FP32 3

// culType encoding: matches optiling::CulTypeEnum
#define POWER_TPL_CUL_ALL_ZEROS        0
#define POWER_TPL_CUL_BROADCAST_SCALAR 1
#define POWER_TPL_CUL_LINEAR           2
#define POWER_TPL_CUL_SQUARE           3
#define POWER_TPL_CUL_CUBE             4
#define POWER_TPL_CUL_GENERIC_POW_POS  5
#define POWER_TPL_CUL_GENERIC_POW_NEG  6

// elewise schMode
#define POWER_TPL_SCH_MODE_0 0
#define POWER_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Power,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, POWER_TPL_SCH_MODE_0, POWER_TPL_SCH_MODE_1),
    ASCENDC_TPL_UINT_DECL(
        culType, 4, ASCENDC_TPL_UI_LIST,
        POWER_TPL_CUL_ALL_ZEROS,
        POWER_TPL_CUL_BROADCAST_SCALAR,
        POWER_TPL_CUL_LINEAR,
        POWER_TPL_CUL_SQUARE,
        POWER_TPL_CUL_CUBE,
        POWER_TPL_CUL_GENERIC_POW_POS,
        POWER_TPL_CUL_GENERIC_POW_NEG),
    ASCENDC_TPL_DTYPE_DECL(dType, POWER_TPL_DTYPE_FP16, POWER_TPL_DTYPE_BF16, POWER_TPL_DTYPE_FP32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, POWER_TPL_SCH_MODE_0, POWER_TPL_SCH_MODE_1),
    ASCENDC_TPL_UINT_SEL(
        culType, ASCENDC_TPL_UI_LIST,
        POWER_TPL_CUL_ALL_ZEROS,
        POWER_TPL_CUL_BROADCAST_SCALAR,
        POWER_TPL_CUL_LINEAR,
        POWER_TPL_CUL_SQUARE,
        POWER_TPL_CUL_CUBE,
        POWER_TPL_CUL_GENERIC_POW_POS,
        POWER_TPL_CUL_GENERIC_POW_NEG),
    ASCENDC_TPL_DTYPE_SEL(dType, POWER_TPL_DTYPE_FP16, POWER_TPL_DTYPE_BF16, POWER_TPL_DTYPE_FP32)));
} // namespace PowerOp

#endif // OPS_MATH_POWER_STRUCT_H_
