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
 * \file adds_struct.h
 * \brief Adds 算子 TilingKey 定义（atvoss 框架 - Elewise 模式）
 */

#ifndef ADDS_STRUCT_H_
#define ADDS_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace AddsOp {
#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3
#define TPL_INT16 4
#define TPL_INT32 5
#define TPL_INT64 6

#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Adds,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32, TPL_INT16, TPL_INT32,TPL_INT64)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16, TPL_BF16, TPL_FP32, TPL_INT16, TPL_INT32,TPL_INT64)
    )
);
} // namespace AddsOp


#endif  // ADDS_STRUCT_H_