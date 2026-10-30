/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file neg_struct.h
 * \brief neg_struct
 */
#ifndef NEG_STRUCT_H_
#define NEG_STRUCT_H_

#include "tiling/template_argument.h"

namespace NegOp {
#define NEG_TPL_FP16 1
#define NEG_TPL_BF16 2
#define NEG_TPL_FP32 3
#define NEG_TPL_INT8 4
#define NEG_TPL_INT32 5
#define NEG_TPL_INT64 6

#define NEG_TPL_SCH_MODE_0 0
#define NEG_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Neg, ASCENDC_TPL_UINT_DECL(scheMode, 1, ASCENDC_TPL_UI_LIST, NEG_TPL_SCH_MODE_0, NEG_TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(
        dType, NEG_TPL_FP16, NEG_TPL_BF16, NEG_TPL_FP32, NEG_TPL_INT8, NEG_TPL_INT32, NEG_TPL_INT64));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(scheMode, ASCENDC_TPL_UI_LIST, NEG_TPL_SCH_MODE_0, NEG_TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_SEL(
        dType, NEG_TPL_FP16, NEG_TPL_BF16, NEG_TPL_FP32, NEG_TPL_INT8, NEG_TPL_INT32, NEG_TPL_INT64)));

}
#endif // NEG_STRUCT_H_
