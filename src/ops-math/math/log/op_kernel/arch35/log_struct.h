/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_struct.h
 * \brief
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef OPS_MATH_MATH_LOG_OP_KERNEL_ARCH35_LOG_STRUCT_H
#define OPS_MATH_MATH_LOG_OP_KERNEL_ARCH35_LOG_STRUCT_H
namespace LogOp {
#define TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE 1
#define TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE 2

#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Log, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(
        dType, TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE, TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_SEL(
        dType, TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE, TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE)));
} // namespace LogOp

#endif // OPS_MATH_MATH_LOG_OP_KERNEL_ARCH35_LOG_STRUCT_H
