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
 * \file right_shift_struct.h
 * \brief right_shift_struct
 */

#ifndef OPS_MATH_RIGHT_SHIFT_STRUCT_H
#define OPS_MATH_RIGHT_SHIFT_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "atvoss/broadcast/broadcast_base_struct.h"

namespace RightShiftOp {
#define TPL_INT8 1
#define TPL_UINT8 2
#define TPL_INT16 3
#define TPL_UINT16 4
#define TPL_INT32 5
#define TPL_UINT32 6
#define TPL_INT64 7
#define TPL_UINT64 8

#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    RightShift, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
    ASCENDC_TPL_DTYPE_DECL(
        dType, TPL_INT8, TPL_UINT8, TPL_INT16, TPL_UINT16, TPL_INT32, TPL_UINT32, TPL_INT64, TPL_UINT64));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_INT8)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_UINT8)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_INT16)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_UINT16)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_INT32)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_UINT32)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_INT64)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, TPL_UINT64)));
} // namespace RightShiftOp

#endif // OPS_MATH_RIGHT_SHIFT_STRUCT_H
