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
 * \file round_struct.h
 * \brief round_struct
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef _ROUND_STRUCT_H_
#define _ROUND_STRUCT_H_

namespace RoundOp
{
#define ROUND_TPL_INT32 1
#define ROUND_TPL_ZERO 2
#define ROUND_TPL_POSITIVE_DECIMALS 3
#define ROUND_TPL_NEGATIVE_DECIMALS 4
#define ROUND_TPL_NAN_DECIMALS 5
#define ROUND_TPL_INT32_CONST 6
#define ROUND_TPL_INT32_NEG 7
#define ROUND_TPL_INT32_NEGINF 8
#define ROUND_TPL_INT32_NEG_NINE 9

#define ROUND_TPL_SCH_MODE_0 0
#define ROUND_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Round,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ROUND_TPL_SCH_MODE_0, ROUND_TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, ROUND_TPL_INT32, ROUND_TPL_ZERO, ROUND_TPL_POSITIVE_DECIMALS,
                           ROUND_TPL_NEGATIVE_DECIMALS, ROUND_TPL_NAN_DECIMALS, ROUND_TPL_INT32_CONST,
                           ROUND_TPL_INT32_NEG, ROUND_TPL_INT32_NEGINF, ROUND_TPL_INT32_NEG_NINE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ROUND_TPL_SCH_MODE_0, ROUND_TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, ROUND_TPL_INT32, ROUND_TPL_ZERO, ROUND_TPL_POSITIVE_DECIMALS,
                           ROUND_TPL_NEGATIVE_DECIMALS, ROUND_TPL_NAN_DECIMALS, ROUND_TPL_INT32_CONST,
                           ROUND_TPL_INT32_NEG, ROUND_TPL_INT32_NEGINF, ROUND_TPL_INT32_NEG_NINE)
    )
);
}  // namespace RoundOp

#endif  // _ROUND_STRUCT_H_