/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SPACE_TO_BATCH_TILING_KEY_H_
#define _SPACE_TO_BATCH_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_UB_AXIS_N 0
#define TPL_UB_AXIS_H 1
#define TPL_UB_AXIS_W 2
#define TPL_UB_AXIS_C 3

ASCENDC_TPL_ARGS_DECL(SpaceToBatch,
                      ASCENDC_TPL_FORMAT_DECL(ubAxis, TPL_UB_AXIS_N, TPL_UB_AXIS_H, TPL_UB_AXIS_W, TPL_UB_AXIS_C), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_FORMAT_SEL(ubAxis, TPL_UB_AXIS_N), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_FORMAT_SEL(ubAxis, TPL_UB_AXIS_H), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_FORMAT_SEL(ubAxis, TPL_UB_AXIS_W), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_FORMAT_SEL(ubAxis, TPL_UB_AXIS_C), ), );

#endif // _SPACE_TO_BATCH_TILING_KEY_H_
