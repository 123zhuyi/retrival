/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BATCH_TO_SPACE_TILING_KEY_H_
#define _BATCH_TO_SPACE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// ubAxis 取值 0(N), 1(H), 2(W), 3(C)
// UI_RANGE 生成统一模板函数 batch_to_space<UbAxis>，UI_LIST 会为每个值生成独立函数
ASCENDC_TPL_ARGS_DECL(BatchToSpace, ASCENDC_TPL_UINT_DECL(ubAxis, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 3), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ubAxis, ASCENDC_TPL_UI_RANGE, 1, 0, 3), ), );

#endif // _BATCH_TO_SPACE_TILING_KEY_H_
