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
 * \file softshrink_grad_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __SOFTSHRINKGRAD_TILING_KEY_H__
#define __SOFTSHRINKGRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SOFTSHRINKGRAD_TPL_SCH_MODE_0 0
#define SOFTSHRINKGRAD_TPL_SCH_MODE_1 1
#define SOFTSHRINKGRAD_TPL_SCH_MODE_2 2
#define SOFTSHRINKGRAD_TPL_SCH_MODE_3 3
#define SOFTSHRINKGRAD_TPL_SCH_MODE_4 4
#define SOFTSHRINKGRAD_TPL_SCH_MODE_5 5

ASCENDC_TPL_ARGS_DECL(SoftshrinkGrad,
                      ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, SOFTSHRINKGRAD_TPL_SCH_MODE_0,
                                            SOFTSHRINKGRAD_TPL_SCH_MODE_1, SOFTSHRINKGRAD_TPL_SCH_MODE_2,
                                            SOFTSHRINKGRAD_TPL_SCH_MODE_3, SOFTSHRINKGRAD_TPL_SCH_MODE_4,
                                            SOFTSHRINKGRAD_TPL_SCH_MODE_5));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SOFTSHRINKGRAD_TPL_SCH_MODE_0,
                                                          SOFTSHRINKGRAD_TPL_SCH_MODE_1, SOFTSHRINKGRAD_TPL_SCH_MODE_2,
                                                          SOFTSHRINKGRAD_TPL_SCH_MODE_3, SOFTSHRINKGRAD_TPL_SCH_MODE_4,
                                                          SOFTSHRINKGRAD_TPL_SCH_MODE_5)));

#endif
