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
 * \file right_shift_tiling_key.h
 * \brief RightShift tiling key declare
 */

#ifndef RIGHT_SHIFT_TILING_KEY_H
#define RIGHT_SHIFT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "right_shift_tiling_data.h"

#define RIGHT_SHIFT_TPL_KEY_BW 6

ASCENDC_TPL_ARGS_DECL(RightShift,
                      ASCENDC_TPL_UINT_DECL(RIGHT_SHIFT_TPL_KEY, RIGHT_SHIFT_TPL_KEY_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2,
                                            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RIGHT_SHIFT_TPL_KEY, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6,
                                                          7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                          23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                                                          38, 39), ), );

#endif // RIGHT_SHIFT_TILING_KEY_H
