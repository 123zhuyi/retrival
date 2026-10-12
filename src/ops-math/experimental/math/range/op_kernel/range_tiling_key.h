/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file range.cpp
 * \brief
 */
#ifndef __RANGE_TILING_KEY_H__
#define __RANGE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define RANGE_TPL_FP32 0 // float32

ASCENDC_TPL_ARGS_DECL(
    Range, ASCENDC_TPL_DTYPE_DECL(TYPE_START, RANGE_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(
        TYPE_END, RANGE_TPL_FP32

        ),
    ASCENDC_TPL_DTYPE_DECL(
        TYPE_STEP, RANGE_TPL_FP32

        )

);

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(TYPE_START, RANGE_TPL_FP32), ASCENDC_TPL_DTYPE_SEL(TYPE_END, RANGE_TPL_FP32),
    ASCENDC_TPL_DTYPE_SEL(
        TYPE_STEP, RANGE_TPL_FP32

        )));

#endif // __RANGE_TILING_KEY_H__