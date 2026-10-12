/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file lin_space_d_tiling_key.h
 * \brief LinSpaceD operator tiling template parameter declaration
 */

#ifndef __LIN_SPACE_D_TILING_KEY_H__
#define __LIN_SPACE_D_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define LIN_SPACE_D_TPL_BF16 27 // bfloat16
#define LIN_SPACE_D_TPL_UINT8 4 // uint8
#define LIN_SPACE_D_TPL_FP16 1  // float16
#define LIN_SPACE_D_TPL_FP32 0  // float32
#define LIN_SPACE_D_TPL_INT8 2  // int8
#define LIN_SPACE_D_TPL_INT16 6 // int16
#define LIN_SPACE_D_TPL_INT32 3 // int32

ASCENDC_TPL_ARGS_DECL(
    LinSpaceD,
    ASCENDC_TPL_DTYPE_DECL(
        TYPE_START, LIN_SPACE_D_TPL_BF16, LIN_SPACE_D_TPL_UINT8, LIN_SPACE_D_TPL_FP16, LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8, LIN_SPACE_D_TPL_INT16, LIN_SPACE_D_TPL_INT32),
    ASCENDC_TPL_DTYPE_DECL(
        TYPE_END, LIN_SPACE_D_TPL_BF16, LIN_SPACE_D_TPL_UINT8, LIN_SPACE_D_TPL_FP16, LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8, LIN_SPACE_D_TPL_INT16, LIN_SPACE_D_TPL_INT32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(
        TYPE_START, LIN_SPACE_D_TPL_BF16, LIN_SPACE_D_TPL_UINT8, LIN_SPACE_D_TPL_FP16, LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8, LIN_SPACE_D_TPL_INT16, LIN_SPACE_D_TPL_INT32),
    ASCENDC_TPL_DTYPE_SEL(
        TYPE_END, LIN_SPACE_D_TPL_BF16, LIN_SPACE_D_TPL_UINT8, LIN_SPACE_D_TPL_FP16, LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8, LIN_SPACE_D_TPL_INT16, LIN_SPACE_D_TPL_INT32)));

#endif // __LIN_SPACE_D_TILING_KEY_H__