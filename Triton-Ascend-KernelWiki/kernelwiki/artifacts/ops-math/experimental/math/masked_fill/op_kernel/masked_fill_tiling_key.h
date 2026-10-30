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
 * \file masked_fill_tiling_key.h
 * \brief masked_fill tiling key declare
 */

#ifndef __MASKED_FILL_TILING_KEY_H__
#define __MASKED_FILL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MASKED_FILL_TPL_BF16 27   // bfloat16
#define MASKED_FILL_TPL_FP16 1    // float16
#define MASKED_FILL_TPL_FP32 0    // float32
#define MASKED_FILL_TPL_FP64 11   // float64
#define MASKED_FILL_TPL_INT8 2    // int8
#define MASKED_FILL_TPL_INT16 6   // int16
#define MASKED_FILL_TPL_INT32 3   // int32
#define MASKED_FILL_TPL_INT64 9   // int64
#define MASKED_FILL_TPL_UINT8 4   // uint8
#define MASKED_FILL_TPL_UINT16 7  // uint16
#define MASKED_FILL_TPL_UINT32 8  // uint32
#define MASKED_FILL_TPL_UINT64 10 // uint64

ASCENDC_TPL_ARGS_DECL(
    MaskedFill, ASCENDC_TPL_DTYPE_DECL(
                    TYPE_X, MASKED_FILL_TPL_BF16, MASKED_FILL_TPL_UINT8, MASKED_FILL_TPL_FP16, MASKED_FILL_TPL_FP32,
                    MASKED_FILL_TPL_INT8, MASKED_FILL_TPL_INT16, MASKED_FILL_TPL_INT32, MASKED_FILL_TPL_FP64,
                    MASKED_FILL_TPL_UINT16, MASKED_FILL_TPL_UINT32, MASKED_FILL_TPL_UINT64, MASKED_FILL_TPL_INT64));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(
    TYPE_X, MASKED_FILL_TPL_BF16, MASKED_FILL_TPL_UINT8, MASKED_FILL_TPL_FP16, MASKED_FILL_TPL_FP32,
    MASKED_FILL_TPL_INT8, MASKED_FILL_TPL_INT16, MASKED_FILL_TPL_INT32, MASKED_FILL_TPL_FP64, MASKED_FILL_TPL_UINT16,
    MASKED_FILL_TPL_UINT32, MASKED_FILL_TPL_UINT64, MASKED_FILL_TPL_INT64)));

#endif // __MASKED_FILL_TILING_KEY_H__
