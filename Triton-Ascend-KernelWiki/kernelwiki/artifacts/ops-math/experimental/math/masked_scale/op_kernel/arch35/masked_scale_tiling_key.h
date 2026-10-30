// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#ifndef MASKED_SCALE_TILING_KEY_H
#define MASKED_SCALE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define MASKED_SCALE_TPL_FP16 1
#define MASKED_SCALE_TPL_FP32 0
#define MASKED_SCALE_TPL_BF16 27
#define MASKED_SCALE_TPL_UINT8 4
#define MASKED_SCALE_TPL_INT8 2
#define MASKED_SCALE_KEY_FP16_UINT8 10
#define MASKED_SCALE_KEY_FP16_INT8 11
#define MASKED_SCALE_KEY_FP16_FP16 12
#define MASKED_SCALE_KEY_FP16_FP32 13
#define MASKED_SCALE_KEY_FP32_UINT8 20
#define MASKED_SCALE_KEY_FP32_INT8 21
#define MASKED_SCALE_KEY_FP32_FP16 22
#define MASKED_SCALE_KEY_FP32_FP32 23
#define MASKED_SCALE_KEY_BF16_UINT8 30
#define MASKED_SCALE_KEY_BF16_INT8 31
#define MASKED_SCALE_KEY_BF16_FP16 32
#define MASKED_SCALE_KEY_BF16_FP32 33

ASCENDC_TPL_ARGS_DECL(MaskedScale,
                      ASCENDC_TPL_DTYPE_DECL(TYPE_X, MASKED_SCALE_TPL_FP16, MASKED_SCALE_TPL_FP32,
                                             MASKED_SCALE_TPL_BF16),
                      ASCENDC_TPL_DTYPE_DECL(TYPE_MASK, MASKED_SCALE_TPL_UINT8, MASKED_SCALE_TPL_INT8,
                                             MASKED_SCALE_TPL_FP16, MASKED_SCALE_TPL_FP32), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_UINT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_INT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP32), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_UINT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_INT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP32), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_UINT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_INT8), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(TYPE_X, MASKED_SCALE_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(TYPE_MASK, MASKED_SCALE_TPL_FP32), ));

#endif // MASKED_SCALE_TILING_KEY_H
