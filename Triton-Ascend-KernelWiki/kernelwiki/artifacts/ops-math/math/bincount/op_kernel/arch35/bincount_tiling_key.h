/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bincount_tiling_key.h
 * \brief bincount tiling key declare
 */

#ifndef _BINCOUNT_TILING_KEY_DECL_H_
#define _BINCOUNT_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCH_ID_FULL_LOAD 0     // full load to UB
#define TPL_SCH_ID_BATCH_LOAD 1    // batch load to UB
#define TPL_SCH_ID_NOT_FULL_LOAD 2 // not load to UB
#define TPL_SCH_ID_DETERMIN 3      // determin

#define TPL_OUTPUT_DTYPE_FLOAT 0
#define TPL_OUTPUT_DTYPE_INT32 1
#define TPL_OUTPUT_DTYPE_INT64 2

#define TPL_EMPTY_WEIGHT 0
#define TPL_HAS_WEIGHT 1

#define BINCOUNT_TPL_KEY_DECL()                                                                                 \
    ASCENDC_TPL_UINT_DECL(                                                                                      \
        schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_FULL_LOAD, TPL_SCH_ID_BATCH_LOAD,              \
        TPL_SCH_ID_NOT_FULL_LOAD, TPL_SCH_ID_DETERMIN),                                                         \
        ASCENDC_TPL_UINT_DECL(                                                                                  \
            outputDtype, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_OUTPUT_DTYPE_FLOAT, TPL_OUTPUT_DTYPE_INT32, \
            TPL_OUTPUT_DTYPE_INT64),                                                                            \
        ASCENDC_TPL_UINT_DECL(isWeight, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),

#define BINCOUNT_TPL_KEY_SEL()                                                                                         \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_RANGE, 1, 0, TPL_SCH_ID_DETERMIN),                                      \
        ASCENDC_TPL_UINT_SEL(                                                                                          \
            outputDtype, ASCENDC_TPL_UI_LIST, TPL_OUTPUT_DTYPE_FLOAT, TPL_OUTPUT_DTYPE_INT32, TPL_OUTPUT_DTYPE_INT64), \
        ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, TPL_EMPTY_WEIGHT, TPL_HAS_WEIGHT),

ASCENDC_TPL_ARGS_DECL(Bincount, BINCOUNT_TPL_KEY_DECL());

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BINCOUNT_TPL_KEY_SEL()));
#endif