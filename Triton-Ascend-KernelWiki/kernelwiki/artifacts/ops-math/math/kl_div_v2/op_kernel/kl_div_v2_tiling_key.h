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
 * \file kl_div_v2_tiling_key.h
 * \brief kl_div_v2_tiling_key
 */

#ifndef KL_DIV_TILING_KEY_H
#define KL_DIV_TILING_KEY_H

#include "atvoss/reduce/reduce_tiling_key_decl.h"

ASCENDC_TPL_ARGS_DECL(KLDivV2, REDUCE_TPL_KEY_DECL(),
                ASCENDC_TPL_UINT_DECL(reduction, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                ASCENDC_TPL_UINT_DECL(logTarget, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 1));

ASCENDC_TPL_SEL(
    // empty
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(),
                ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                ASCENDC_TPL_UINT_SEL(logTarget, ASCENDC_TPL_UI_RANGE, 1, 0, 1)),
    // A
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(),
                ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                ASCENDC_TPL_UINT_SEL(logTarget, ASCENDC_TPL_UI_RANGE, 1, 0, 1)),
    // AR
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_NORMAL(),
                ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                ASCENDC_TPL_UINT_SEL(logTarget, ASCENDC_TPL_UI_RANGE, 1, 0, 1)),

    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_GROUP(),
                ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                ASCENDC_TPL_UINT_SEL(logTarget, ASCENDC_TPL_UI_RANGE, 1, 0, 1))
);

#endif