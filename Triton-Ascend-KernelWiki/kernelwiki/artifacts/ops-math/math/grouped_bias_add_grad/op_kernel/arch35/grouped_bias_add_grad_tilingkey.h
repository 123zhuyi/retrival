/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file GROUPED_BIAS_ADD_grad_tiling_key.h
\brief repeat interleave grad tiling key
*/
#ifndef GROUPED_BIAS_ADD_GRAD_TILING_KEY_H
#define GROUPED_BIAS_ADD_GRAD_TILING_KEY_H

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define GROUPED_BIAS_ADD_GRAD_BIT_WIDTH 4
#define GROUPED_BIAS_ADD_GRAD_GROUP_IDX_DTYPE_BIT_WIDTH 1

ASCENDC_TPL_ARGS_DECL(
    groupedBiasAddGrad, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(TemplateNum, GROUPED_BIAS_ADD_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
    ASCENDC_TPL_UINT_DECL(GroupIdxDtype, GROUPED_BIAS_ADD_GRAD_GROUP_IDX_DTYPE_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 1));

ASCENDC_TPL_SEL(
    // empty
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_RANGE, 1, 0, 1)),
    // A
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_LIST, 0)),
    // AR
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_AR_NORMAL(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_AR_GROUP(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_LIST, 0));
    // ARA
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(GroupIdxDtype, ASCENDC_TPL_UI_LIST, 0)));

#endif