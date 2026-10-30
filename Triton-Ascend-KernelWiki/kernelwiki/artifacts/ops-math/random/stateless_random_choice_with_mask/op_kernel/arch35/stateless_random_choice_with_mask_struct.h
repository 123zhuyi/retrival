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
 * \file stateless_random_choice_with_mask_struct.h
 * \brief
 */

#include "ascendc/host_api/tiling/template_argument.h"
#ifndef CANN_CUSTOM_OPS_STATELESS_RANDOM_CHOICE_WITH_MASK_STRUCT_H_
#define CANN_CUSTOM_OPS_STATELESS_RANDOM_CHOICE_WITH_MASK_STRUCT_H_

#define TPL_SCH_MODE_0 0
#define TILING_ARRAY_LEN_EIGHT 8

class StatelessRandomChoiceWithMaskSimtTilingData {
public:
    int64_t blockNum = 0;
    int64_t normalCoreProNum = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t inputSize = 0;
    int64_t noZeroCalcCount = 0;
    int64_t noZeroWorkspaceSize = 0;
    int64_t randomWorkspaceSize = 0;
    int64_t ubSize = 0;
    int32_t count = 0;
    uint32_t inputDim = 0;
    uint32_t inputShape[TILING_ARRAY_LEN_EIGHT] = {0};
};

ASCENDC_TPL_ARGS_DECL(
    StatelessRandomChoiceWithMask, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0)));
#endif // CANN_CUSTOM_OPS_STATELESS_RANDOM_CHOICE_WITH_MASK_STRUCT_H_