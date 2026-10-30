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
 * \file random_uniform_int_v2_struct.h
 * \brief tiling base data
 */

#ifndef Random_Uniform_Int_V2_STRUCT_H
#define Random_Uniform_Int_V2_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

class RandomUniformIntV2TilingData4RegBase {
public:
    int64_t blockNum = 0;
    int64_t normalCoreProNum = 0;
    int64_t tailCoreProNum = 0;
    int64_t singleUbSize = 0;
    int64_t seed = 0;
    int64_t seed2 = 0;
    int64_t outputSize = 0;
    uint64_t range = 0;
    int64_t lo = 0;
};

#define RANDOM_UNIFORM_INT_V2_TPL 0

ASCENDC_TPL_ARGS_DECL(RandomUniformIntV2,
    ASCENDC_TPL_UINT_DECL(opType, 1, ASCENDC_TPL_UI_LIST, RANDOM_UNIFORM_INT_V2_TPL)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(opType, ASCENDC_TPL_UI_LIST, RANDOM_UNIFORM_INT_V2_TPL)));

#endif // Random_Uniform_Int_V2_STRUCT_H