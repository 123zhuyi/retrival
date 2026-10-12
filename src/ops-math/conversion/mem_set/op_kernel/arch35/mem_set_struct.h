/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_OPS_MEMSET_STRUCT_H
#define CANN_OPS_MEMSET_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

template <int len>
struct MemSetTilingData {
    int64_t perCoreSizes[len];
    int64_t lastCoreSizes[len];
    int64_t intValue[len];
    float floatValue[len];
    int16_t listType[len];
    int16_t useCore[len];
    int halfUbSize;
    int inputCount;
};
namespace MemSetTpl {
ASCENDC_TPL_ARGS_DECL(MemSet, ASCENDC_TPL_UINT_DECL(inputCount, 8, ASCENDC_TPL_UI_RANGE, 1, 1, 256));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(inputCount, ASCENDC_TPL_UI_RANGE, 1, 1, 256)));
//支持tensor范围为1-256
} // namespace MemSetTpl

#endif