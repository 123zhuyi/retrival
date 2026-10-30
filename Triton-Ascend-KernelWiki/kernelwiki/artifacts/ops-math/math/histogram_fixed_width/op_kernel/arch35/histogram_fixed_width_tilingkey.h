/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HISTOGRAM_FIXED_WIDTH_TILINGKEY_H_
#define _HISTOGRAM_FIXED_WIDTH_TILINGKEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// Tiling key template args for HistogramFixedWidth
// LoadMode: 0=UB_FULL, 1=UB_NOT_FULL, 2=UB_NOT_FULL_SIMT
// Input dtype is determined by DTYPE_X at compile time, no need for tiling key.

ASCENDC_TPL_ARGS_DECL(HistogramFixedWidth, ASCENDC_TPL_UINT_DECL(LoadMode, 2, ASCENDC_TPL_UI_RANGE, 1, 0, 2), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(LoadMode, ASCENDC_TPL_UI_RANGE, 1, 0, 2), ), );

constexpr uint64_t TPL_LOAD_MODE_UB_FULL = 0;
constexpr uint64_t TPL_LOAD_MODE_UB_NOT_FULL = 1;
constexpr uint64_t TPL_LOAD_MODE_UB_NOT_FULL_SIMT = 2;

#endif // _HISTOGRAM_FIXED_WIDTH_TILINGKEY_H_
