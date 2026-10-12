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
 * \file diag_v2_tiling_key.h
 * \brief DiagV2 TilingKey template parameter definition (arch35)
 *
 * Design: DESIGN.md v2.4 Sec 3.1.2
 *
 * IS_1D_INPUT = 0: 2D→1D diagonal extraction (DiagV2Simd)
 * IS_1D_INPUT = 1: 1D→2D diagonal matrix construction (DiagFlatSimd)
 */

#ifndef __DIAG_V2_ARCH35_TILING_KEY_H__
#define __DIAG_V2_ARCH35_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(DiagV2,
    ASCENDC_TPL_UINT_DECL(IS_1D_INPUT, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(IS_1D_INPUT, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);

#endif // __DIAG_V2_ARCH35_TILING_KEY_H__
