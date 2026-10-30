/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 */

#ifndef __FRESNEL_SIN_TILING_KEY_H__
#define __FRESNEL_SIN_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// TilingKey values:
//   0 = fp32 direct compute
//   1 = fp16, Cast to fp32 internally
//   2 = bf16, Cast to fp32 internally
#define FRESNEL_SIN_KEY_FP32 0
#define FRESNEL_SIN_KEY_FP16 1
#define FRESNEL_SIN_KEY_BF16 2

ASCENDC_TPL_ARGS_DECL(
    FresnelSin, ASCENDC_TPL_UINT_DECL(
                    schMode, 8, ASCENDC_TPL_UI_LIST, FRESNEL_SIN_KEY_FP32, FRESNEL_SIN_KEY_FP16, FRESNEL_SIN_KEY_BF16));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(
        schMode, ASCENDC_TPL_UI_LIST, FRESNEL_SIN_KEY_FP32, FRESNEL_SIN_KEY_FP16, FRESNEL_SIN_KEY_BF16)), );

#endif // __FRESNEL_SIN_TILING_KEY_H__
