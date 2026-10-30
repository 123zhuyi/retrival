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
 * \file mod_tiling_key.h
 * \brief mod tiling key declare
 */
#ifndef MOD_TILING_KEY_H
#define MOD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace ModNs {

#define MOD_TPL_BF16 10
#define MOD_TPL_FP16 20
#define MOD_TPL_FP32 30
#define MOD_TPL_INT32 40

ASCENDC_TPL_ARGS_DECL(
    Mod,
    ASCENDC_TPL_DTYPE_DECL(D_T_X1, MOD_TPL_INT32, MOD_TPL_FP16, MOD_TPL_BF16, MOD_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_X2, MOD_TPL_INT32, MOD_TPL_FP16, MOD_TPL_BF16, MOD_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_Y, MOD_TPL_INT32, MOD_TPL_FP16, MOD_TPL_BF16, MOD_TPL_FP32), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X1, MOD_TPL_INT32), ASCENDC_TPL_DTYPE_SEL(D_T_X2, MOD_TPL_INT32),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, MOD_TPL_INT32), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X1, MOD_TPL_FP16), ASCENDC_TPL_DTYPE_SEL(D_T_X2, MOD_TPL_FP16),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, MOD_TPL_FP16), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X1, MOD_TPL_FP32), ASCENDC_TPL_DTYPE_SEL(D_T_X2, MOD_TPL_FP32),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, MOD_TPL_FP32), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X1, MOD_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(D_T_X2, MOD_TPL_BF16),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, MOD_TPL_BF16), ));

} // namespace ModNs

#endif // MOD_TILING_KEY_H