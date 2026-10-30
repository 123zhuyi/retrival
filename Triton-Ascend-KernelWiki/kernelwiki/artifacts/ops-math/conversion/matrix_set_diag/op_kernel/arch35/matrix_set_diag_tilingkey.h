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
 * \file matrix_set_diag_tilingkey.h
 * \brief
 */

#ifndef _MATRIX_SET_DIAG_TILING_KEY_H_
#define _MATRIX_SET_DIAG_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    MatrixSetDiag,
    // 是否切尾轴
    ASCENDC_TPL_BOOL_DECL(IsCutW, 0, 1),
    // 是否为simt
    ASCENDC_TPL_BOOL_DECL(IsSIMT, 0, 1), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(IsCutW, 0, 1),
        // 忽略 SIMT 参数
        ASCENDC_TPL_BOOL_SEL(IsSIMT, 0), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(IsCutW, 0),
        // 忽略 SIMT 参数
        ASCENDC_TPL_BOOL_SEL(IsSIMT, 1), ), );

#endif