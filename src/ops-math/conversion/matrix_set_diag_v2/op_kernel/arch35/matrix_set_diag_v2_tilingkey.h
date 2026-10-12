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
 * \file matrix_set_diag_v2_tilingkey.h
 * \brief
 */

#ifndef _MATRIX_SET_DIAG_V2_TILING_KEY_H_
#define _MATRIX_SET_DIAG_V2_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_WAY_DEFAULT 0
#define TPL_WAY_GATHER 1
#define TPL_WAY_SCATTER 2
#define TPL_WAY_SIMT 3
#define TPL_WAY_V1 4

ASCENDC_TPL_ARGS_DECL(MatrixSetDiagV2,
                      // 实现路径
                      ASCENDC_TPL_UINT_DECL(Way, 8, ASCENDC_TPL_UI_LIST, TPL_WAY_DEFAULT, TPL_WAY_GATHER,
                                            TPL_WAY_SCATTER, TPL_WAY_SIMT, TPL_WAY_V1),
                      // VL是否满载
                      ASCENDC_TPL_BOOL_DECL(IsVLFullLoad, 0, 1),
                      // 是否为大shape
                      ASCENDC_TPL_BOOL_DECL(isBigShape, 0, 1),
                      // 是否切尾轴
                      ASCENDC_TPL_BOOL_DECL(IsCutTail, 0, 1), );

ASCENDC_TPL_SEL(
    // V1 切尾轴
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(Way, ASCENDC_TPL_UI_LIST, TPL_WAY_V1),
                         ASCENDC_TPL_BOOL_SEL(IsVLFullLoad, 0), ASCENDC_TPL_BOOL_SEL(isBigShape, 0),
                         ASCENDC_TPL_BOOL_SEL(IsCutTail, 1), ),
    // V2 切尾轴
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(Way, ASCENDC_TPL_UI_LIST, TPL_WAY_DEFAULT),
                         ASCENDC_TPL_BOOL_SEL(IsVLFullLoad, 0),
                         // SIMT 参数
                         ASCENDC_TPL_BOOL_SEL(isBigShape, 0, 1), ASCENDC_TPL_BOOL_SEL(IsCutTail, 1), ),
    // 非尾轴
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(Way, ASCENDC_TPL_UI_LIST, TPL_WAY_GATHER, TPL_WAY_SCATTER, TPL_WAY_SIMT),
                         ASCENDC_TPL_BOOL_SEL(IsVLFullLoad, 0, 1),
                         // SIMT 参数
                         ASCENDC_TPL_BOOL_SEL(isBigShape, 0), ASCENDC_TPL_BOOL_SEL(IsCutTail, 0), ), );

#endif
