/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BATCH_TO_SPACE_N_D_TILING_KEY_H_
#define _BATCH_TO_SPACE_N_D_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_SIMT 0
#define TPL_MODE_LARGE_C 1
#define TPL_MODE_SMALL_C 2

ASCENDC_TPL_ARGS_DECL(BatchToSpaceND,
                      // data format
                      ASCENDC_TPL_FORMAT_DECL(mode, TPL_MODE_SIMT, TPL_MODE_LARGE_C, TPL_MODE_SMALL_C),
                      // block_shape 的维度
                      ASCENDC_TPL_UINT_DECL(blockShapeDimNum, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      // 是否为大shape
                      ASCENDC_TPL_BOOL_DECL(isBigShape, 0, 1), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
                    // SIMT
                    ASCENDC_TPL_FORMAT_SEL(mode, TPL_MODE_SIMT),
                    // 忽略Small C参数
                    ASCENDC_TPL_UINT_SEL(blockShapeDimNum, ASCENDC_TPL_UI_LIST, 0),
                    // SIMT 参数
                    ASCENDC_TPL_BOOL_SEL(isBigShape, 0, 1), ),
                ASCENDC_TPL_ARGS_SEL(
                    // Large C
                    ASCENDC_TPL_FORMAT_SEL(mode, TPL_MODE_LARGE_C),
                    // 忽略Small C参数
                    ASCENDC_TPL_UINT_SEL(blockShapeDimNum, ASCENDC_TPL_UI_LIST, 0),
                    // 忽略SIMT参数
                    ASCENDC_TPL_BOOL_SEL(isBigShape, 0), ),
                ASCENDC_TPL_ARGS_SEL(
                    // Small C
                    ASCENDC_TPL_FORMAT_SEL(mode, TPL_MODE_SMALL_C),
                    // Small C 参数
                    ASCENDC_TPL_UINT_SEL(blockShapeDimNum, ASCENDC_TPL_UI_RANGE, 1, 1, 3),
                    // 忽略 SIMT 参数
                    ASCENDC_TPL_BOOL_SEL(isBigShape, 0), ), );

#endif // _BATCH_TO_SPACE_N_D_TILING_KEY_H_
