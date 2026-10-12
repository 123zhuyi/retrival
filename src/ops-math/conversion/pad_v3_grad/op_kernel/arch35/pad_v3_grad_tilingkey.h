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
 * \file pad_v3_grad_tilingkey.h
 * \brief
 */

#ifndef _PAD_V3_GRAD_TILING_KEY_H_
#define _PAD_V3_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_CONSTANT 0
#define TPL_MODE_EDGE 1
#define TPL_MODE_REFLECT 2
#define TPL_MODE_SYMMETRIC 3
#define TPL_MODE_CIRCULAR 4

#define TPL_SIMD_BIG 0
#define TPL_SIMD_NORMAL 1
#define TPL_SIMD_SMALL 2

ASCENDC_TPL_ARGS_DECL(pad_v3_grad,
                      // 哪种模式
                      ASCENDC_TPL_UINT_DECL(ModeName, 3, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      // 是否为大shape
                      ASCENDC_TPL_BOOL_DECL(IsBigShape, 0, 1),
                      // 是否为simt
                      ASCENDC_TPL_BOOL_DECL(IsSIMT, 0, 1),
                      // 切分模式
                      ASCENDC_TPL_UINT_DECL(CutMode, 2, ASCENDC_TPL_UI_RANGE, 1, 0, 2), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
                    // 模式选择
                    ASCENDC_TPL_UINT_SEL(ModeName, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                    // 大小选择
                    ASCENDC_TPL_BOOL_SEL(IsBigShape, 0, 1),
                    // simt
                    ASCENDC_TPL_BOOL_SEL(IsSIMT, 1),
                    // 切分模式 忽略
                    ASCENDC_TPL_UINT_SEL(CutMode, ASCENDC_TPL_UI_RANGE, 1, 0, 2), ),

                ASCENDC_TPL_ARGS_SEL(
                    // 模式选择 当前只支持TPL_MODE_REFLECT和TPL_MODE_SYMMETRIC
                    ASCENDC_TPL_UINT_SEL(ModeName, ASCENDC_TPL_UI_RANGE, 1, 2, 4),
                    // 大小选择 忽略
                    ASCENDC_TPL_BOOL_SEL(IsBigShape, 0, 1),
                    // simt
                    ASCENDC_TPL_BOOL_SEL(IsSIMT, 0),
                    // 切分模式
                    ASCENDC_TPL_UINT_SEL(CutMode, ASCENDC_TPL_UI_RANGE, 1, 0, 2), ), );
#endif
