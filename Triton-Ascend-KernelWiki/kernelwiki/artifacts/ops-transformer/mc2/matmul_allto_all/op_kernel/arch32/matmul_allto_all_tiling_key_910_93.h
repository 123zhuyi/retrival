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
 * \file matmul_allto_all_tiling_key_910_93.h
 * \brief 定义tiling_key
 */
#ifndef MATMUL_ALLTO_ALL_TILING_KEY_910_93_H
#define MATMUL_ALLTO_ALL_TILING_KEY_910_93_H

#include <ascendc/host_api/tiling/template_argument.h>

// 量化组合模式
#define NON_QUANT_MODE 0

// bias的数据类型
#define DTYPE_BIAS_SAME_WITH_X 0
#define DTYPE_BIAS_FP32 1

// 模板参数范围声明
ASCENDC_TPL_ARGS_DECL(MatmulAlltoAllA3,
                      ASCENDC_TPL_UINT_DECL(QUANTMODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                      ASCENDC_TPL_BOOL_DECL(X2TRANSPOSE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(DTYPEBIAS, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X, DTYPE_BIAS_FP32), );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 0),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_FP32), ),
                                     
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_FP32), ),
                                     
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 0),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X), ),
                                     
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X), )
);                                                          

#endif // MATMUL_ALLTO_ALL_TILING_KEY_910_93_H