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
 * \file squared_difference_tiling_key.h
 * \brief Tiling 模板参数定义
 *        schMode 编码：dtypeKey*2 + mode  (10 tilingKeys = 5 dtype x 2 mode)
 *        0: fp32  OneDim  1: fp32  BRC
 *        2: fp16  OneDim  3: fp16  BRC
 *        4: bf16  OneDim  5: bf16  BRC
 *        6: int32 OneDim  7: int32 BRC
 *        8: int64 OneDim  9: int64 BRC
 */

#ifndef __SQUAREDDIFFERENCE_TILING_KEY_H__
#define __SQUAREDDIFFERENCE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SD_KEY_FP32_ONEDIM 0
#define SD_KEY_FP32_BRC 1
#define SD_KEY_FP16_ONEDIM 2
#define SD_KEY_FP16_BRC 3
#define SD_KEY_BF16_ONEDIM 4
#define SD_KEY_BF16_BRC 5
#define SD_KEY_INT32_ONEDIM 6
#define SD_KEY_INT32_BRC 7
#define SD_KEY_INT64_ONEDIM 8
#define SD_KEY_INT64_BRC 9

ASCENDC_TPL_ARGS_DECL(SquaredDifference,
                      ASCENDC_TPL_UINT_DECL(schMode, 4, ASCENDC_TPL_UI_LIST, SD_KEY_FP32_ONEDIM, SD_KEY_FP32_BRC,
                                            SD_KEY_FP16_ONEDIM, SD_KEY_FP16_BRC, SD_KEY_BF16_ONEDIM, SD_KEY_BF16_BRC,
                                            SD_KEY_INT32_ONEDIM, SD_KEY_INT32_BRC, SD_KEY_INT64_ONEDIM,
                                            SD_KEY_INT64_BRC));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SD_KEY_FP32_ONEDIM,
                                                          SD_KEY_FP32_BRC, SD_KEY_FP16_ONEDIM, SD_KEY_FP16_BRC,
                                                          SD_KEY_BF16_ONEDIM, SD_KEY_BF16_BRC, SD_KEY_INT32_ONEDIM,
                                                          SD_KEY_INT32_BRC, SD_KEY_INT64_ONEDIM, SD_KEY_INT64_BRC)));

#endif
