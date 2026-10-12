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
 * \file pad_v3_grad_replication_tilingkey.h
 * \brief tiling key definition for pad_v3_grad_replication
 */

#ifndef _PAD_V3_GRAD_REPLICATION_TILING_KEY_H_
#define _PAD_V3_GRAD_REPLICATION_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// TilingKey 编码（须与 ASCENDC FastEncodeTilingKeyDirect 一致）：
//   字段按 ASCENDC_TPL_ARGS_DECL 声明顺序拼接，先声明的占低位；UINT 字段编的是 vals 数组的 index，
//   不是原始 value。
//   DimNum   声明 RANGE [1..8] → index = value - 1，占低 4 bit；
//   SplitAxis 声明 RANGE [0..7] → index = value，    占高 4 bit。
//   → tilingKey = (SplitAxis << 4) | (DimNum - 1)
// 示例：N=3、axis=0 → (0<<4)|(3-1) = 2；N=4、axis=2 → (2<<4)|(4-1) = 35。

ASCENDC_TPL_ARGS_DECL(pad_v3_grad_replication, ASCENDC_TPL_UINT_DECL(DimNum, 4, ASCENDC_TPL_UI_RANGE, 1, 1, 8),
                      ASCENDC_TPL_UINT_DECL(SplitAxis, 4, ASCENDC_TPL_UI_RANGE, 1, 0, 7), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DimNum, ASCENDC_TPL_UI_RANGE, 1, 1, 8),
                                     ASCENDC_TPL_UINT_SEL(SplitAxis, ASCENDC_TPL_UI_RANGE, 1, 0, 7), ), );

#endif // _PAD_V3_GRAD_REPLICATION_TILING_KEY_H_
