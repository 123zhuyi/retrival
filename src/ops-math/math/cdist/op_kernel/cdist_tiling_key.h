/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_tiling_key.h
 * \brief cdist tiling key declare
 */

#ifndef __CDIST_TILING_KEY_H__
#define __CDIST_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// KERNEL_MODE: 0=Normal(M>256) / 1=SIMT(M<=256 && M>1) / 2=Broadcast(M==1 快路径, 方案2)
//            / 3=ReduceHL(M∈[2,256] 矢量胜: broadcast + 高层 ReduceRepeat reduce, 方案三 round2 crossover)。
// 原为 ASCENDC_TPL_BOOL_DECL(IS_SMALL_M,0,1)，为新增 Broadcast/ReduceHL 路径扩为 4 取值 UINT 轴；
// 值 0/1 与原 IS_SMALL_M 的 0(Normal)/1(SIMT) 语义完全对应，新增值 2/3 为优化路径标记。
ASCENDC_TPL_ARGS_DECL(Cdist, ASCENDC_TPL_UINT_DECL(KERNEL_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KERNEL_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)));

#endif
