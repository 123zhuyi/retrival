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
 * \file zeros_like_tiling_key.h
 * \brief ascend910b TilingKey 声明（按字节宽度 4 桶：1/2/4/8）。
 *        采用整型模板实参机制 ASCENDC_TPL_UINT_DECL（非 DTYPE_DECL），与 host SetTilingKey、
 *        kernel template<int BYTE_KEY> + if constexpr 一一映射；binary DtypeByte 4 桶各自命中一个实例。
 */
#ifndef ZEROS_LIKE_TILING_KEY_H
#define ZEROS_LIKE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 字节宽度 TilingKey 取值（直接用字节数，便于与 DtypeByte 桶对齐、可读性高）
#define ZL_KEY_1B 1
#define ZL_KEY_2B 2
#define ZL_KEY_4B 4
#define ZL_KEY_8B 8

ASCENDC_TPL_ARGS_DECL(
    ZerosLike, ASCENDC_TPL_UINT_DECL(byteKey, 4, ASCENDC_TPL_UI_LIST, ZL_KEY_1B, ZL_KEY_2B, ZL_KEY_4B, ZL_KEY_8B));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(byteKey, ASCENDC_TPL_UI_LIST, ZL_KEY_1B, ZL_KEY_2B, ZL_KEY_4B, ZL_KEY_8B)));

#endif // ZEROS_LIKE_TILING_KEY_H
