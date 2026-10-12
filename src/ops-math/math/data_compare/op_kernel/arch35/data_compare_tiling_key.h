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
 * \file data_compare_tiling_key.h
 * \brief DataCompare TilingKey 模板参数声明（All Reduce 简化版：2 个 bool）
 *
 * All Reduce 下 isTailR 恒为 true，不进 TilingKey。
 *   templateType == 0：
 *     isEmptyTensor == 0 → normal 模板（isTailR=true 恒定）
 *     isEmptyTensor == 1 → 空 tensor 模板
 *   templateType == 1 → group 模板（isEmptyTensor 固定 0）
 *
 * dtype 走 DTYPE_X1 编译期实例化（6 种），与 tilingkey 的 3 组合相乘 ⇒ 18 份 binary。
 */
#ifndef OPS_DATA_COMPARE_TILING_KEY_H_
#define OPS_DATA_COMPARE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    DataCompare, ASCENDC_TPL_BOOL_DECL(templateType, 0, 1), // 0=normal/empty, 1=group
    ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1)              // 空 tensor 模板
);

ASCENDC_TPL_SEL(
    // normal 模板：isTailR 固定 true（All Reduce），仅 1 组
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    // 空 tensor 模板
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
    // group 模板：isEmptyTensor 固定 0（与 empty 互斥）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // OPS_DATA_COMPARE_TILING_KEY_H_
