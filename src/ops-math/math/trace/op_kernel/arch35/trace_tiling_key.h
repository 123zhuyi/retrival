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
 * \file trace_tiling_key.h
 * \brief Tiling key declaration for trace operator
 *
 * Single template parameter:
 *   schMode (UINT 1-bit): execution mode
 *     0 = DEFAULT (single-core warp reduction)
 *
 * Naming note (review issue #4):
 *   MDE §3.2 使用 TRACE_TPL_SCH_MODE_0 命名，代码使用 TRACE_TPL_MODE_DEFAULT。
 *   两者功能等价（值均为 0），命名差异仅为风格选择：
 *   - MDE 命名强调"schedule mode 0"（调度模式编号）
 *   - 代码命名强调"default mode"（默认模式，语义更直观）
 *   不影响功能和编译，保留代码命名以保持与 tiling.cpp 一致性。
 */

#ifndef TRACE_TILING_KEY_H_
#define TRACE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// MDE 命名为 TRACE_TPL_SCH_MODE_0，此处使用语义更直观的 TRACE_TPL_MODE_DEFAULT
#define TRACE_TPL_MODE_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(
    Trace,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST,
        TRACE_TPL_MODE_DEFAULT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TRACE_TPL_MODE_DEFAULT),
        ASCENDC_TPL_TILING_STRUCT_SEL(TraceTilingData)
    )
);

#endif  // TRACE_TILING_KEY_H_
