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
 * \file stride_add_tiling_key.h
 * \brief StrideAdd tiling key declare
 */

#ifndef __STRIDE_ADD_TILING_KEY_H__
#define __STRIDE_ADD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 单场景模式，dtype 由 DTYPE_X1 宏自动实例化
#define STRIDE_ADD_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(
    StrideAdd,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, STRIDE_ADD_TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, STRIDE_ADD_TPL_SCH_MODE_0),
    ASCENDC_TPL_TILING_STRUCT_SEL(StrideAddTilingData)));

#endif // __STRIDE_ADD_TILING_KEY_H__