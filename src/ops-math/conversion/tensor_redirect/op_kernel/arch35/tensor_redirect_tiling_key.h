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
 * \file tensor_redirect_tiling_key.h
 * \brief tiling key declare for tensor_redirect
 */

#ifndef TENSOR_REDIRECT_TILING_KEY_H
#define TENSOR_REDIRECT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 唯一调度形态，schMode 单值
#define TPL_SCH_MODE_0 0 // 唯一调度模式：CopyIn -> CopyOut 双缓冲

ASCENDC_TPL_ARGS_DECL(TensorRedirect, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0)));

#endif // TENSOR_REDIRECT_TILING_KEY_H
