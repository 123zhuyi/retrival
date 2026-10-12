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
 * \file ndtri_tiling_key.h
 * \brief Ndtri Tiling 模板参数定义
 *
 * dtype 由 def 驱动（构建系统按 def 的 DataType profile 生成 DTYPE_SELF 宏），
 * tiling_key 不再编码 dtype 维度，只保留 K_ALIGN。
 */

#ifndef NDTRI_TILING_KEY_H_
#define NDTRI_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Ndtri, ASCENDC_TPL_UINT_DECL(K_ALIGN, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif // NDTRI_TILING_KEY_H_
