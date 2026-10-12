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
 * \file reciprocal_grad_struct.h
 * \brief ReciprocalGrad 算子 TilingKey 定义（atvoss 框架 - Elewise 模式）
 */

#ifndef RECIPROCAL_GRAD_STRUCT_H
#define RECIPROCAL_GRAD_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace ReciprocalGradOp {
ASCENDC_TPL_ARGS_DECL(ReciprocalGrad, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 0, 1)));
} // namespace ReciprocalGradOp

#endif // RECIPROCAL_GRAD_STRUCT_H
