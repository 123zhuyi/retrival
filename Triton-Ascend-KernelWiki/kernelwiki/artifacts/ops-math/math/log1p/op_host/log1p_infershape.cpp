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
 * \file log1p_infershape.cpp
 * \brief
 */

#include "op_host/infershape_elewise_util.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape4Log1p(gert::InferShapeContext *context)
{
    return Ops::Base::InferShape4Elewise(context);
}

IMPL_OP_INFERSHAPE(Log1p).InferShape(InferShape4Log1p);
} // namespace ops