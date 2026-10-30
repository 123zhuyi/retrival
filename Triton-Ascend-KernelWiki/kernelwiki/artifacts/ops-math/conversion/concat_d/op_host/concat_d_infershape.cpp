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
 * \file concat_d_infershape.cpp
 * \brief
 */

#include "../../concat/op_host/concat_infershape.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ConcatD(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* concat_dim = attrs->GetAttrPointer<int64_t>(INDEX_CONCAT_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, concat_dim);
    const int64_t* N = attrs->GetAttrPointer<int64_t>(INDEX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, N);
    return ConcatInferShapeCommon(context, 0, *N, *concat_dim);
}

IMPL_OP_INFERSHAPE(ConcatD).InferShape(InferShape4ConcatD);
} // namespace ops
