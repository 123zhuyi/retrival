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
 * \file as_strided_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/const_util.h"

namespace ge {
static constexpr size_t IN_SIZE = 1;
static constexpr size_t OUT_Y = 0;

graphStatus InferShapeForAsStrided(gert::InferShapeContext* context)
{
    gert::Shape* yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Tensor* sizeTensor = context->GetInputTensor(IN_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);
    const ge::DataType sizeDtype = sizeTensor->GetDataType();

    switch (sizeDtype) {
        case ge::DT_INT32: {
            Ops::Base::GetValueToShape<int32_t>(sizeTensor, *yShape);
            break;
        }
        case ge::DT_INT64: {
            Ops::Base::GetValueToShape<int64_t>(sizeTensor, *yShape);
            break;
        }
        default:
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(context->GetNodeName(), "size", Ops::Base::ToString(sizeDtype).c_str(),
                                             "[int32, int64]");
            return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

graphStatus InferDataTypeForAsStrided(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUT_Y, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(AsStrided)
    .InferShape(ge::InferShapeForAsStrided)
    .InferDataType(ge::InferDataTypeForAsStrided)
    .InputsDataDependency({1, 2, 3});
} // namespace ops
