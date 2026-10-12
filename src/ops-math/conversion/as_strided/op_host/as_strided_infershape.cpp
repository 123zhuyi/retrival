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
 * \file as_strided_infershape.cpp
 * \brief
 */
#include "util/const_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_SIZE = 1;
static constexpr size_t IN_STRIDE = 2;
static constexpr size_t IN_OFFSET = 3;
static constexpr size_t OUT_Y = 0;

static ge::graphStatus InferShapeForAsStrided(gert::InferShapeContext* context)
{
    gert::Shape* yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* size_tensor = context->GetInputTensor(IN_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, size_tensor);

    ge::DataType size_dtype = size_tensor->GetDataType();

    switch (size_dtype) {
        case ge::DT_INT32: {
            Ops::Base::GetValueToShape<int32_t>(size_tensor, *yShape);
            break;
        }
        case ge::DT_INT64: {
            Ops::Base::GetValueToShape<int64_t>(size_tensor, *yShape);
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(
                context->GetNodeName(), "size", Ops::Base::ToString(size_dtype).c_str(), "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsStrided).InferShape(InferShapeForAsStrided).InputsDataDependency({IN_SIZE, IN_STRIDE, IN_OFFSET});
} // namespace ops
