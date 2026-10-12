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
 * \file expand_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t INPUT_INDEX_SHAPE = 1;

template <typename T>
static void GetConstValueToShape(const gert::Tensor* tensor, size_t size, gert::Shape* shape)
{
    const T* value = tensor->GetData<T>();
    shape->SetDimNum(size);
    for (size_t i = 0; i < size; i++) {
        shape->SetDim(i, value[i]);
    }
}

static ge::graphStatus InferShape4Expand(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(),"Enter Math InferShape4Expand!");
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto shape_tensor = context->GetInputTensor(INPUT_INDEX_SHAPE);
    auto shape_size = static_cast<size_t>(shape_tensor->GetShapeSize());
    OP_CHECK_IF(
        shape_size < x_shape->GetDimNum(),
        OP_LOGE(context->GetNodeName(), "%s",
        ConcatString("input1 size ", shape_size, " cannot be less than input0 size ", x_shape->GetDimNum(), ", error!")
                .c_str()),
        return ge::GRAPH_FAILED);
    out_shape->SetDimNum(shape_size);
    OP_LOGD(context->GetNodeName(), "input1 size is %zu", shape_size);
    DataType data_type = shape_tensor->GetDataType();
    OP_CHECK_IF(
        (data_type != DT_INT32) && (data_type != DT_INT64),
        OP_LOGE(context->GetNodeName(), "%s",
            ConcatString("input1 dtype ", Ops::Base::ToString(data_type), " must be in (int32,int64)!").c_str()),
        return ge::GRAPH_FAILED);
    size_t diff = shape_size - x_shape->GetDimNum();
    if (data_type == DT_INT32) {
        GetConstValueToShape<int32_t>(shape_tensor, shape_size, out_shape);
    } else {
        GetConstValueToShape<int64_t>(shape_tensor, shape_size, out_shape);
    }
    OP_LOGD(context->GetNodeName(), "%s",
        ConcatString("input0 shape is ", Ops::Base::ToString(*x_shape).c_str(),
            ", input1 shape is ", Ops::Base::ToString(*out_shape).c_str()).c_str());
    for (size_t i = 0; i < shape_size; i++) {
        if (i >= diff) {
            int64_t x_dim = x_shape->GetDim(i - diff);
            int64_t out_dim = out_shape->GetDim(i);
            // Handle -1: replace with x dimension
            if (out_dim == -1) {
                out_shape->SetDim(i, x_dim);
                continue;
            }
            // If target is 1 but x is not 1, use x dimension (no bidirectional broadcast)
            if (out_dim == 1 && x_dim != 1) {
                out_shape->SetDim(i, x_dim);
                continue;
            }
            // Broadcast check: x must be 1 or equal to target dimension
            if (x_dim != 1 && x_dim != out_dim) {
                OP_LOGE(context->GetNodeName(), "%s",
                    ConcatString("x dimension ", x_dim, " at axis ", (i - diff),
                        " cannot be broadcast to ", out_dim, " at axis ", i).c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            // New dimension added in front, treat as 1 for broadcasting
            int64_t out_dim = out_shape->GetDim(i);
            if (out_dim == -1) {
                out_shape->SetDim(i, 1);
            }
        }
    }
    OP_LOGD(context->GetNodeName(), "%s",
        ConcatString("input0 and input1 infer output, output shape is ",
            Ops::Base::ToString(*out_shape).c_str()).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Expand).InferShape(InferShape4Expand).InputsDataDependency({INPUT_INDEX_SHAPE});
} // namespace ops
