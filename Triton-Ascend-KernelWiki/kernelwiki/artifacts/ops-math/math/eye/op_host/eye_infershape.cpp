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
 * \file eye.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"


namespace {
constexpr size_t EYE_ATTR_NUM_ROWS = 0;
constexpr size_t EYE_ATTR_NUM_COLUMNS = 1;
constexpr size_t EYE_ATTR_BATCH_SHAPE = 2;
}  // namespace

namespace ops {
using namespace Ops::Base;
ge::graphStatus InferShapeForEye(gert::InferShapeContext* context) {
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);

    auto num_rows_ptr = attr_ptr->GetAttrPointer<int64_t>(EYE_ATTR_NUM_ROWS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_rows_ptr);
    auto num_columns_ptr = attr_ptr->GetAttrPointer<int64_t>(EYE_ATTR_NUM_COLUMNS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_columns_ptr);
    auto batch_shape_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(EYE_ATTR_BATCH_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, batch_shape_ptr);
    auto batch_shape_array = reinterpret_cast<const int64_t*>(batch_shape_ptr->GetData());

    std::vector<int64_t> output_shape_vec;
    for (size_t i = 0; i < batch_shape_ptr->GetSize(); ++i) {
        if (batch_shape_array[i] <= 0) {
            OP_LOGE(context->GetNodeName(), "value of batch_shape must greater than 0, but got %ld",
                    batch_shape_array[i]);
            return ge::GRAPH_FAILED;
        }
        output_shape_vec.push_back(batch_shape_array[i]);
    }

    auto num_rows = *num_rows_ptr;
    auto num_columns = *num_columns_ptr;
    if (num_rows <= 0) {
        OP_LOGE(context->GetNodeName(), "num_rows must greater than 0, but got %ld",
                num_rows);
        return ge::GRAPH_FAILED;
    }
    if (num_columns <= 0) {
        num_columns = num_rows;
    }
    output_shape_vec.push_back(num_rows);
    output_shape_vec.push_back(num_columns);

    out_shape->SetDimNum(output_shape_vec.size());
    for (size_t i = 0; i < output_shape_vec.size(); ++i) {
        out_shape->SetDim(i, output_shape_vec[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Eye).InferShape(InferShapeForEye);
}  // namespace ops
