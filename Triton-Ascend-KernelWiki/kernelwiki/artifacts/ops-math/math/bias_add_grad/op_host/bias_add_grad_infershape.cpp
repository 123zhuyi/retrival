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
 * \file bias_add_grad_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <algorithm>

using namespace ge;
namespace ops {
const size_t DIM_SIZE2 = 2;
static const std::vector<std::pair<std::string, Format>> kFormatMap = {
    {"NCHW", FORMAT_NCHW},
    {"NHWC", FORMAT_NHWC},
};

static ge::graphStatus InferShape4BiasAddGrad(gert::InferShapeContext* context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* data_format = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);

    size_t dim_num = in_shape->GetDimNum();
    auto it = std::find_if(
        kFormatMap.begin(), kFormatMap.end(),
        [&data_format](const std::pair<std::string, Format>& item) -> bool { return item.first == data_format; });
    OP_CHECK_IF(
        it == kFormatMap.end(), OP_LOGE(context->GetNodeName(), "data_format %s must in (NCHW, NHWC).", data_format),
        return GRAPH_FAILED);
    if (dim_num < DIM_SIZE2) {
        OP_LOGE(
            context->GetNodeName(),
            "The bias add grad op dimension(%lu) must be greater than or equal to 2 when format is NCHW!", dim_num);
        return GRAPH_FAILED;
    }
    out_shape->SetDimNum(0);
    out_shape->AppendDim(in_shape->GetDim((it->second == FORMAT_NHWC) ? (dim_num - 1) : 1));

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BiasAddGrad).InferShape(InferShape4BiasAddGrad);
} // namespace ops
