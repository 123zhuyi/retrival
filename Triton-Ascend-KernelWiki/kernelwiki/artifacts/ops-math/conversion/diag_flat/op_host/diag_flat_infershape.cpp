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
 * \file diag_flat_infershape.cpp
 * \brief
 */

#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
// ------------------- Diagflat Ops START---------------------  1->2
static constexpr size_t DIAGFLAT_IN_X_IDX = 0;
static constexpr size_t DIAGFLAT_OUT_Y_IDX = 0;
constexpr size_t kAxisAttrIdx = 0U;
static constexpr size_t INT_DATA_2 = 2;

static ge::graphStatus InfershapeForDiagFlat(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do DiagflatInfershape.");
    // 获取输入值shape
    const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    // 获取属性值
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t diagonal = *(attrs->GetInt(kAxisAttrIdx));

    // 获取输出值shape
    gert::Shape* output_y_shape = context->GetOutputShape(DIAGFLAT_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    const size_t input_dim_num = 2;
    if (Ops::Base::IsUnknownRank(*inputShape)) {
        output_y_shape->SetDimNum(input_dim_num);
        output_y_shape->SetDim(0, -1);
        output_y_shape->SetDim(1, -1);
    } else if (Ops::Base::IsUnknownShape(*inputShape)) {
        output_y_shape->SetDimNum(input_dim_num);
        output_y_shape->SetDim(0, -1);
        output_y_shape->SetDim(1, -1);
    } else {
        // 获取元素element的个数, 2D->1D的场景
        output_y_shape->SetDimNum(input_dim_num);
        auto total_element_num = inputShape->GetShapeSize();
        auto output_width = total_element_num + std::abs(diagonal);
        output_y_shape->SetDim(0, output_width);
        output_y_shape->SetDim(1, output_width);
    }

    OP_LOGI(context->GetNodeName(), "output_y_shape = %s.", Ops::Base::ToString(*output_y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do DiagFlatInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagFlat).InferShape(InfershapeForDiagFlat);
// -------------------DiagFlat Ops END---------------------

} // namespace ops
