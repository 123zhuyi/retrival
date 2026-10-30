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
 * \file diag_v2_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
// -------------------DiagV2 Ops START---------------------
static constexpr size_t DiagV2_IN_X_IDX = 0;
static constexpr size_t DiagV2_OUT_Y_IDX = 0;
static constexpr int64_t INT_DATA_2 = 2;
constexpr size_t kAxisAttrIdx = 0U;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t NEG_TWO = -2;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;

static ge::graphStatus Infershape4DiagV2(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do DiagV2Infershape.");
    // 获取输入值shape
    const gert::Shape* input_x_shape = context->GetInputShape(DiagV2_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);

    size_t x_dim_num = input_x_shape->GetDimNum();
    if (x_dim_num > DIM_TWO) {
        OP_LOGD("The dim number of input should be less than 3.");
        return ge::GRAPH_FAILED;
    }

    // 获取属性值
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto pkAxisAttrIdx = attrs->GetInt(kAxisAttrIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, pkAxisAttrIdx);
    int64_t diagonal = *pkAxisAttrIdx;

    // 获取输出值shape
    gert::Shape* output_y_shape = context->GetOutputShape(DiagV2_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    if (Ops::Base::IsUnknownRank(*input_x_shape)) {
        output_y_shape->SetDimNum(1);
        output_y_shape->SetDim(0, NEG_TWO);
    } else if (Ops::Base::IsUnknownShape(*input_x_shape)) {
        if (x_dim_num < DIM_TWO) {
            output_y_shape->SetDimNum(DIM_TWO);
            output_y_shape->SetDim(0, NEG_ONE);
            output_y_shape->SetDim(1, NEG_ONE);
        } else {
            output_y_shape->SetDimNum(1);
            output_y_shape->SetDim(0, NEG_ONE);
        }
    } else if (x_dim_num < DIM_TWO) { // 1D->2D
        output_y_shape->SetDimNum(0);
        // 获取元素element的个数
        auto total_element_num = input_x_shape->GetShapeSize();
        for (int64_t k = 0; k < INT_DATA_2; k++) {
            output_y_shape->AppendDim(total_element_num + std::abs(diagonal));
        }
    } else { // 2D->1D
        output_y_shape->SetDimNum(1);
        if (diagonal > 0) {
            output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0), input_x_shape->GetDim(1) - diagonal));
            // 判断偏移量是否超出上限
            if (diagonal >= input_x_shape->GetDim(1)) {
                output_y_shape->SetDim(0, 0);
            }
        } else {
            output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0) + diagonal, input_x_shape->GetDim(1)));
            if (-diagonal >= input_x_shape->GetDim(0)) {
                output_y_shape->SetDim(0, 0);
            }
        }
    }

    OP_LOGD(context, "output_y_shape = %s.", Ops::Base::ToString(*output_y_shape).c_str());
    OP_LOGD(context, "End to do DiagV2Infershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagV2).InferShape(Infershape4DiagV2);
// -------------------DiagV2 Ops END---------------------
} // namespace ops