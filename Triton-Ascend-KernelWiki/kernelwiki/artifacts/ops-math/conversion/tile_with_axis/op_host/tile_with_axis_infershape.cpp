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
 * \file tile_with_axis_infershape.cpp
 * \brief TileWithAxis 算子形状推导实现
 *
 * 设计依据: DESIGN.md v1.6
 * 输出 y 的 shape: y.shape[axis] = x.shape[axis] * tiles, 其余维度与 x 相同
 * rank=0 时 y.shape = [tiles]
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "op_host/util/shape_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4TileWithAxis(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // 获取属性 axis 和 tiles
    constexpr size_t ATTR_AXIS_IDX = 0;
    constexpr size_t ATTR_TILES_IDX = 1;
    auto attrs = context->GetAttrs();
    const int64_t* axis_ptr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    const int64_t* tiles_ptr = attrs->GetAttrPointer<int64_t>(ATTR_TILES_IDX);
    int64_t axis = (axis_ptr != nullptr) ? *axis_ptr : 0;
    int64_t tiles = (tiles_ptr != nullptr) ? *tiles_ptr : 1;

    // 获取输入 rank
    int64_t rank = input_shape->GetDimNum();

    // unknown rank 处理: input rank 未知时 output 也设未知
    if (Ops::Base::IsUnknownRank(*input_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    // 空 tensor 检查
    if (rank == 0) {
        // 标量输入: 输出 shape = [tiles]
        output_shape->SetDimNum(1);
        output_shape->SetDim(0, tiles);
        return ge::GRAPH_SUCCESS;
    }

    // 归一化 axis 到非负
    if (axis < 0) {
        axis = axis + rank;
    }

    // 复制 input shape 到 output shape
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; i++) {
        if (i == axis) {
            output_shape->SetDim(i, input_shape->GetDim(i) * tiles);
        } else {
            output_shape->SetDim(i, input_shape->GetDim(i));
        }
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TileWithAxis).InferShape(InferShape4TileWithAxis);

} // namespace ops
