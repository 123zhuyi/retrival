/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stack_ball_query_infershape.cpp
 * \brief stack_ball_query operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

static constexpr int INPUT_NODE_NUM = 4;
static constexpr int OUTPUT_NODE_NUM = 1;
static constexpr int INDEX_ATTR_SAMPLE_NUM = 1;
static constexpr int INDEX_INPUT_XYZ = 0;
static constexpr int INDEX_INPUT_CENTER_XYZ = 1;
static constexpr int INDEX_INPUT_XYZ_BATCH_CNT = 2;
static constexpr int INDEX_INPUT_CENTER_XYZ_BATCH_CNT = 3;
static constexpr int INDEX_OUTPUT_IDX = 0;

static constexpr int IDX_DIM_NUM = 2;
static constexpr int IDX_DIM0 = 0;
static constexpr int IDX_DIM1 = 1;

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4StackBallQuery(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4StackBallQuery", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != INPUT_NODE_NUM || context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }

    const gert::Shape* xyz_shape = context->GetInputShape(INDEX_INPUT_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyz_shape);
    const gert::Shape* center_xyz_shape = context->GetInputShape(INDEX_INPUT_CENTER_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, center_xyz_shape);
    const gert::Shape* xyz_batch_cnt_shape = context->GetInputShape(INDEX_INPUT_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyz_batch_cnt_shape);
    const gert::Shape* center_xyz_batch_cnt_shape = context->GetInputShape(INDEX_INPUT_CENTER_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, center_xyz_batch_cnt_shape);

    gert::Shape* idx_shape = context->GetOutputShape(INDEX_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx_shape);

    // unknown rank：输出置 unknown rank 后直接返回
    if (Ops::Base::IsUnknownRank(*xyz_shape) || Ops::Base::IsUnknownRank(*center_xyz_shape) ||
        Ops::Base::IsUnknownRank(*xyz_batch_cnt_shape) || Ops::Base::IsUnknownRank(*center_xyz_batch_cnt_shape)) {
        Ops::Base::SetUnknownRank(*idx_shape);
        return ge::GRAPH_SUCCESS;
    }
    // unknown shape：维度数已知但维度值未知，输出置 unknown shape 后返回
    if (Ops::Base::IsUnknownShape(*xyz_shape) || Ops::Base::IsUnknownShape(*center_xyz_shape) ||
        Ops::Base::IsUnknownShape(*xyz_batch_cnt_shape) || Ops::Base::IsUnknownShape(*center_xyz_batch_cnt_shape)) {
        Ops::Base::SetUnknownShape(IDX_DIM_NUM, *idx_shape);
        return ge::GRAPH_SUCCESS;
    }

    // 维度校验
    if (xyz_shape->GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "xyz",
                                     (std::to_string(xyz_shape->GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }
    if (center_xyz_shape->GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "center_xyz",
                                     (std::to_string(center_xyz_shape->GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }
    if (xyz_batch_cnt_shape->GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "xyz_batch_cnt",
                                     (std::to_string(xyz_batch_cnt_shape->GetDimNum()) + "D").c_str(), "1D");
        return GRAPH_FAILED;
    }
    if (center_xyz_batch_cnt_shape->GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "center_xyz_batch_cnt",
                                     (std::to_string(center_xyz_batch_cnt_shape->GetDimNum()) + "D").c_str(), "1D");
        return GRAPH_FAILED;
    }

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* sampl_num_ptr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SAMPLE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampl_num_ptr);
    auto sampl_num = *sampl_num_ptr;

    idx_shape->SetDimNum(IDX_DIM_NUM);
    idx_shape->SetDim(IDX_DIM0, center_xyz_shape->GetDim(0));
    idx_shape->SetDim(IDX_DIM1, static_cast<int64_t>(sampl_num));

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StackBallQuery).InferShape(InferShape4StackBallQuery);
} // namespace ops
