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
 * \file top_k_pq_distance_infershape.cpp
 * \brief InferShape for TopKPQDistance operator:
 *        - 三个输出 shape 均为 {k}（一维）
 *        - topk_distance.dtype ← pq_distance[0].dtype
 *        - topk_ivf.dtype      ← pq_ivf[0].dtype
 *        - topk_index.dtype    ← pq_index[0].dtype
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

// REG_OP(TopKPQDistance) 输入顺序（5 个 DYNAMIC_INPUT）：
//   0: actual_count, 1: pq_distance, 2: grouped_extreme_distance,
//   3: pq_ivf,       4: pq_index
static constexpr size_t INPUT_IDX_PQ_DISTANCE = 1;
static constexpr size_t INPUT_IDX_PQ_IVF = 3;
static constexpr size_t INPUT_IDX_PQ_INDEX = 4;

// REG_OP(TopKPQDistance) 输出顺序（3 个 OUTPUT）：
//   0: topk_distance, 1: topk_ivf, 2: topk_index
static constexpr size_t OUTPUT_IDX_TOPK_DISTANCE = 0;
static constexpr size_t OUTPUT_IDX_TOPK_IVF = 1;
static constexpr size_t OUTPUT_IDX_TOPK_INDEX = 2;
static constexpr size_t TOPK_PQ_DISTANCE_OUTPUT_NUM = 3;

// REG_OP(TopKPQDistance) 属性顺序：order(0), k(1), group_size(2)
static constexpr size_t ATTR_IDX_K = 1;
static constexpr size_t ATTR_IDX_GROUP_SIZE = 2;

// 默认 dynamic input 实例索引
static constexpr size_t DYNAMIC_INSTANCE_ZERO = 0;

static ge::graphStatus InferShape4TopKPQDistance(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TopKPQDistance InferShape");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* k_ptr = attrs->GetInt(ATTR_IDX_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, k_ptr);
    const int64_t topK = *k_ptr;
    const int64_t* group_size_ptr = attrs->GetInt(ATTR_IDX_GROUP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, group_size_ptr);
    const int64_t groupSize = *group_size_ptr;

    OP_CHECK_IF(topK <= 0, OP_LOGE(context, "k should be bigger than zero, but got k[%ld].", topK),
                return GRAPH_FAILED);
    OP_CHECK_IF(groupSize <= 0,
                OP_LOGE(context, "group_size should be bigger than zero, but got group_size[%ld].", groupSize),
                return GRAPH_FAILED);

    // 三个输出 shape 均为 {topK}
    for (size_t i = 0; i < TOPK_PQ_DISTANCE_OUTPUT_NUM; i++) {
        auto out_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
        out_shape->SetDimNum(1);
        out_shape->SetDim(0, topK);
    }
    OP_LOGD(context->GetNodeName(), "End to do TopKPQDistance InferShape");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4TopKPQDistance(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TopKPQDistance InferDataType");
    // topk_distance.dtype ← pq_distance[0].dtype
    context->SetOutputDataType(OUTPUT_IDX_TOPK_DISTANCE,
                               context->GetDynamicInputDataType(INPUT_IDX_PQ_DISTANCE, DYNAMIC_INSTANCE_ZERO));
    // topk_ivf.dtype ← pq_ivf[0].dtype
    context->SetOutputDataType(OUTPUT_IDX_TOPK_IVF,
                               context->GetDynamicInputDataType(INPUT_IDX_PQ_IVF, DYNAMIC_INSTANCE_ZERO));
    // topk_index.dtype ← pq_index[0].dtype
    context->SetOutputDataType(OUTPUT_IDX_TOPK_INDEX,
                               context->GetDynamicInputDataType(INPUT_IDX_PQ_INDEX, DYNAMIC_INSTANCE_ZERO));
    OP_LOGD(context->GetNodeName(), "End to do TopKPQDistance InferDataType");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TopKPQDistance).InferShape(InferShape4TopKPQDistance).InferDataType(InferDataType4TopKPQDistance);

} // namespace ops
