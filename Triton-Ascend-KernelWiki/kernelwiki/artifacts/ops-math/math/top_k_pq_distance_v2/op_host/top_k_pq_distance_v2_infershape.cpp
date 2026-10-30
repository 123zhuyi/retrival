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
 * \file top_k_pq_distance_v2_infershape.cpp
 * \brief InferShape for TopKPQDistanceV2: both outputs are shaped [batch, k].
 */

#include <cstring>

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kOutputIndex1 = 1U;
constexpr size_t kInputsDimNum = 2U;
constexpr size_t kOutputsDimNum = 2U;
constexpr size_t kAttrOrderIdx = 0U;
constexpr size_t kAttrKIdx = 1U;
constexpr size_t kAttrGroupSizeIdx = 2U;
constexpr const char* kOrderAsc = "ASC";
constexpr const char* kOrderDes = "DES";
} // namespace

// 读取并校验 order / k / group_size，k 通过出参回传（校验含 k>0、k<=elementNum、group_size>0、整除）。
static ge::graphStatus CheckAttrAndGetK(gert::InferShapeContext* context, int64_t elementNum, int64_t& k)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* order = attrs->GetStr(kAttrOrderIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, order);
    const int64_t* kPtr = attrs->GetInt(kAttrKIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    k = *kPtr;
    const int64_t* groupSizePtr = attrs->GetInt(kAttrGroupSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupSizePtr);
    const int64_t groupSize = *groupSizePtr;

    OP_CHECK_IF((strcmp(order, kOrderAsc) != 0) && (strcmp(order, kOrderDes) != 0),
                OP_LOGE(context, "order should be ASC or DES, but got %s.", order), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (k <= 0) || (k > elementNum),
        OP_LOGE(context, "k should be in range (0, element_num], but got k[%ld], element_num[%ld].", k, elementNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupSize <= 0,
                OP_LOGE(context, "group_size should bigger than zero, but got group_size[%ld].", groupSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (elementNum % groupSize) != 0,
        OP_LOGE(context,
                "element_num should be an integer multiple of group_size, but element_num is [%ld], group_size is "
                "[%ld].",
                elementNum, groupSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4TopKPQDistanceV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4TopKPQDistanceV2");

    const gert::Shape* input0Shape = context->GetInputShape(kInputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Shape);
    OP_CHECK_IF(input0Shape->GetDimNum() != kInputsDimNum, OP_LOGE(context, "input pq_distance shape is not 2D."),
                return ge::GRAPH_FAILED);

    const gert::Shape* input1Shape = context->GetInputShape(kInputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1Shape);
    OP_CHECK_IF(input1Shape->GetDimNum() != kInputsDimNum,
                OP_LOGE(context, "input grouped_extreme_distance shape is not 2D."), return ge::GRAPH_FAILED);

    const int64_t batch = input0Shape->GetDim(0);
    const int64_t elementNum = input0Shape->GetDim(1);
    int64_t k = 0;
    if (CheckAttrAndGetK(context, elementNum, k) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output0Shape = context->GetOutputShape(kOutputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0Shape);
    output0Shape->SetDimNum(kOutputsDimNum);
    output0Shape->SetDim(0U, batch);
    output0Shape->SetDim(1U, k);

    gert::Shape* output1Shape = context->GetOutputShape(kOutputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, output1Shape);
    output1Shape->SetDimNum(kOutputsDimNum);
    output1Shape->SetDim(0U, batch);
    output1Shape->SetDim(1U, k);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TopKPQDistanceV2).InferShape(InferShape4TopKPQDistanceV2);
} // namespace ops
