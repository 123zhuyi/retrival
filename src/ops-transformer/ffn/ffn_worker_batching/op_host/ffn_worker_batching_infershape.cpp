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
 * \file ffn_worker_batching_infershape.cpp
 * \brief
 */

#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t EXPERT_NUM_ATTR = 0;
static constexpr int64_t MAX_OUT_SHAPE_ATTR = 1;
static constexpr int64_t TOKEN_DTYPE_ATTR = 2;

static constexpr int64_t Y_OUT = 0;
static constexpr int64_t GROUP_LIST_OUT = 1;
static constexpr int64_t SESSION_IDS_OUT = 2;
static constexpr int64_t MICRO_BATCH_IDS_OUT = 3;
static constexpr int64_t TOKEN_IDS_OUT = 4;
static constexpr int64_t EXPERT_OFFSETS_OUT = 5;
static constexpr int64_t DYNAMIC_SCALE_OUT = 6;
static constexpr int64_t ACTUAL_TOKEN_NUM_OUT = 7;

static constexpr int64_t TOKEN_KIND_ZERO = 0;
static constexpr int64_t TOKEN_KIND_ONE = 1;
static constexpr int64_t TOKEN_KIND_TWO = 2;

static constexpr int64_t NUM_FOUR = 4;
static constexpr uint32_t INDEX_ZERO = 0;
static constexpr uint32_t INDEX_ONE = 1;
static constexpr uint32_t INDEX_TWO = 2;
static constexpr uint32_t INDEX_THREE = 3;

static graphStatus InferShape4FfnWorkerBatching(gert::InferShapeContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertNumPtr);
    auto expertNum = *expertNumPtr;
    auto maxOutShapePtr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxOutShapePtr);
    if (maxOutShapePtr->GetSize() != NUM_FOUR) {
        OP_LOGE(context->GetNodeName(), "attr max_out_shape len must be 4, but is [%d].",
                static_cast<int32_t>(maxOutShapePtr->GetSize()));
        return ge::GRAPH_FAILED;
    }

    const int64_t *maxOutShapeArray = reinterpret_cast<const int64_t *>(maxOutShapePtr->GetData());
    int64_t Y = maxOutShapeArray[INDEX_ZERO] * maxOutShapeArray[INDEX_ONE] * maxOutShapeArray[INDEX_TWO];
    int64_t H = maxOutShapeArray[INDEX_THREE];

    gert::Shape *yShape = context->GetOutputShape(Y_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *groupList = context->GetOutputShape(GROUP_LIST_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupList);

    gert::Shape *sessionIds = context->GetOutputShape(SESSION_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, sessionIds);

    gert::Shape *microBatchIds = context->GetOutputShape(MICRO_BATCH_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, microBatchIds);

    gert::Shape *tokenIds = context->GetOutputShape(TOKEN_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenIds);

    gert::Shape *expertOffsets = context->GetOutputShape(EXPERT_OFFSETS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertOffsets);

    gert::Shape *dynamicScale = context->GetOutputShape(DYNAMIC_SCALE_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dynamicScale);

    gert::Shape *actualTokenNum = context->GetOutputShape(ACTUAL_TOKEN_NUM_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualTokenNum);

    *yShape = {Y, H};
    *groupList = {expertNum, 2};
    *sessionIds = {Y};
    *microBatchIds = {Y};
    *tokenIds = {Y};
    *expertOffsets = {Y};
    *dynamicScale = {Y};
    *actualTokenNum = {1};

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4FfnWorkerBatching(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    int64_t tokenDtype = TOKEN_KIND_ZERO;
    if (tokenDtypePtr != nullptr) {
        tokenDtype = *tokenDtypePtr;
    }

    if (tokenDtype == TOKEN_KIND_ZERO) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT16);
    } else if (tokenDtype == TOKEN_KIND_ONE) {
        context->SetOutputDataType(Y_OUT, ge::DT_BF16);
    } else {
        context->SetOutputDataType(Y_OUT, ge::DT_INT8);
    }

    context->SetOutputDataType(GROUP_LIST_OUT, ge::DT_INT64);
    context->SetOutputDataType(SESSION_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(MICRO_BATCH_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(TOKEN_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(EXPERT_OFFSETS_OUT, ge::DT_INT32);
    context->SetOutputDataType(DYNAMIC_SCALE_OUT, ge::DT_FLOAT);
    context->SetOutputDataType(ACTUAL_TOKEN_NUM_OUT, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFERSHAPE(FfnWorkerBatching)
    .InferShape(InferShape4FfnWorkerBatching)
    .InferDataType(InferDataType4FfnWorkerBatching);
} // namespace ops