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
 * \file calc_buckets_limit_and_offset_infershape.cpp
 * \brief infershape func of CalcBucketsLimitAndOffset
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_IN_BUCKET_LIST = 0;
static constexpr size_t CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_IN_IVF_OFFSET = 2;
static constexpr size_t CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_LIMIT = 0;
static constexpr size_t CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_OFFSET = 1;

static ge::graphStatus CalcBucketsLimitAndOffsetInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do CalcBucketsLimitAndOffsetInferShape");
    const auto* bucketListShape = context->GetInputShape(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_IN_BUCKET_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketListShape);
    auto* bucketsLimitShape = context->GetOutputShape(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_LIMIT);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketsLimitShape);
    auto* bucketsOffsetShape = context->GetOutputShape(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketsOffsetShape);

    *bucketsLimitShape = *bucketListShape;
    *bucketsOffsetShape = *bucketListShape;
    OP_LOGD(context->GetNodeName(), "End to do CalcBucketsLimitAndOffsetInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcBucketsLimitAndOffsetInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do CalcBucketsLimitAndOffsetInferDataType");
    auto bucketListDtype = context->GetInputDataType(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_IN_BUCKET_LIST);
    context->SetOutputDataType(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_LIMIT, bucketListDtype);
    auto ivfOffsetDtype = context->GetInputDataType(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_IN_IVF_OFFSET);
    context->SetOutputDataType(CALC_BUCKETS_LIMIT_AND_OFFSET_IDX_OUT_BUCKETS_OFFSET, ivfOffsetDtype);
    OP_LOGD(context->GetNodeName(), "End to do CalcBucketsLimitAndOffsetInferDataType");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CalcBucketsLimitAndOffset)
    .InferShape(CalcBucketsLimitAndOffsetInferShape)
    .InferDataType(CalcBucketsLimitAndOffsetInferDataType);
} // namespace ops
