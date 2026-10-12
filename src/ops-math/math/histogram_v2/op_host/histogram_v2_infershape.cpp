/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t BINS_IDX = 0;
static constexpr int64_t OUTPUT_IDX = 0;
static constexpr int64_t DEFAULT_BINS = 100;

static ge::graphStatus HistogramV2InferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramV2InferShapeFunc");
    // 获取bins
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    auto binsPtr = attr->GetAttrPointer<int64_t>(BINS_IDX);
    // 属性缺失时回落默认值，与 tiling 侧行为保持一致
    int64_t bins = (binsPtr == nullptr) ? DEFAULT_BINS : *binsPtr;
    OP_LOGD(context, "bins = %ld", bins);
    OP_CHECK_IF(bins <= 0, OP_LOGE(context, "bins has to be positive, but got %ld", bins), return ge::GRAPH_FAILED);

    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, bins);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HistogramV2).InferShape(HistogramV2InferShapeFunc);
} // namespace ops
