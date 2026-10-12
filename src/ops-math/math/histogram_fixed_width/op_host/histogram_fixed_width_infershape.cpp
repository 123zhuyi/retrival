/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/const_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_X = 0;
static constexpr size_t INPUT_RANGE = 1;
static constexpr size_t INPUT_NBINS = 2;
static constexpr size_t ATTR_DTYPE = 0;
static constexpr size_t OUTPUT_IDX = 0;
static constexpr int64_t STC_RANGE_SHAPE_SIZE = 2;
static constexpr int64_t DYN_RANGE_SHAPE_SIZE = -1;
static constexpr int64_t OUTPUT_DTYPE = static_cast<int64_t>(ge::DT_INT32);

static ge::graphStatus InferShapeForHistogramFixedWidth(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* rangeShape = context->GetInputShape(INPUT_RANGE);
    OP_CHECK_NULL_WITH_CONTEXT(context, rangeShape);
    int64_t rangeShapeSize = rangeShape->GetShapeSize();
    if (rangeShapeSize != STC_RANGE_SHAPE_SIZE && rangeShapeSize != DYN_RANGE_SHAPE_SIZE) {
        std::string errorMsg = "The shape size of range should be equal to " + std::to_string(STC_RANGE_SHAPE_SIZE) +
                               " or " + std::to_string(DYN_RANGE_SHAPE_SIZE);
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "range",
                                                  std::to_string(rangeShapeSize).c_str(), errorMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    int64_t dtypeVal = 0;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* dtypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeAttr);
    dtypeVal = *dtypeAttr;
    if (dtypeVal != OUTPUT_DTYPE) {
        std::string errorMsg = "The value of dtype must be " + std::to_string(OUTPUT_DTYPE);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", std::to_string(dtypeVal).c_str(),
                                              errorMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    int32_t nbins = 0;
    if (!Ops::Base::GetConstInt(context, static_cast<int64_t>(INPUT_NBINS), nbins)) {
        Ops::Base::SetUnknownShape(1, *outputShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(nbins <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "nbins", std::to_string(nbins).c_str(),
                                                      "nbins must be greater than 0"),
                return ge::GRAPH_FAILED);
    outputShape->SetDimNum(1);
    outputShape->SetDim(OUTPUT_IDX, nbins);
    OP_LOGD(context->GetNodeName(), "Output shape = %s", Ops::Base::ToString(*outputShape).c_str());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HistogramFixedWidth)
    .InferShape(InferShapeForHistogramFixedWidth)
    .InputsDataDependency({INPUT_NBINS});
} // namespace ops
