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
 * \file reduce_std_with_mean_infershape.cpp
 * \brief InferShape registration for ReduceStdWithMean op
 *
 * Registers IMPL_OP_INFERSHAPE so that INFER_SHAPE macro and graph-mode
 * shape inference correctly derive the output shape from self shape,
 * dim attribute, and keepdim attribute.
 */

#include "op_host/infershape_reduce_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

const size_t INPUT_INDEX_SELF = 0;
const size_t INPUT_INDEX_MEAN = 1;
const size_t ATTR_INDEX_DIM = 0;
const size_t ATTR_INDEX_KEEPDIM = 2;
const size_t OUTPUT_INDEX_OUT = 0;

static ge::graphStatus InferShape4ReduceStdWithMean(gert::InferShapeContext* context)
{
    // Read self input shape
    auto inputShape = context->GetInputShape(INPUT_INDEX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    // Read output shape to set
    auto outShape = context->GetOutputShape(OUTPUT_INDEX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    // Unknown rank: propagate unknown shape and return
    if (Ops::Base::IsUnknownRank(*inputShape)) {
        Ops::Base::SetUnknownRank(*outShape);
        return GRAPH_SUCCESS;
    }

    int64_t inputDimNum = inputShape->GetDimNum();

    // Read dim attribute
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dimPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);

    // Build axes array from dim attribute
    std::vector<int64_t> axes;
    int64_t axesSize = dimPtr->GetSize();
    if (axesSize == 0) {
        // Empty dim means reduce all dimensions
        axes.resize(inputDimNum);
        for (int64_t i = 0; i < inputDimNum; i++) {
            axes[i] = i;
        }
    } else {
        auto axesData = static_cast<const int64_t*>(dimPtr->GetData());
        axes.resize(axesSize);
        for (int64_t i = 0; i < axesSize; i++) {
            axes[i] = axesData[i];
        }
    }

    // Read keepdim attribute
    bool keepDims = false;
    const bool* attrKeepDims = attrs->GetAttrPointer<bool>(ATTR_INDEX_KEEPDIM);
    if (attrKeepDims != nullptr) {
        keepDims = *attrKeepDims;
    }

    // Compute output shape using framework helpers
    ge::graphStatus inferStat;
    if (keepDims) {
        inferStat = Ops::Base::ReduceDimsWithKeepDims<int64_t>(inputShape, &axes[0], static_cast<int32_t>(axes.size()),
                                                               outShape);
    } else {
        inferStat = Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(inputShape, &axes[0],
                                                                  static_cast<int32_t>(axes.size()), outShape);
    }

    OP_LOGD(context->GetNodeName(), "ReduceStdWithMean InferShape: input=%s, keepDims=%d, out=%s",
            Ops::Base::ToString(*inputShape).c_str(), static_cast<int>(keepDims),
            Ops::Base::ToString(*outShape).c_str());

    return inferStat;
}

IMPL_OP_INFERSHAPE(ReduceStdWithMean).InferShape(InferShape4ReduceStdWithMean);

} // namespace ops
