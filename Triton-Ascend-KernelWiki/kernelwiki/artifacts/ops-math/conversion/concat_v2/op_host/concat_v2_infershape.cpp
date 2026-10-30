/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_CONCAT_DIM_FOR_CONCAT_V2 = 1;
constexpr size_t INPUT_IDX_FOR_CONCAT_V2 = 0;
template <typename T>
static inline int64_t GetAxisValue(gert::InferShapeContext* context, int64_t dimIdx)
{
    auto tensor = context->GetRequiredInputTensor(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    auto ptr = tensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ptr);
    return static_cast<int64_t>(ptr[0]);
}

static ge::graphStatus InferShape4ConcatV2(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4ConcatV2 Func");
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto xInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_IDX_FOR_CONCAT_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInfo);

    int64_t numInputs = static_cast<int64_t>(xInfo->GetInstanceNum());
    OP_LOGW(context, "ConcatV2 Dynamic input count = %ld", numInputs);
    if (numInputs <= 1) {
        OP_LOGE(context, "ConcatV2 requires at least 2 inputs");
        return ge::GRAPH_FAILED;
    }

    auto concatDimPtr = context->GetRequiredInputDesc(INDEX_CONCAT_DIM_FOR_CONCAT_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);

    auto dtype = concatDimPtr->GetDataType();
    int64_t axis = 0;
    if (dtype == ge::DT_INT32) {
        axis = GetAxisValue<int32_t>(context, INDEX_CONCAT_DIM_FOR_CONCAT_V2);
    } else if (dtype == ge::DT_INT64) {
        axis = GetAxisValue<int64_t>(context, INDEX_CONCAT_DIM_FOR_CONCAT_V2);
    } else {
        OP_LOGE(context, "ConcatV2: unsupported concat_dim dtype %s", Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto firstShape = context->GetDynamicInputShape(INPUT_IDX_FOR_CONCAT_V2, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, firstShape);

    *outShape = *firstShape;

    size_t rank = outShape->GetDimNum();
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        OP_LOGE(context, "Invalid concat_dim=%ld, rank=%zu.", axis, rank);
        return ge::GRAPH_FAILED;
    }

    int64_t newDim = outShape->GetDim(axis);
    for (int64_t i = 1; i < numInputs; ++i) {
        auto shape_i = context->GetDynamicInputShape(INPUT_IDX_FOR_CONCAT_V2, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape_i);
        newDim += shape_i->GetDim(axis);
    }

    outShape->SetDim(axis, newDim);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ConcatV2).InferShape(InferShape4ConcatV2).InputsDataDependency({INDEX_CONCAT_DIM_FOR_CONCAT_V2});
} // namespace ops