/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "graph/operator_reg.h"
#include "accumulate_nv2_graph_infer.h"

namespace ge {
namespace {

bool IsUnknownRank(const std::vector<int64_t>& dims) { return dims == UNKNOWN_RANK; }

graphStatus MergeBroadcastDim(int64_t lhs, int64_t rhs, int64_t& output)
{
    if (lhs == rhs) {
        // This branch intentionally handles 0 and avoids the legacy 0 % 0
        // SIGFPE in the built-in AccumulateNV2 V1 infer implementation.
        output = lhs;
        return GRAPH_SUCCESS;
    }
    if (lhs == 1) {
        output = rhs;
        return GRAPH_SUCCESS;
    }
    if (rhs == 1) {
        output = lhs;
        return GRAPH_SUCCESS;
    }
    if (lhs == UNKNOWN_DIM || rhs == UNKNOWN_DIM) {
        // A known non-one dimension constrains the broadcast result. If both
        // dimensions are unknown, keep the result unknown.
        output = (lhs == UNKNOWN_DIM && rhs == UNKNOWN_DIM) ? UNKNOWN_DIM : (lhs == UNKNOWN_DIM ? rhs : lhs);
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

} // namespace

namespace accumulate_nv2 {

graphStatus MergeBroadcastShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                                std::vector<int64_t>& output)
{
    if (IsUnknownRank(lhs) || IsUnknownRank(rhs)) {
        output = UNKNOWN_RANK;
        return GRAPH_SUCCESS;
    }

    const size_t outputRank = std::max(lhs.size(), rhs.size());
    output.assign(outputRank, 1);
    const size_t lhsOffset = outputRank - lhs.size();
    const size_t rhsOffset = outputRank - rhs.size();
    for (size_t i = 0; i < outputRank; ++i) {
        const int64_t lhsDim = (i < lhsOffset) ? 1 : lhs[i - lhsOffset];
        const int64_t rhsDim = (i < rhsOffset) ? 1 : rhs[i - rhsOffset];
        if (MergeBroadcastDim(lhsDim, rhsDim, output[i]) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

} // namespace accumulate_nv2

IMPLEMT_COMMON_INFERFUNC(AccumulateNV2LegacyV1InferShape)
{
    int64_t attrN = 0;
    if (op.GetAttr("N", attrN) != GRAPH_SUCCESS || attrN <= 0) {
        return GRAPH_FAILED;
    }

    // GetDynamicInputNum("x") is 0 on the GEIR V1 route even though the
    // flattened dynamic inputs are present. GetInputsSize() is the reliable
    // count used by the legacy registry.
    const size_t inputNum = op.GetInputsSize();
    if (inputNum == 0U || static_cast<int64_t>(inputNum) != attrN) {
        return GRAPH_FAILED;
    }

    const TensorDesc firstInputDesc = op.GetDynamicInputDesc("x", 0U);
    const DataType dataType = firstInputDesc.GetDataType();
    std::vector<int64_t> outputDims = firstInputDesc.GetShape().GetDims();
    for (size_t i = 1; i < inputNum; ++i) {
        const TensorDesc inputDesc = op.GetDynamicInputDesc("x", static_cast<uint32_t>(i));
        if (inputDesc.GetDataType() != dataType) {
            return GRAPH_FAILED;
        }
        std::vector<int64_t> mergedDims;
        if (accumulate_nv2::MergeBroadcastShape(outputDims, inputDesc.GetShape().GetDims(), mergedDims) !=
            GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        outputDims = std::move(mergedDims);
    }

    TensorDesc outputDesc = op.GetOutputDesc(0U);
    outputDesc.SetShape(Shape(outputDims));
    outputDesc.SetDataType(dataType);
    return op.UpdateOutputDesc(0U, outputDesc);
}

// GEIR still invokes the legacy V1 registry for this built-in op name. Register
// the package implementation before libopsproto.so is loaded so zero-element
// broadcast shapes use the safe implementation above instead of 0 % 0.
COMMON_INFER_FUNC_REG(AccumulateNV2, AccumulateNV2LegacyV1InferShape);

} // namespace ge
