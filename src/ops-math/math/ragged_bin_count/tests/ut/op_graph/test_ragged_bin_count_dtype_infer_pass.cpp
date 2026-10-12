/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "gtest/gtest.h"

#include "../../../op_graph/ragged_bin_count_graph_infer_internal.h"

namespace {
struct DtypeCase {
    ge::DataType splits;
    ge::DataType values;
    ge::DataType size;
    ge::DataType weights;
    ge::graphStatus expected;
};
} // namespace

TEST(RaggedBinCountGraphInferTest, test_infer_datatype_accepts_only_two_native_combinations)
{
    const std::vector<DtypeCase> cases = {
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_DOUBLE, ge::GRAPH_FAILED},
    };

    for (const auto& dtypeCase : cases) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("RaggedBinCount").OpName("RaggedBinCount");
        builder.IONum(4, 1);
        builder.InputTensorDesc(0, dtypeCase.splits, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, dtypeCase.values, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(2, dtypeCase.size, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(3, dtypeCase.weights, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto contextHolder = builder.Build();
        auto* context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(ops::ragged_bin_count_graph_infer_internal::InferDataType(context), dtypeCase.expected);
        if (dtypeCase.expected == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
        }
    }
}
