/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
using Ops::Math::AnyValue;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::InfershapeContextPara BuildCase(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& kernel,
                                      const std::vector<int64_t>& strides, const std::vector<int64_t>& dilations,
                                      const std::vector<int64_t>& pads, const std::string& paddingMode = "CALCULATED")
{
    const gert::StorageShape inputStorageShape = MakeStorageShape(inputShape);
    const gert::StorageShape outputStorageShape = {{}, {}};
    const std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {inputStorageShape, ge::DT_FLOAT16, ge::FORMAT_NCHW},
    };
    const std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {outputStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    const std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"ksizes", AnyValue::CreateFrom<std::vector<int64_t>>(kernel)},
        {"strides", AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
        {"dilations", AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
        {"padding_mode", AnyValue::CreateFrom<std::string>(paddingMode)},
        {"pads", AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
    };
    return gert::InfershapeContextPara("Im2col", inputs, outputs, attrs);
}

class Im2colInfershapeTest : public testing::Test {};

TEST_F(Im2colInfershapeTest, infers_public_three_dimensional_output)
{
    auto context = BuildCase({2, 3, 5, 6}, {3, 2}, {2, 1}, {1, 1}, {1, 1, 0, 0});
    const std::vector<std::vector<int64_t>> expected = {{2, 18, 15}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST_F(Im2colInfershapeTest, propagates_unknown_spatial_dimension)
{
    auto context = BuildCase({2, -1, -1, 6}, {3, 2}, {2, 1}, {1, 1}, {1, 1, 0, 0});
    const std::vector<std::vector<int64_t>> expected = {{2, -1, -1}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST_F(Im2colInfershapeTest, rejects_empty_output_dimension)
{
    auto context = BuildCase({1, 2, 2, 2}, {5, 1}, {1, 1}, {1, 1}, {0, 0, 0, 0});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, rejects_non_calculated_padding_mode)
{
    auto context = BuildCase({1, 2, 4, 4}, {3, 3}, {1, 1}, {1, 1}, {0, 0, 0, 0}, "SAME");
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, rejects_invalid_stride)
{
    auto context = BuildCase({1, 2, 4, 4}, {3, 3}, {0, 1}, {1, 1}, {0, 0, 0, 0});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, rejects_output_channel_overflow)
{
    auto context = BuildCase({1, 2, 1, 1}, {3037000500LL, 3037000500LL}, {1, 1}, {1, 1},
                             {3037000500LL, 3037000500LL, 3037000500LL, 3037000500LL});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
} // namespace
