/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
constexpr char kOpType[] = "GetDynamicDims";

gert::InfershapeContextPara BuildGetDynamicDimsPara(size_t inputNum, const std::vector<int64_t>& shapeInfo, int64_t n,
                                                    ge::DataType dtype = ge::DT_INT64)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    inputs.reserve(inputNum);
    for (size_t i = 0U; i < inputNum; ++i) {
        inputs.push_back({{{1}, {1}}, dtype, ge::FORMAT_ND});
    }

    return gert::InfershapeContextPara(
        kOpType, inputs, {{{{}, {}}, dtype, ge::FORMAT_ND}},
        {{"shape_info", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(shapeInfo)},
         {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(n)}},
        {static_cast<uint32_t>(inputNum)}, {1U});
}

} // namespace

class GetDynamicDimsInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GetDynamicDimsInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GetDynamicDimsInfershapeTest TearDown" << std::endl; }
};

TEST_F(GetDynamicDimsInfershapeTest, InferShapeSuccess)
{
    auto para = BuildGetDynamicDimsPara(2U, {2, -1, 4, -1}, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2}});
}

TEST_F(GetDynamicDimsInfershapeTest, InferShapeCountsAllUnknownDims)
{
    auto para = BuildGetDynamicDimsPara(3U, {-1, 3, -1, 5, -1}, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3}});
}

TEST_F(GetDynamicDimsInfershapeTest, InferShapeFailedWhenInputNumNotEqualN)
{
    auto para = BuildGetDynamicDimsPara(2U, {2, -1, 4}, 3);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GetDynamicDimsInfershapeTest, InferShapeFailedWhenNoUnknownDim)
{
    auto para = BuildGetDynamicDimsPara(1U, {2, 3, 4}, 1);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
