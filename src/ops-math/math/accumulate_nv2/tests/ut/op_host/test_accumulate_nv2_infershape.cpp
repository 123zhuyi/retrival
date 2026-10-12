/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace {
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& runtimeShape = storageShape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    runtimeShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        runtimeShape.SetDim(i, dimensions[i]);
    }
    return storageShape;
}

static gert::InfershapeContextPara MakeInferShapeContext(const std::vector<std::vector<int64_t>>& inputShapes,
                                                         ge::DataType dataType, int64_t attrN = -1)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    for (const auto& shape : inputShapes) {
        inputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {MakeStorageShape({}), dataType, ge::FORMAT_ND},
    };
    int64_t inputNum = static_cast<int64_t>(inputShapes.size());
    if (attrN < 0) {
        attrN = inputNum;
    }
    return gert::InfershapeContextPara(
        "AccumulateNV2", inputs, outputs,
        {gert::InfershapeContextPara::OpAttr("N", Ops::Math::AnyValue::CreateFrom<int64_t>(attrN))},
        {static_cast<uint32_t>(inputNum)}, {1});
}
} // namespace

class AccumulateNV2InferShape : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AccumulateNV2InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AccumulateNV2InferShape TearDown" << std::endl; }
};

TEST_F(AccumulateNV2InferShape, same_shape)
{
    auto context = MakeInferShapeContext({{8, 1024}, {8, 1024}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{8, 1024}});
}

TEST_F(AccumulateNV2InferShape, broadcast_shape)
{
    auto context = MakeInferShapeContext({{2, 1, 5}, {1, 3, 1}, {5}}, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 5}});
}

TEST_F(AccumulateNV2InferShape, scalar_and_tensor)
{
    auto context = MakeInferShapeContext({{}, {7, 11}}, ge::DT_INT32);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{7, 11}});
}

TEST_F(AccumulateNV2InferShape, incompatible_shapes_fail)
{
    auto context = MakeInferShapeContext({{2, 3}, {4, 3}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}

TEST_F(AccumulateNV2InferShape, n_must_be_positive)
{
    auto context = MakeInferShapeContext({{1}}, ge::DT_FLOAT, 0);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}
