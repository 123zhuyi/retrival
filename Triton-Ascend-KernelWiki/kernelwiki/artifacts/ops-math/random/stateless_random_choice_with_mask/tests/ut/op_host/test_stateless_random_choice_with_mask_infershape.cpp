/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

class StatelessRandomChoiceWithMaskProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessRandomChoiceWithMaskProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessRandomChoiceWithMaskProtoTest TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(
    std::vector<std::vector<int64_t>> expectResults, const std::vector<gert::StorageShape>& inputShapes,
    std::vector<gert::StorageShape*>& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& xStorageShape = inputShapes[0];
    const auto& countStorageShape = inputShapes[1];
    const auto& seedStorageShape = inputShapes[2];
    const auto& offsetStorageShape = inputShapes[3];

    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xStorageShape, (gert::Tensor*)&countStorageShape, (gert::Tensor*)&seedStorageShape,
        (gert::Tensor*)&offsetStorageShape};
    std::vector<gert::StorageShape*> outputShapes = outStorageShape;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("StatelessRandomChoiceWithMask")
                             .NodeIoNum(4, 2)
                             .NodeInputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("StatelessRandomChoiceWithMask")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_unknownrank)
{
    std::vector<gert::StorageShape> inputShapes = {{{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}};
    std::vector<std::vector<int64_t>> expectResults = {{-2}, {-2}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_unknownshape)
{
    std::vector<gert::StorageShape> inputShapes = {{{-1}, {-1}}, {{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, -1}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_1d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3}, {3}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 1}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_2d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 3}, {3, 3}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 2}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_3d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 3, 1}, {3, 3, 1}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 3}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_4d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 4, 5, 6}, {3, 4, 5, 6}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 4}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_5d)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 5}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}
