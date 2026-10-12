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
 * \file test_arg_max_with_value_infershape.cpp
 * \brief ArgMaxWithValue infer-shape unit tests.
 */

#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ArgMaxWithValue : public testing::Test {};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const std::vector<int64_t>& expectResult, const gert::StorageShape& xShape, int64_t dimension,
                        bool keepDims, gert::StorageShape& indiceShape, gert::StorageShape& valueShape,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&xShape};
    std::vector<gert::StorageShape*> outputShapes = {&indiceShape, &valueShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("ArgMaxWithValue")
                             .NodeIoNum(1, 2)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("dimension", dimension)
                             .Attr("keep_dims", keepDims)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("ArgMaxWithValue")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    // both outputs (indice, values) carry the same reduced shape
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), expectResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(1)), expectResult);
}

TEST_F(ArgMaxWithValue, ArgMaxWithValue_infershape_keepdim_false)
{
    gert::StorageShape xShape = {{2, 100, 4}, {2, 100, 4}};
    gert::StorageShape indiceShape = {};
    gert::StorageShape valueShape = {};
    ExeTestCase({2, 4}, xShape, 1, false, indiceShape, valueShape, ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxWithValue, ArgMaxWithValue_infershape_keepdim_true)
{
    gert::StorageShape xShape = {{2, 100, 4}, {2, 100, 4}};
    gert::StorageShape indiceShape = {};
    gert::StorageShape valueShape = {};
    ExeTestCase({2, 1, 4}, xShape, 1, true, indiceShape, valueShape, ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxWithValue, ArgMaxWithValue_infershape_negative_dim)
{
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape indiceShape = {};
    gert::StorageShape valueShape = {};
    ExeTestCase({8}, xShape, -1, false, indiceShape, valueShape, ge::GRAPH_SUCCESS);
}
