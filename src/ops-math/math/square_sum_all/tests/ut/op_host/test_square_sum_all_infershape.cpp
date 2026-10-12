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
 * \file test_square_sum_all_infershape.cpp
 * \brief SquareSumAll shape inference tests.
 */

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
using ShapeList = std::vector<std::vector<int64_t>>;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape().SetDimNum(dimensions.size());
    storageShape.MutableStorageShape().SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        storageShape.MutableOriginShape().SetDim(i, dimensions[i]);
        storageShape.MutableStorageShape().SetDim(i, dimensions[i]);
    }
    return storageShape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& x1Shape, const std::vector<int64_t>& x2Shape)
{
    using TensorDescription = gert::InfershapeContextPara::TensorDescription;
    const std::vector<TensorDescription> inputs = {
        {MakeStorageShape(x1Shape), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeStorageShape(x2Shape), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<TensorDescription> outputs = {
        {MakeStorageShape({}), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeStorageShape({}), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    return gert::InfershapeContextPara("SquareSumAll", inputs, outputs);
}
} // namespace

class SquareSumAllInfershapeTest : public testing::Test {};

TEST_F(SquareSumAllInfershapeTest, RankOneProducesTwoScalars)
{
    auto context = MakeContext({65}, {65});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, ShapeList{{}, {}});
}

TEST_F(SquareSumAllInfershapeTest, RankEightProducesTwoScalars)
{
    auto context = MakeContext({2, 1, 2, 1, 2, 1, 2, 1}, {2, 1, 2, 1, 2, 1, 2, 1});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, ShapeList{{}, {}});
}

TEST_F(SquareSumAllInfershapeTest, DynamicShapeProducesTwoScalars)
{
    auto context = MakeContext({-1, 32}, {-1, 32});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, ShapeList{{}, {}});
}

TEST_F(SquareSumAllInfershapeTest, DynamicRankProducesTwoScalars)
{
    auto context = MakeContext({-2}, {-2});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, ShapeList{{}, {}});
}

// canndev 的 tiling 显式把 rank-0 当 1 个元素处理（(GetDimNum()==0) ? 1 : GetShapeSize()），
// A5 的支持面不得低于它，故标量输入必须接受。
TEST_F(SquareSumAllInfershapeTest, AcceptsRankZeroScalarInput)
{
    auto context = MakeContext({}, {});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, ShapeList{{}, {}});
}

TEST_F(SquareSumAllInfershapeTest, RejectsRankNine)
{
    auto context = MakeContext({1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    ExecuteTestCase(context, ge::GRAPH_FAILED, ShapeList{});
}

TEST_F(SquareSumAllInfershapeTest, RejectsDifferentRanks)
{
    auto context = MakeContext({2, 3}, {6});
    ExecuteTestCase(context, ge::GRAPH_FAILED, ShapeList{});
}

TEST_F(SquareSumAllInfershapeTest, RejectsDifferentConcreteShapes)
{
    auto context = MakeContext({2, 3}, {3, 2});
    ExecuteTestCase(context, ge::GRAPH_FAILED, ShapeList{});
}

TEST_F(SquareSumAllInfershapeTest, RejectsEmptyTensor)
{
    const std::vector<std::vector<int64_t>> emptyShapes = {
        {0}, {0, 2, 3}, {2, 0, 3}, {2, 3, 0}, {0, 2, 0}, {0, 0},
    };
    for (const auto& shape : emptyShapes) {
        auto context = MakeContext(shape, shape);
        ExecuteTestCase(context, ge::GRAPH_FAILED, ShapeList{});
    }
}

TEST_F(SquareSumAllInfershapeTest, RejectsInvalidNegativeDimension)
{
    auto context = MakeContext({-3, 4}, {-3, 4});
    ExecuteTestCase(context, ge::GRAPH_FAILED, ShapeList{});
}
