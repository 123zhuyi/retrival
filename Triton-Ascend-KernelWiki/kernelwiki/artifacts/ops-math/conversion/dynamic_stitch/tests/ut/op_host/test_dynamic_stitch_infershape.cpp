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
 * \file test_dynamic_stitch_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DynamicStitchTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicStitchTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicStitchTest TearDown" << std::endl;
    }
};

// N=1, indices has known positive values, x has extra dims
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_n1_known)
{
    std::vector<int32_t> indicesValues = {0, 1, 2, 5, 8};

    gert::StorageShape indicesShape = {{5}, {5}};
    gert::StorageShape xShape = {{5, 62}, {5, 62}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND, true, indicesValues.data());
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // max(indices) = 8, so output dim0 = 8 + 1 = 9
    // constant = x.shape - indices.shape = [62], so output = [9, 62]
    std::vector<std::vector<int64_t>> expectOutputShape = {{9, 62}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=2, both inputs have same extra dims, indices have known values
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_n2_known)
{
    std::vector<int32_t> indices0Values = {0, 3, 5};
    std::vector<int32_t> indices1Values = {10, 15, 20};

    gert::StorageShape indices0Shape = {{3}, {3}};
    gert::StorageShape x0Shape = {{3, 32}, {3, 32}};
    gert::StorageShape indices1Shape = {{3}, {3}};
    gert::StorageShape x1Shape = {{3, 32}, {3, 32}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices0(
        indices0Shape, ge::DT_INT32, ge::FORMAT_ND, true, indices0Values.data());
    gert::InfershapeContextPara::TensorDescription x0(
        x0Shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices1(
        indices1Shape, ge::DT_INT32, ge::FORMAT_ND, true, indices1Values.data());
    gert::InfershapeContextPara::TensorDescription x1(
        x1Shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "DynamicStitch", {indices0, indices1, x0, x1}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)}}, {2, 2}, {1});
    // max(indices) across all = max(5, 20) = 20, so output dim0 = 21
    // constant = [32], output = [21, 32]
    std::vector<std::vector<int64_t>> expectOutputShape = {{21, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=1, indices is scalar (empty shape / no extra dims)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_indices_scalar)
{
    std::vector<int32_t> indicesValues = {7};

    gert::StorageShape indicesShape = {{}, {}};
    gert::StorageShape xShape = {{32}, {32}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND, true, indicesValues.data());
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // max(indices) = 7, output dim0 = 8, constant = [32], output = [8, 32]
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=1, indices shape [0] (zero-element indices)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_zero_indices)
{
    gert::StorageShape indicesShape = {{0}, {0}};
    gert::StorageShape xShape = {{0, 5}, {0, 5}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // All indices have zero shape, so output = [0]
    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=1, x has -1 in the suffix (extra dims) → output has -2 (unknown rank)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_unknown_dim)
{
    gert::StorageShape indicesShape = {{4, 128}, {4, 128}};
    gert::StorageShape xShape = {{4, 128, -1}, {4, 128, -1}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // -1 found in suffix dims (dims = [-1] after StartWith) → output is [-2]
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=1, x shape has -2 (unknown rank)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_unknown_rank)
{
    gert::StorageShape indicesShape = {{24, 4, 128}, {24, 4, 128}};
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // CheckValidShapeSize returns false for unknown rank → loop continues without processing
    // No valid shapes processed, zeroShapeNum != numIndices, so falls through
    // commonShape stays as [-2], maxIndex stays -1, SetOuputShape gives [0, -2]
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, -2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// N=1, indices dim > x dim (error case)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_indices_dim_larger_than_x)
{
    gert::StorageShape indicesShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // indicesDimNum (3) > xDimNum (2) → error
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// N=2, different extra shapes between the two inputs (error case)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_different_extra_dims)
{
    std::vector<int32_t> indices0Values = {0, 3};
    std::vector<int32_t> indices1Values = {10, 15};

    gert::StorageShape indices0Shape = {{2}, {2}};
    gert::StorageShape x0Shape = {{2, 32}, {2, 32}};
    gert::StorageShape indices1Shape = {{2}, {2}};
    gert::StorageShape x1Shape = {{2, 64}, {2, 64}}; // different extra dim: 64 vs 32
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices0(
        indices0Shape, ge::DT_INT32, ge::FORMAT_ND, true, indices0Values.data());
    gert::InfershapeContextPara::TensorDescription x0(
        x0Shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices1(
        indices1Shape, ge::DT_INT32, ge::FORMAT_ND, true, indices1Values.data());
    gert::InfershapeContextPara::TensorDescription x1(
        x1Shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "DynamicStitch", {indices0, indices1, x0, x1}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)}}, {2, 2}, {1});
    // x0 extra dim = [32], x1 extra dim = [64] → SameExtraShape fails
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// N=1, indices has negative value (error case)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_negative_index)
{
    std::vector<int32_t> indicesValues = {-1, 2};

    gert::StorageShape indicesShape = {{2}, {2}};
    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND, true, indicesValues.data());
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // indices value -1 is negative → error
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// N=1, indices with no const data (dynamic shape inference — output dim0 is -1)
TEST_F(DynamicStitchTest, dynamic_stitch_infer_shape_no_const_data)
{
    gert::StorageShape indicesShape = {{24, 4, 128}, {24, 4, 128}};
    gert::StorageShape xShape = {{24, 4, 128, 62}, {24, 4, 128, 62}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription indices(
        indicesShape, ge::DT_INT32, ge::FORMAT_ND); // isConst=false, no const data
    gert::InfershapeContextPara::TensorDescription x(
        xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(
        yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("DynamicStitch", {indices, x}, {y},
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}}, {1, 1}, {1});
    // No const indices data → getAllIndicesData=false → output dim0 = -1
    // constant = [62], output = [-1, 62]
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 62}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
