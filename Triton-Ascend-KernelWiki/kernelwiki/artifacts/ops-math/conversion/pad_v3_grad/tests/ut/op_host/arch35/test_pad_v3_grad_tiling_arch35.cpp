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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV3GradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV3GradTilingTest TearDown" << std::endl; }
};

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_1)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{60}, {60}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    std::vector<int32_t> pad_value = {6, 4};
    gert::StorageShape yShape = {{50}, {50}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 18;
    string expectTilingData = "1 0 0 60 0 0 0 0 0 0 0 50 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 6 0 0 0 0 0 0 0 "
                              "4 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_2)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> paddingsValue = {20, 25, 10, 18};
    gert::StorageShape xShape = {{1284, 1053}, {1284, 1053}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{1254, 1010}, {1254, 1010}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectedTilingKey = 35;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_3)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> paddingsValue = {15, 22, 11, 1, 22, 3};

    gert::StorageShape xShape = {{146, 7090, 29}, {146, 7090, 29}};
    gert::StorageShape paddingsShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{130, 7046, 15}, {130, 7046, 15}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectedTilingKey = 20;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_edge_4)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "edge", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> paddingsValue = {5, 3, 4, 4, 4, 4, 5, 4};
    gert::StorageShape xShape = {{78, 81, 82, 78}, {78, 81, 82, 78}};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{70, 73, 74, 69}, {70, 73, 74, 69}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectedTilingKey = 17;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_const_5)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> paddingsValue = {24, 14, 6, 17, 6, 7, 15, 17, 5, 6};
    gert::StorageShape yShape = {{4, 45, 42, 21, 10}, {4, 45, 42, 21, 10}};
    gert::StorageShape paddingsShape = {{5, 2}, {5, 2}};
    gert::StorageShape xShape = {{42, 68, 55, 53, 21}, {42, 68, 55, 53, 21}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectedTilingKey = 16;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_simt_3d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{15, 15, 15}, {15, 15, 15}};
    gert::StorageShape padShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{13, 13, 13}, {13, 13, 13}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 18;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_big_1d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1000, 1000};
    gert::StorageShape xShape = {{200000}, {200000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{198000}, {198000}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_normal_4d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {2, 2, 2, 2, 2, 2, 50, 50};
    gert::StorageShape xShape = {{20, 20, 20, 200}, {20, 20, 20, 200}};
    gert::StorageShape padShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{16, 16, 16, 100}, {16, 16, 16, 100}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 34;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_small_2d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {10, 10, 2, 2};
    gert::StorageShape xShape = {{5000, 20}, {5000, 20}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{4980, 16}, {4980, 16}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 66;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_simt_4d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{8, 8, 8, 8}, {8, 8, 8, 8}};
    gert::StorageShape padShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{6, 6, 6, 6}, {6, 6, 6, 6}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 19;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_big_3d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 1000, 1000};
    gert::StorageShape xShape = {{10, 10, 12000}, {10, 10, 12000}};
    gert::StorageShape padShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{8, 8, 10000}, {8, 8, 10000}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_normal_2d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {500, 500, 50, 50};
    gert::StorageShape xShape = {{10000, 500}, {10000, 500}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{9000, 400}, {9000, 400}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 35;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_small_5d)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    gert::StorageShape xShape = {{20, 20, 20, 20, 20}, {20, 20, 20, 20, 20}};
    gert::StorageShape padShape = {{5, 2}, {5, 2}};
    gert::StorageShape yShape = {{16, 16, 16, 16, 16}, {16, 16, 16, 16, 16}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 67;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ============================================================================
// Invalid parameter test cases
// ============================================================================

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_invalid_pad_too_large)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {40, 40};
    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{20}, {20}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_invalid_pad_too_large)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {40, 40};
    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{20}, {20}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// ============================================================================
// Invalid parameter test cases - error validation
// ============================================================================

// Test scenario: input dimension exceeds max (dim > 8), expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_dim_exceeds_max)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape padShape = {{9, 2}, {9, 2}};
    gert::StorageShape yShape = {{2, 2, 2, 2, 2, 2, 2, 2, 0}, {2, 2, 2, 2, 2, 2, 2, 2, 0}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: reflect mode with dim > 5, expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_dim_exceeds_non_constant_max)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}};
    gert::StorageShape padShape = {{6, 2}, {6, 2}};
    gert::StorageShape yShape = {{2, 2, 2, 2, 2, 0}, {2, 2, 2, 2, 2, 0}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: invalid paddings dtype (DT_FLOAT), expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_invalid_paddings_dtype)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<float> pad_value = {1.0f, 1.0f};
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{8}, {8}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_FLOAT, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: circular mode with pad larger than output, expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_invalid_pad_too_large)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {50, 50};
    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{0}, {0}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: edge mode with zero output dim and nonzero pad, expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_edge_zero_output_with_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "edge", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {5, 5};
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{0}, {0}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: output dim becomes negative after subtracting pads, expect tiling to fail
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_negative_output_dim)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {100, 100};
    gert::StorageShape xShape = {{50}, {50}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{0}, {0}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// ============================================================================
// Empty tensor test cases
// ============================================================================

// Test scenario: edge mode with empty tensor (dim=0), expect tiling to succeed with SIMT
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_edge_empty_tensor)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "edge", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 17, expectWorkspaces);
}

// Test scenario: reflect mode with empty tensor (dim=0), expect tiling to fail due to reflect check
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_empty_tensor)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}

// Test scenario: circular mode with empty tensor, expect tiling to succeed with SIMT
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_empty_tensor)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: constant mode with empty tensor, expect tiling to succeed with SIMT
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_empty_tensor)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16, expectWorkspaces);
}

// ============================================================================
// SIMT big shape test cases
// ============================================================================

// Test scenario: reflect mode with shape exceeding INT32_MAX after expansion, expect big shape SIMD path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_big_shape_simt)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1};
    gert::StorageShape xShape = {{2200000000}, {2200000000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2199999998}, {2199999998}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10, expectWorkspaces);
}

// Test scenario: edge mode with shape exceeding INT32_MAX, expect SIMT with big shape flag
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_edge_big_shape_simt)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "edge", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1};
    gert::StorageShape xShape = {{2200000000}, {2200000000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2199999998}, {2199999998}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 25, expectWorkspaces);
}

// Test scenario: circular mode with shape exceeding INT32_MAX, expect SIMT with big shape flag
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_big_shape_simt)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {1, 1};
    gert::StorageShape xShape = {{2200000000}, {2200000000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2199999998}, {2199999998}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4, expectWorkspaces);
}

// Test scenario: constant mode with shape exceeding INT32_MAX, expect SIMT with big shape flag
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_big_shape_simt)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1};
    gert::StorageShape xShape = {{2200000000}, {2200000000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2199999998}, {2199999998}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 24, expectWorkspaces);
}

// ============================================================================
// Dimension collapse test cases
// ============================================================================

// Test scenario: reflect mode with all-1 shape and no padding, expect collapse to single axis
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_all_one_shape)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{1, 1}, {1, 1}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// Test scenario: constant mode with consecutive zero-pad axes that collapse together
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_consecutive_zero_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 2, 3, 0, 0, 0, 0};
    gert::StorageShape xShape = {{10, 20, 30, 40}, {10, 20, 30, 40}};
    gert::StorageShape padShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{10, 15, 30, 40}, {10, 15, 30, 40}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16, expectWorkspaces);
}

// Test scenario: reflect mode with 1-axis elimination (inShape=1, padFront+padBack=0)
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_one_axis_elimination)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 2, 2};
    gert::StorageShape xShape = {{1, 10}, {1, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{1, 6}, {1, 6}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// INT64 paddings test cases
// ============================================================================

// Test scenario: reflect mode with INT64 paddings, expect tiling to succeed
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_int64_paddings)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {3, 3};
    gert::StorageShape xShape = {{20}, {20}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{14}, {14}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// SIMD mirror 1D test cases
// ============================================================================

// Test scenario: reflect mode with 1D shape where lastDim * dtypeBytes > vectorSize/2, expect SIMD big cut
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_1d_big_last_dim)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {10, 10};
    gert::StorageShape xShape = {{50000}, {50000}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{49980}, {49980}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2, expectWorkspaces);
}

// ============================================================================
// Paddings contiguous=false test cases
// ============================================================================

// Test scenario: reflect mode with non-contiguous paddings, expect tiling to succeed
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_non_contiguous)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {2, 3, 4, 5};
    gert::StorageShape xShape = {{20, 30}, {20, 30}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{14, 21}, {14, 21}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Circular SIMT small shape test
// ============================================================================

// Test scenario: circular mode with small shape (<48K elements), expect SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_simt_small)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {1, 2};
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{7}, {7}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// ============================================================================
// BF16 dtype test cases
// ============================================================================

// Test scenario: reflect mode with BF16 dtype, expect tiling to succeed
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_bf16)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {2, 2, 3, 3};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{96, 194}, {96, 194}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Circular SIMD test case (large shape)
// ============================================================================

// Test scenario: circular mode with large shape, expect tiling to succeed
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_large_shape)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {1, 1, 2, 2};
    gert::StorageShape xShape = {{1000, 500}, {1000, 500}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{998, 496}, {998, 496}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// ============================================================================
// Mirror mode with negative-only paddings
// ============================================================================

// Test scenario: reflect mode with all-zero paddings (isPadAllNegative), expect SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_all_zero_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 200}, {100, 200}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Mirror mode mixed pad test case
// ============================================================================

// Test scenario: reflect mode with mixed paddings (some zero, some positive), expect SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_mixed_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 5, 5};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 190}, {100, 190}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Circular mode mixed/negative paddings test cases
// ============================================================================

// Test scenario: circular mode with all-zero paddings, expect SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_all_zero_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 200}, {100, 200}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular mode with mixed paddings (some zero, some positive), expect SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_mixed_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {0, 0, 3, 3};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 194}, {100, 194}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// ============================================================================
// Symmetric mode with empty tensor (triggers EmptyTensorCollapse in mirror path)
// ============================================================================

// Test scenario: symmetric mode with empty tensor, expect EmptyTensorCollapse + SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_empty_tensor)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "symmetric", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 19, expectWorkspaces);
}

// ============================================================================
// Constant mode with 6-7 dims (triggers DoFindSplitAxisByInput high dim path)
// ============================================================================

// Test scenario: constant mode with 6-dim shape, expect high-dim split axis path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_6d_shape)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{2, 2, 2, 2, 200, 200}, {2, 2, 2, 2, 200, 200}};
    gert::StorageShape padShape = {{6, 2}, {6, 2}};
    gert::StorageShape yShape = {{2, 2, 2, 2, 198, 198}, {2, 2, 2, 2, 198, 198}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16, expectWorkspaces);
}

// Test scenario: constant mode with 7-dim shape with large last dims, expect split axis tuning
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_7d_shape)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2};
    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 100, 100}, {2, 2, 2, 2, 2, 100, 100}};
    gert::StorageShape padShape = {{7, 2}, {7, 2}};
    gert::StorageShape yShape = {{2, 2, 2, 2, 2, 96, 96}, {2, 2, 2, 2, 2, 96, 96}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16, expectWorkspaces);
}

// ============================================================================
// Mirror mode with negative paddings (isPadAllNegative_ path)
// ============================================================================

// Test scenario: reflect mode with all-negative paddings, expect isPadAllNegative SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_negative_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {-1, -1, -1, -1};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{12, 22}, {12, 22}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// Test scenario: reflect mode with mixed positive/negative paddings, expect mixed SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_mixed_positive_negative_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {-1, -1, 2, 2};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{11, 16}, {11, 16}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Circular mode with negative/mixed paddings
// ============================================================================

// Test scenario: circular mode with all-negative paddings, expect isPadAllNegative SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_negative_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {-1, -1, -1, -1};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{12, 22}, {12, 22}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular mode with mixed positive/negative paddings, expect mixed SIMT path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_mixed_positive_negative_pad)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "circular", 0, 0, 0, false, "ascend950"};

    std::vector<int64_t> pad_value = {-1, -1, 3, 3};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{11, 14}, {11, 14}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// ============================================================================
// Mirror SIMD with large last dim to trigger additionTileSize and normal path
// ============================================================================

// Test scenario: reflect mode SIMD with large last dim (>64), expect additionTileSize path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_simd_large_last_dim)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {1, 1, 1, 1, 2, 2};
    gert::StorageShape xShape = {{500, 500, 100}, {500, 500, 100}};
    gert::StorageShape padShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{498, 498, 96}, {498, 498, 96}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 34, expectWorkspaces);
}

// Test scenario: reflect mode with small last dim, expect SIMT path (shape too small for SIMD)
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_simd_small_last_dim)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{8, 8, 8, 6}, {8, 8, 8, 6}};
    gert::StorageShape padShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{8, 6, 6, 4}, {8, 6, 6, 4}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 18, expectWorkspaces);
}

// ============================================================================
// Constant mode 8-dim shape (max supported dim) for high-dim path coverage
// ============================================================================

// Test scenario: constant mode with 8-dim shape (max dims), expect tiling to succeed
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_constant_8d_shape)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2};
    gert::StorageShape xShape = {{2, 3, 4, 5, 6, 7, 8, 200}, {2, 3, 4, 5, 6, 7, 8, 200}};
    gert::StorageShape padShape = {{8, 2}, {8, 2}};
    gert::StorageShape yShape = {{2, 3, 4, 5, 6, 7, 6, 196}, {2, 3, 4, 5, 6, 7, 6, 196}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 16, expectWorkspaces);
}

// ============================================================================
// Reflect SIMD with FP16 dtype to trigger different buffer calculations
// ============================================================================

// Test scenario: reflect mode SIMD with FP16 and large shape, expect SIMD normal path
TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_simd_fp16_large)
{
    PadV3GradCompileInfo compileInfo = {0, 0, 0, 0, 0, false, "reflect", 0, 0, 0, false, "ascend950"};

    std::vector<int32_t> pad_value = {2, 2, 2, 2, 2, 2, 2, 2};
    gert::StorageShape xShape = {{20, 20, 20, 200}, {20, 20, 20, 200}};
    gert::StorageShape padShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{16, 16, 16, 196}, {16, 16, 16, 196}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 34, expectWorkspaces);
}
