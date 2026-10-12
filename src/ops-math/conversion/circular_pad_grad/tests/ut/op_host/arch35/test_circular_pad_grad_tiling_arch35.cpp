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
 * \file test_circular_pad_grad_tiling_arch35.cpp
 * \brief UT tests for circular_pad_grad arch35 tiling
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace Ops::Math;

// Dummy attrs for CircularPadGrad - the op sets isCircularPadGrad_ internally,
// but the shared PadV3GradACTiling code requires non-null attrs from GetAttrs()
static const auto DUMMY_ATTRS = std::vector<gert::TilingContextPara::OpAttr>{
    gert::TilingContextPara::OpAttr("mode", AnyValue::CreateFrom<std::string>("circular")),
    gert::TilingContextPara::OpAttr("paddings_contiguous", AnyValue::CreateFrom<bool>(true)),
};

class CircularPadGradTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CircularPadGradTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CircularPadGradTilingArch35 TearDown" << std::endl; }
};

// Test scenario: circular grad with 1D FP32 shape and small paddings, expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_1d_fp32_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {2, 3};
    gert::StorageShape xShape = {{20}, {20}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{15}, {15}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with 2D FP16 shape, expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_2d_fp16_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {5, 5, 10, 10};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{90, 180}, {90, 180}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with 3D BF16 shape, expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_3d_bf16_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {2, 1, 3, 2, 1, 5};
    gert::StorageShape xShape = {{50, 60, 30}, {50, 60, 30}};
    gert::StorageShape paddingsShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{46, 55, 24}, {46, 55, 24}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BF16, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with large 2D shape, expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_large_shape_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {1, 1, 2, 2};
    gert::StorageShape xShape = {{1000, 500}, {1000, 500}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{998, 496}, {998, 496}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with INT32 paddings, expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_int32_paddings_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int32_t> paddingsValue = {3, 3};
    gert::StorageShape xShape = {{20}, {20}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{14}, {14}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with all-zero paddings (isPadAllNegative path), expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_all_zero_paddings_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {0, 0, 0, 0};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 200}, {100, 200}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with mixed zero/positive paddings (mixed pad path), expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_mixed_paddings_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {0, 0, 3, 3};
    gert::StorageShape xShape = {{100, 200}, {100, 200}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{100, 194}, {100, 194}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with all-negative paddings (isPadAllNegative path), expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_negative_paddings_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {-1, -1, -1, -1};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{12, 22}, {12, 22}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with mixed positive/negative paddings (mixed pad SIMT path), expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_mixed_positive_negative_paddings_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {-1, -1, 3, 3};
    gert::StorageShape xShape = {{10, 20}, {10, 20}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{11, 14}, {11, 14}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with empty tensor (dim=0), expect SIMT path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_empty_tensor_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {0, 0, 0, 0};
    gert::StorageShape xShape = {{0, 10}, {0, 10}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{0, 10}, {0, 10}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20, expectWorkspaces);
}

// Test scenario: circular grad with big shape exceeding INT32_MAX, expect SIMT big shape path to succeed
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_big_shape_success)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {1, 1};
    gert::StorageShape xShape = {{2200000000}, {2200000000}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2199999998}, {2199999998}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4, expectWorkspaces);
}

// Test scenario: circular grad with pad larger than inferred output shape, expect tiling to fail
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_pad_too_large_fail)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {50, 50};
    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{0}, {0}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: circular grad with invalid paddings dtype (DT_FLOAT), expect tiling to fail
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_invalid_paddings_dtype_fail)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<float> paddingsValue = {1.0f, 1.0f};
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{8}, {8}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: circular grad with dim > 5, expect tiling to fail (non-constant max dim)
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_dim_exceeds_max_fail)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}};
    gert::StorageShape paddingsShape = {{6, 2}, {6, 2}};
    gert::StorageShape yShape = {{2, 2, 2, 2, 2, 0}, {2, 2, 2, 2, 2, 0}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: circular grad with negative output dimension after subtracting pads, expect tiling to fail
TEST_F(CircularPadGradTilingArch35, circular_pad_grad_tiling_negative_output_dim_fail)
{
    struct CircularPadGradCompileInfo {};
    CircularPadGradCompileInfo compileInfo = {};

    std::vector<int64_t> paddingsValue = {60, 60};
    gert::StorageShape xShape = {{50}, {50}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{0}, {0}};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, DUMMY_ATTRS, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
