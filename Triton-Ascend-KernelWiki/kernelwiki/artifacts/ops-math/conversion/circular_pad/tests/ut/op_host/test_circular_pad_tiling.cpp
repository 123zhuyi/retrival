/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_circular_pad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/circular_pad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class CircularPadTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CircularPadTiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CircularPadTiling  TearDown" << std::endl;
    }
};

// Scenario: 3D input with large positive paddings and FP16 dtype, expect successful tiling
TEST_F(CircularPadTiling, circular_pad_tiling_test_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "CircularPad",
        {{{{1, 1, 300, 300}, {1, 1, 300, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 3, 500, 500}, {1, 3, 500, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 322;
    std::string expectTilingData = "300 300 500 500 100 100 100 100 1 1 1 3 0 1 67200 ";
    std::vector<size_t> expectWorkspaces = {17046016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: unsupported dtype (DT_INT64) for input x, expect tiling to fail
TEST_F(CircularPadTiling, circular_pad_tiling_test_failed)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "CircularPad",
        {{{{1, 1, 300, 300}, {1, 1, 300, 300}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 500, 500}, {1, 1, 500, 500}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: 2D input with FP16 and small shape, expect shapeType=1 and 2D tiling key
TEST_F(CircularPadTiling, circular_pad_2d_fp16_small_shape_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 221;
    std::string expectTilingData = "4 8 6 12 2 2 1 1 0 0 0 0 0 1 192 ";
    std::vector<size_t> expectWorkspaces = {16777984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: 2D input with INT8 dtype, expect successful tiling with INT8 type mode
TEST_F(CircularPadTiling, circular_pad_2d_int8_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 211;
    std::string expectTilingData = "4 8 6 12 2 2 1 1 0 0 0 0 0 1 384 ";
    std::vector<size_t> expectWorkspaces = {16778752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: 2D input with BF16 dtype, expect successful tiling with BF16 type mode
TEST_F(CircularPadTiling, circular_pad_2d_bf16_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 231;
    std::string expectTilingData = "4 8 6 12 2 2 1 1 0 0 0 0 0 1 192 ";
    std::vector<size_t> expectWorkspaces = {16777984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: 2D input with FLOAT dtype, expect successful tiling with FLOAT type mode
TEST_F(CircularPadTiling, circular_pad_2d_float_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 241;
    std::string expectTilingData = "4 8 6 12 2 2 1 1 0 0 0 0 0 1 96 ";
    std::vector<size_t> expectWorkspaces = {16777600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: 2D input with INT32 dtype, expect successful tiling with INT32 type mode
TEST_F(CircularPadTiling, circular_pad_2d_int32_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 251;
    std::string expectTilingData = "4 8 6 12 2 2 1 1 0 0 0 0 0 1 96 ";
    std::vector<size_t> expectWorkspaces = {16777600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: 3D input with all negative paddings that are valid (abs values < dim sizes), expect success with negative
// params
TEST_F(CircularPadTiling, circular_pad_3d_negative_paddings_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {-1, -1, -1, -1, -1, -1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 10, 8}, {4, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 8, 6}, {2, 8, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 321;
    std::string expectTilingData = "10 8 8 6 -1 -1 -1 -1 -1 -1 4 2 0 4 128 ";
    std::vector<size_t> expectWorkspaces = {16779264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: left and right both positive but left exceeds inputW, expect failure
TEST_F(CircularPadTiling, circular_pad_left_right_both_positive_left_exceeds_inputw)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, 10, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 20}, {2, 4, 12, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: left positive and right negative, left exceeds inputW+right, expect failure
TEST_F(CircularPadTiling, circular_pad_left_pos_right_neg_left_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, 6, -3};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 11}, {2, 4, 12, 11}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: left positive and right negative, abs(right) >= inputW, expect failure
TEST_F(CircularPadTiling, circular_pad_left_pos_right_neg_abs_right_exceeds_inputw)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, 1, -8};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 1}, {2, 4, 12, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: left negative and right positive, abs(left) >= inputW, expect failure
TEST_F(CircularPadTiling, circular_pad_left_neg_right_pos_abs_left_exceeds_inputw)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, -8, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 1}, {2, 4, 12, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: left negative and right positive, right exceeds inputW+left, expect failure
TEST_F(CircularPadTiling, circular_pad_left_neg_right_pos_right_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, -3, 6};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 11}, {2, 4, 12, 11}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: left and right both negative, abs(left+right) >= inputW with outputW=0, expect failure
TEST_F(CircularPadTiling, circular_pad_left_right_both_negative_exceeds_inputw)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, -4, -4};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 0}, {2, 4, 12, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top and bottom both positive but top exceeds inputH, expect failure
TEST_F(CircularPadTiling, circular_pad_top_bottom_both_positive_top_exceeds_inputh)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 12, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 23, 10}, {2, 4, 23, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top positive and bottom negative, top exceeds inputH+bottom, expect failure
TEST_F(CircularPadTiling, circular_pad_top_pos_bottom_neg_top_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 8, -3, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 15, 10}, {2, 4, 15, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top positive and bottom negative, abs(bottom) >= inputH, expect failure
TEST_F(CircularPadTiling, circular_pad_top_pos_bottom_neg_abs_bottom_exceeds_inputh)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, -10, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 1, 10}, {2, 4, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top negative and bottom positive, abs(top) >= inputH, expect failure
TEST_F(CircularPadTiling, circular_pad_top_neg_bottom_pos_abs_top_exceeds_inputh)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, -10, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 1, 10}, {2, 4, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top negative and bottom positive, bottom exceeds inputH+top, expect failure
TEST_F(CircularPadTiling, circular_pad_top_neg_bottom_pos_bottom_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, -3, 8, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 15, 10}, {2, 4, 15, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: top and bottom both negative, abs(top+bottom) >= inputH with outputH=0, expect failure
TEST_F(CircularPadTiling, circular_pad_top_bottom_both_negative_exceeds_inputh)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, -5, -5, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 0, 10}, {2, 4, 0, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front and back both positive but front exceeds inputL, expect failure
TEST_F(CircularPadTiling, circular_pad_front_back_both_positive_front_exceeds_inputl)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {3, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 6, 12, 10}, {2, 6, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front positive and back negative, front exceeds inputL+back, expect failure
TEST_F(CircularPadTiling, circular_pad_front_pos_back_neg_front_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {2, -1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 3, 12, 10}, {2, 3, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front positive and back negative, abs(back) >= inputL, expect failure
TEST_F(CircularPadTiling, circular_pad_front_pos_back_neg_abs_back_exceeds_inputl)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, -2, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 1, 12, 10}, {2, 1, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front negative and back positive, abs(front) >= inputL, expect failure
TEST_F(CircularPadTiling, circular_pad_front_neg_back_pos_abs_front_exceeds_inputl)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {-2, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 1, 12, 10}, {2, 1, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front negative and back positive, back exceeds inputL+front, expect failure
TEST_F(CircularPadTiling, circular_pad_front_neg_back_pos_back_exceeds_combined)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {-1, 2, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 3, 12, 10}, {2, 3, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: front and back both negative, abs(front+back) >= inputL with outputL=0, expect failure
TEST_F(CircularPadTiling, circular_pad_front_back_both_negative_exceeds_inputl)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {-1, -1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 0, 12, 10}, {2, 0, 12, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: inputW exceeds UB size limit, expect failure at CheckInput
TEST_F(CircularPadTiling, circular_pad_inputw_exceeds_ub_limit)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "CircularPad",
        {{{{2, 10, 70000}, {2, 10, 70000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{2, 2, 10, 70000}, {2, 2, 10, 70000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: output shape dimensions don't match input+padding formula, expect failure
TEST_F(CircularPadTiling, circular_pad_output_shape_mismatch)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{2, 10, 8}, {2, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{2, 4, 12, 11}, {2, 4, 12, 11}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: paddings tensor has fewer than 4 elements, expect failure at GetShapeAttrsInfo
TEST_F(CircularPadTiling, circular_pad_paddings_less_than_4)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{6, 12}, {6, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: input x has fewer than 2 dimensions, expect failure at GetShapeAttrsInfo
TEST_F(CircularPadTiling, circular_pad_x_dims_less_than_2)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{12}, {12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: output y has fewer than 2 dimensions, expect failure at GetShapeAttrsInfo
TEST_F(CircularPadTiling, circular_pad_y_dims_less_than_2)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 2, 2};
    gert::TilingContextPara tilingContextPara("CircularPad",
                                              {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
                                              {
                                                  {{{12}, {12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
