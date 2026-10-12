/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/stack_ball_query_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class StackBallQueryTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StackBallQuery Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StackBallQuery Tiling TearDown" << std::endl; }
};

// ===== 正常路径 =====

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_fp32_int32)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_fp16_int32)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_fp32_int64)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_fp16_int64)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    uint64_t expectTilingKey = 4;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ===== 非法 dtype 拦截 =====

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_xyz_dtype_double)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_DOUBLE, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_DOUBLE, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_center_dtype_mismatch)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_batch_cnt_dtype_invalid)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_batch_cnt_dtype_mismatch)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ===== 非法 shape 拦截 =====

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_xyz_not_2d)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_center_xyz_not_2d)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3, 1}, {10, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_batch_cnt_not_1d)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_batch_cnt_len_mismatch)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ===== 非法 attrs 拦截 =====

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_sample_num_zero)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_max_radius_zero)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(0.0f)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryTiling, stack_ball_query_tiling_neg_max_radius_negative)
{
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "StackBallQuery",
        {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(-1.0f)),
         gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
