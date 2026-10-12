/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/broadcast/broadcast_tiling_base.h"

using namespace std;
using namespace ge;

class BiasTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BiasTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BiasTilingTest TearDown" << std::endl; }
};

TEST_F(BiasTilingTest, bias_test_tiling_fp16_tail_broadcast)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, bias_test_tiling_fp32_num_axes_neg1)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{4, 3, 8}, {4, 3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 3, 8}, {4, 3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, bias_test_tiling_bf16_non_tail_broadcast)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3, 1}, {3, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, bias_test_tiling_scalar_broadcast)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{7, 11, 13}, {7, 11, 13}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{7, 11, 13}, {7, 11, 13}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, bias_test_tiling_bias_from_blob_false)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3, 5, 7}, {2, 3, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 7}, {5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 5, 7}, {2, 3, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, bias_test_tiling_negative_axis)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, EMPTY_EXPECT_TILING_DATA, {16777216});
}

TEST_F(BiasTilingTest, tiling_invalid_axis)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {0});
}

TEST_F(BiasTilingTest, tiling_invalid_num_axes)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {0});
}

TEST_F(BiasTilingTest, tiling_mixed_dtype)
{
    Ops::Base::BroadcastCompileInfo compileInfo = {true, 36, 253952};
    gert::TilingContextPara tilingContextPara(
        "Bias",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("bias_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {0});
}
