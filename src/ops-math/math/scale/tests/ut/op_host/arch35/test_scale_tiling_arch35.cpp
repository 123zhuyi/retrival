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
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/scale_tiling_arch35.h"

using namespace std;

class ScaleTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScaleTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScaleTilingTest TearDown" << std::endl;
  }
};

TEST_F(ScaleTilingTest, scale_tiling_float_with_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(ScaleTilingTest, scale_tiling_float_no_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(ScaleTilingTest, scale_tiling_float16_with_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_bf16)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_scalar_scale)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_1d)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_4d)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_empty_tensor)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_unsupported_dtype)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_scale_from_blob_false)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        {1, 1, 1}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_negative_axis)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}
