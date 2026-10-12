/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/pad_v3_grad_replicate_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class PadV3GradReplicateTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3GradReplicateTiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "PadV3GradReplicateTiling  TearDown" << std::endl;
    }
};

// Scenario: float16 with no H padding and int64 padding, expect tiling key 2101
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_no_h_pad_success)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 64, 62}, {1, 1, 64, 62}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2101;
    std::string expectTilingData =
        "4294967297 274877907008 274877907008 266287972416 274877907008 0 4294967297 81363860455488 1 137438955573 "
        "128 ";
    std::vector<size_t> expectWorkspaces = {16785408};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: unsupported dtype (int32), expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_unsupported_dtype_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 64, 62}, {1, 1, 64, 62}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: float32 with no W padding (padLeft=0, padRight=0, padTop>0), expect key 1110
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_no_w_pad)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 10, 8}, {1, 1, 10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1110);
}

// Scenario: float32 with no H padding (padTop=0, padBottom=0, padLeft>0), expect key 1101
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_no_h_pad)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 8, 10}, {1, 1, 8, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1101);
}

// Scenario: float32 with both H and W padding, small shape (h<=64, w<=64), expect key 1000
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_mini_shape)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1000);
}

// Scenario: float32 with both H and W padding, small H large W (h<=64, w>64), expect key 1100
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_small_h_large_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 100}, {1, 1, 32, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 98}, {1, 1, 30, 98}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1100);
}

// Scenario: float32 with both H and W padding, large H small W (h>64, w<=64), expect key 1010
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_large_h_small_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 32}, {1, 1, 100, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 30}, {1, 1, 98, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1010);
}

// Scenario: float32 with both H and W padding, h>64, w>64, outHeight=1, expect key 11111
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_hw_one)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 32, 33, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 66, 100}, {1, 1, 66, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 1, 98}, {1, 1, 1, 98}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 11111);
}

// Scenario: float32 with both H and W padding, big shape (h>64, w>64), expect key 1111
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_hw_pad_big)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 100}, {1, 1, 100, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 98}, {1, 1, 98, 98}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1111);
}

// Scenario: float16 with no W padding, expect key 2110
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_no_w_pad)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 10, 8}, {1, 1, 10, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2110);
}

// Scenario: float16 with both H and W padding, small shape (h<=64, w<=64), expect key 2000
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_mini_shape)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2000);
}

// Scenario: float16 with both H and W padding, small H large W, expect key 2100
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_small_h_large_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 100}, {1, 1, 32, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 98}, {1, 1, 30, 98}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2100);
}

// Scenario: float16 with both H and W padding, large H small W, expect key 2010
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_large_h_small_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 32}, {1, 1, 100, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 30}, {1, 1, 98, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2010);
}

// Scenario: float16 with both H and W padding, h>64, w>64, outHeight=1, expect key 22222
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_hw_one)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 32, 33, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 66, 100}, {1, 1, 66, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 1, 98}, {1, 1, 1, 98}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 22222);
}

// Scenario: float16 with both H and W padding, big shape, expect key 2111
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float16_hw_pad_big)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 100}, {1, 1, 100, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 98}, {1, 1, 98, 98}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2111);
}

// Scenario: bfloat16 with no W padding, expect key 3110
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_no_w_pad)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 10, 8}, {1, 1, 10, 8}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3110);
}

// Scenario: bfloat16 with no H padding, expect key 3101
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_no_h_pad)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 8, 10}, {1, 1, 8, 10}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3101);
}

// Scenario: bfloat16 with both H and W padding, small shape, expect key 3000
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_mini_shape)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3000);
}

// Scenario: bfloat16 with both H and W padding, small H large W, expect key 3100
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_small_h_large_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 100}, {1, 1, 32, 100}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 98}, {1, 1, 30, 98}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3100);
}

// Scenario: bfloat16 with both H and W padding, large H small W, expect key 3010
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_large_h_small_w)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 32}, {1, 1, 100, 32}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 30}, {1, 1, 98, 30}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3010);
}

// Scenario: bfloat16 with both H and W padding, h>64, w>64, outHeight=1, expect key 33333
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_hw_one)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 32, 33, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 66, 100}, {1, 1, 66, 100}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 1, 98}, {1, 1, 1, 98}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 33333);
}

// Scenario: bfloat16 with both H and W padding, big shape, expect key 3111
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_bf16_hw_pad_big)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 100, 100}, {1, 1, 100, 100}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 98, 98}, {1, 1, 98, 98}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 3111);
}

// Scenario: float16 with int32 padding type, expect tiling to succeed with key 2000
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_int32_padding)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int32_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2000);
}

// Scenario: float32 with int32 padding type, expect tiling to succeed with float template
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_int32_padding)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int32_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1000);
}

// Scenario: large batch*channel exceeding coreNum, expect multi-core distribution
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_multi_core_distribution)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {4, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{10, 10, 32, 32}, {10, 10, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{10, 10, 30, 30}, {10, 10, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 1000);
    // nMulC = 10*10 = 100 > coreNum=4, so blockNum should be 4
    EXPECT_EQ(tilingInfo.blockNum, 4);
}

// Scenario: unsupported padding dtype (DT_FLOAT), expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_unsupported_padding_dtype_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: unsupported mode ("constant"), expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_unsupported_mode_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: mismatched output shape, expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_mismatch_output_shape_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 28, 28}, {1, 1, 28, 28}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: zero coreNum in compileInfo, expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_zero_core_num_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {0, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: zero ubSize in compileInfo, expect tiling to fail
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_zero_ub_size_failed)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 0, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: reflect mode with float dtype, no EDGE_MODE branch matches, tilingKey remains 0
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_float_reflect_mode)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 32, 32}, {1, 1, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

// Scenario: large batch*channel exceeding coreNum with NO_H_PAD, multi-core for NO_H_PAD path
TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_no_h_pad_multi_core)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {4, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{10, 10, 8, 10}, {10, 10, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{10, 10, 8, 8}, {10, 10, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, 2101);
    // nMulC = 10*10=100, for NO_H_PAD multiply by height=8 → 800 > coreNum=4
    EXPECT_EQ(tilingInfo.blockNum, 4);
}
