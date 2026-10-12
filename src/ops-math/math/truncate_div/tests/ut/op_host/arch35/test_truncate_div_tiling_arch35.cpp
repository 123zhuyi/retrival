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
 * \file test_truncate_div_tiling_arch35.cpp
 * \brief TruncateDiv tiling test
 */

#include "math/truncate_div/op_host/arch35/truncate_div_tiling_arch35.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;
using namespace ge;
using namespace Ops::Base;

class TruncateDivTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TruncateDivTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TruncateDivTilingTest TearDown" << std::endl; }
};

TEST_F(TruncateDivTilingTest, truncate_div_fp16_fp_scalar)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000111;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_fp_fp1)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'00000000'00001000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_bf16_1)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b0'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_f32_scalar_3)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_f16_scalar_4)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_bf16_scalar_5)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, test_ez0020_scalar_dtype_int32)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, x2.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

// FP16 x1 with scalar FP32 x2=0.0, reciprocal is INFINITY, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_scalar_zero_divisor)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<float> x2Data = {0.0f};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2Data.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP16 x1 with scalar FP32 x2=-0.0, reciprocal is -INFINITY, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_scalar_negative_zero)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<float> x2Data = {-0.0f};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2Data.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with scalar FP32 x2=INFINITY, reciprocal is 0.0, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_scalar_infinity)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<float> x2Data = {std::numeric_limits<float>::infinity()};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2Data.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with scalar FP32 x2=-INFINITY, reciprocal is -0.0, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_scalar_neg_infinity)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<float> x2Data = {-std::numeric_limits<float>::infinity()};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2Data.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with INT32 x2, non-scalar, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_float_int32)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with FP16 x2, non-scalar, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_float_float16)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with scalar FP16 x2, tiling succeeds via scalar multiply path
TEST_F(TruncateDivTilingTest, truncate_div_float_float16_scalar)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    // FP16 bits for 2.0: sign=0, exponent=16, mantissa=0 → 0x4000
    std::vector<uint16_t> x2Data = {0x4000};
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, x2Data.data()},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT8 x1 and x2 same type, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_int8_same)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// UINT8 x1 and x2 same type, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_uint8_same)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT16 x1 and x2 same type, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_int16_same)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT32 x1 and x2 same type, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_int32_same)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT64 x1 and x2 same type, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_int64_same)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT32 x1 with FP32 x2, tiling succeeds
TEST_F(TruncateDivTilingTest, truncate_div_int32_float)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// FP32 x1 with INT8 x2, unsupported dtype combination, tiling fails
TEST_F(TruncateDivTilingTest, truncate_div_float_int8_unsupported)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Scalar FP32 x2 with null const data, GetConstData fails, tiling fails
TEST_F(TruncateDivTilingTest, truncate_div_scalar_null_data)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

TEST_F(TruncateDivTilingTest, test_ez0021_dtype_combination_double)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    gert::TilingContextPara tilingContextPara("TruncateDiv",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
