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
 * \file test_reflection_pad3d_grad_tiling.cpp
 * \brief UT for ReflectionPad3dGrad tiling function
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../op_host/reflection_pad3d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ReflectionPad3dGradTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingTest  TearDown" << std::endl;
    }
};

// Scenario: float32 with small aligned dimensions, expect SMALL tiling key 100
TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float32_success_case0) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 128, 18, 66, 66}, {2, 128, 18, 66, 66}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 128, 16, 64, 64}, {2, 128, 16, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "549755813890 283467841554 66 343597383760 274877906960 64 0 4294967297 4294967297 4294967297 38482906972208 68719476741 100 ";
    std::vector<size_t> expectWorkspaces = {17268736};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: float32 with mid-size dimensions requiring MID tiling key, expect key 101
TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float32_success_case1) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 64, 18, 130, 130}, {2, 64, 18, 130, 130}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 64, 16, 128, 128}, {2, 64, 16, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "274877906946 558345748498 130 618475290768 549755813904 128 0 4294967297 4294967297 4294967297 38482906972208 137438953474 101 ";
    std::vector<size_t> expectWorkspaces = {17661952};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: float16 with flat dimensions (height <= 32), expect FLAT tiling key 202
TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float16_success_case0) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 128, 18, 22, 1220}, {2, 128, 18, 22, 1220}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 128, 16, 20, 1218}, {2, 128, 16, 20, 1218}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 202;
    string expectTilingData = "549755813890 94489280530 1220 5291399708704 85899345936 1218 0 4294967297 4294967297 4294967297 47828755808304 68719476741 202 ";
    std::vector<size_t> expectWorkspaces = {24346624};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Scenario: bf16 with small batch/channel where nMulC <= coreNum, expect SMALL tiling key 300 and single-block dispatch
TEST_F(ReflectionPad3dGradTilingTest, bf16_small_tiling_small_nc)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{1, 1, 18, 18, 18}, {1, 1, 18, 18, 18}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{1, 1, 16, 16, 16}, {1, 1, 16, 16, 16}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              &compileInfo);
    uint64_t expectTilingKey = 300;
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: bf16 with flat dimensions (height <= 32), expect FLAT tiling key 302
TEST_F(ReflectionPad3dGradTilingTest, bf16_flat_tiling)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad3dGrad",
        {{{{1, 1, 18, 20, 600}, {1, 1, 18, 20, 600}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
        {{{{1, 1, 16, 18, 598}, {1, 1, 16, 18, 598}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 302;
    std::vector<size_t> expectWorkspaces = {16855040};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: bf16 with large dimensions exceeding UB capacity for both SMALL and MID, expect BIG tiling key 303
TEST_F(ReflectionPad3dGradTilingTest, bf16_big_tiling)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad3dGrad",
        {{{{1, 1, 18, 100, 100}, {1, 1, 18, 100, 100}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
        {{{{1, 1, 16, 98, 98}, {1, 1, 16, 98, 98}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 303;
    std::vector<size_t> expectWorkspaces = {16827392};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: float16 with large dimensions exceeding UB, expect BIG tiling key 203
TEST_F(ReflectionPad3dGradTilingTest, float16_big_tiling)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad3dGrad",
        {{{{1, 1, 18, 100, 100}, {1, 1, 18, 100, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
        {{{{1, 1, 16, 98, 98}, {1, 1, 16, 98, 98}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 203;
    std::vector<size_t> expectWorkspaces = {16827392};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: float32 input with int32 padding dtype, expect tiling succeeds with SMALL key 100
TEST_F(ReflectionPad3dGradTilingTest, float32_int32_padding_small)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int32_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{1, 1, 18, 20, 20}, {1, 1, 18, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{1, 1, 16, 18, 18}, {1, 1, 16, 18, 18}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: float32 with height > width triggering Mymax(a > b) branch, expect MID tiling key 101
TEST_F(ReflectionPad3dGradTilingTest, float32_mid_tiling_height_gt_width)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad3dGrad",
        {{{{2, 64, 18, 144, 120}, {2, 64, 18, 144, 120}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
        {{{{2, 64, 16, 142, 118}, {2, 64, 16, 142, 118}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 101;
    std::vector<size_t> expectWorkspaces = {17661952};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Scenario: both depth paddings are zero, nMulC multiplied by depth for core assignment, expect SMALL key 100
TEST_F(ReflectionPad3dGradTilingTest, float32_zero_depth_pad_nc_less_than_core)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{1, 1, 16, 20, 20}, {1, 1, 16, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{1, 1, 16, 18, 18}, {1, 1, 16, 18, 18}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16842752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Note: The ubFactorElement <= 0 check (line 348-349) is unreachable via tiling paths.
// When availableUb is small, uint32_t subtraction in SplitUb underflows instead of yielding 0,
// making ubFactorElement a very large value. This is documented as unreachable code.

// Scenario: input tensor has 4 dimensions instead of required 5, expect failure
TEST_F(ReflectionPad3dGradTilingTest, wrong_input_dim)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 128, 66, 66}, {2, 128, 66, 66}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{2, 128, 64, 64}, {2, 128, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: padding tensor shape has 6 elements instead of required 10 (2*5), expect failure
TEST_F(ReflectionPad3dGradTilingTest, wrong_padding_shape)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{1, 1, 18, 20, 20}, {1, 1, 18, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{1, 1, 16, 18, 18}, {1, 1, 16, 18, 18}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Scenario: output shape inconsistent with input shape minus paddings, expect failure
TEST_F(ReflectionPad3dGradTilingTest, output_shape_mismatch)
{
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{1, 1, 18, 20, 20}, {1, 1, 18, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()}},
                                              {{{{1, 1, 16, 16, 16}, {1, 1, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
