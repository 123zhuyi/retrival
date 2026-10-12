/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_kernel/arch35/matrix_set_diag_v2_tilingdata.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MatrixSetDiagV2TilingTest : public testing::Test {
protected:
    MatrixSetDiagCompileInfo compileInfo{};

    static void SetUpTestCase() { std::cout << "MatrixSetDiagV2TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatrixSetDiagV2TilingTest TearDown" << std::endl; }
};

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_int16)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_uint8)
{
    std::vector<int32_t> kValues = {1};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_float)
{
    std::vector<int32_t> kValues = {-1, 1};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_bool)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{2, 2, 2}, {2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_should_have_same_type)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_diag_dim_should_less_input_dim)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_input_dim_from_2)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_align)
{
    std::vector<int32_t> kValues = {-31, 0};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiagV2",
        {
            {{{10000, 2, 10000, 32}, {10000, 2, 10000, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10000, 2, 32, 32}, {10000, 2, 32, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{10000, 2, 10000, 32}, {10000, 2, 10000, 32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalCntPerCore, 12500);
    EXPECT_EQ(data->xRowFactor, 256);
    EXPECT_EQ(data->xColFactor, 32);
    EXPECT_EQ(data->input.coreNum, 64);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_not_align)
{
    std::vector<int32_t> kValues = {-30, 0};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiagV2",
        {
            {{{10000, 2, 10000, 31}, {10000, 2, 10000, 31}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{10000, 2, 31, 31}, {10000, 2, 31, 31}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{10000, 2, 10000, 31}, {10000, 2, 10000, 31}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalCntPerCore, 5938);
    EXPECT_EQ(data->xRowFactor, 528);
    EXPECT_EQ(data->xColFactor, 31);
    EXPECT_EQ(data->input.coreNum, 64);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_1)
{
    std::vector<int32_t> kValues = {-8, 23};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{23, 144, 138}, {23, 144, 138}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{23, 32, 138}, {23, 32, 138}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{23, 144, 138}, {23, 144, 138}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_2)
{
    std::vector<int32_t> kValues = {-2, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{657, 3, 8}, {657, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{657, 10, 3}, {657, 10, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{657, 3, 8}, {657, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_3)
{
    std::vector<int32_t> kValues = {-3, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1507, 4, 8}, {1507, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1507, 11, 4}, {1507, 11, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1507, 4, 8}, {1507, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_4)
{
    std::vector<int32_t> kValues = {7, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{83, 192, 3}, {83, 192, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_5)
{
    std::vector<int32_t> kValues = {7, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{83, 192, 3}, {83, 192, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_6)
{
    std::vector<int32_t> kValues = {4, 4};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1252, 3, 4}, {1252, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1252, 3}, {1252, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1252, 3, 4}, {1252, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_optimize_core_num1)
{
    std::vector<int32_t> kValues = {-1, 4};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 21, 121, 100}, {1, 21, 121, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 21, 6, 100}, {1, 21, 6, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 21, 121, 100}, {1, 21, 121, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 42);
    EXPECT_EQ(data->xRowFactor, 81);
    EXPECT_EQ(data->xColFactor, 100);
    EXPECT_EQ(data->totalCntPerCore, 1);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_optimize_core_num2)
{
    std::vector<int32_t> kValues = {-20532, 0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{65538, 2}, {65538, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{20533, 2}, {20533, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{65538, 2}, {65538, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 33);
    EXPECT_EQ(data->xRowFactor, 2048);
    EXPECT_EQ(data->xColFactor, 2);
    EXPECT_EQ(data->totalCntPerCore, 1);
}

// 场景：k0==k1==0 且 dSize<=2、tailAxisDataSize>=32767，触发 Tiling4CutTail 后走 V1 路径
// （Tiling4CutW/CalUbFactor/GetOptimizeTiling 循环并在 sizeTaken<=4096 时 break）。
// 输入 x{10000,4} INT16，diag{4}，k{0}。
// 期望：成功；tilingKey=0x404（V1+切尾轴）；tilingData 为 MatrixSetDiagTilingData：
// coreNum=23, mergeDimSize=1, xRowNum=10000, xColNum=4, diagLen=4, ubPerCore=1,
// ubFactor=1740, ubTotalCount=23, ubPerTail=23, tailAxisDataSize=40000；blockNum=23；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_v1_cutw_optimize_loop)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{10000, 4}, {10000, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{10000, 4}, {10000, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 23);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000100);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MatrixSetDiagTilingData));
    MatrixSetDiagTilingData* data = reinterpret_cast<MatrixSetDiagTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->coreNum, 23);
    EXPECT_EQ(data->mergeDimSize, 1);
    EXPECT_EQ(data->xRowNum, 10000);
    EXPECT_EQ(data->xColNum, 4);
    EXPECT_EQ(data->diagLen, 4);
    EXPECT_EQ(data->ubPerCore, 1);
    EXPECT_EQ(data->ubFactor, 1740);
    EXPECT_EQ(data->ubTotalCount, 23);
    EXPECT_EQ(data->ubPerTail, 23);
    EXPECT_EQ(data->tailAxisDataSize, 40000);
}

// 场景：k0==k1==0 且 dSize<=2、tailAxisDataSize>=32767，V1 路径中
// realCoreNum(52)/coreNum(64)=0.8125>=MIN_USED_CORES_RATIO(0.8)，
// 触发 GetOptimizeTiling 的提前返回分支。输入 x{52,10000,4} INT16，diag{52,4}，k{0}。
// 期望：成功；tilingKey=0x404；tilingData：coreNum=52, mergeDimSize=52, xRowNum=10000, xColNum=4,
// diagLen=4, ubPerCore=1, ubFactor=40000, ubTotalCount=52, ubPerTail=1, tailAxisDataSize=40000；
// blockNum=52；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_v1_cutw_optimize_early_return)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{52, 10000, 4}, {52, 10000, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{52, 4}, {52, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{52, 10000, 4}, {52, 10000, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 52);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000100);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MatrixSetDiagTilingData));
    MatrixSetDiagTilingData* data = reinterpret_cast<MatrixSetDiagTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->coreNum, 52);
    EXPECT_EQ(data->mergeDimSize, 52);
    EXPECT_EQ(data->xRowNum, 10000);
    EXPECT_EQ(data->xColNum, 4);
    EXPECT_EQ(data->diagLen, 4);
    EXPECT_EQ(data->ubPerCore, 1);
    EXPECT_EQ(data->ubFactor, 40000);
    EXPECT_EQ(data->ubTotalCount, 52);
    EXPECT_EQ(data->ubPerTail, 1);
    EXPECT_EQ(data->tailAxisDataSize, 40000);
}

// 场景：k0==k1==0，xColNum*dSize >= bufferSize_，触发 V1 路径 CalUbFactor 的 if 分支
// （ubFactor=validBufSize/dSize）。输入 x{2,16384} INT64，diag{2}，k{0}。
// 期望：成功；tilingKey=0x404；tilingData：coreNum=51, mergeDimSize=1, xRowNum=2, xColNum=16384,
// diagLen=2, ubPerCore=1, ubFactor=643, ubTotalCount=51, ubPerTail=51, tailAxisDataSize=32768；
// blockNum=51；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_v1_cutw_calubfactor_if_branch)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 16384}, {2, 16384}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 16384}, {2, 16384}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 51);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000100);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MatrixSetDiagTilingData));
    MatrixSetDiagTilingData* data = reinterpret_cast<MatrixSetDiagTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->coreNum, 51);
    EXPECT_EQ(data->mergeDimSize, 1);
    EXPECT_EQ(data->xRowNum, 2);
    EXPECT_EQ(data->xColNum, 16384);
    EXPECT_EQ(data->diagLen, 2);
    EXPECT_EQ(data->ubPerCore, 1);
    EXPECT_EQ(data->ubFactor, 643);
    EXPECT_EQ(data->ubTotalCount, 51);
    EXPECT_EQ(data->ubPerTail, 51);
    EXPECT_EQ(data->tailAxisDataSize, 32768);
}

// 场景：ratio<SIMT_RATIO 使 way=SIMT，且 ubSize(16384)<SIMT_DCACHE_SIZE(32768)，
// CalculateValidBufSize 的 SIMT 分支校验失败。输入 x{20,20} FLOAT，diag{20}，k{0}，ubSize=16384。
// 期望：GRAPH_FAILED。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_simt_ub_too_small_fail)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{20, 20}, {20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{20, 20}, {20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 场景：ratio=1.99>=SCATTER_RATIO 使 way=GATHER 且 isVLFullLoad=false（additionTileSize=整块 tail 数据），
// validBufSize < totalTailSize 使 ubFactor=0，Tiling4NoCutTail 回退 Tiling4CutTail。
// 输入 x{100,100} FLOAT，diag{199,100}，k{-99,99}。
// 期望：成功；tilingKey=0x400（回退后切尾轴）；tilingData 为 MSDV2CutTailTilingData：
// input.coreNum=3, mergeDimSize=1, xRowNum=100, xColNum=100, diagNum=199, maxDiagLen=100,
// k0=-99, k1=99, xRowFactor=40, xColFactor=100, totalCntPerCore=1；blockNum=3；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_gather_ubfactor_zero_fallback_cuttail)
{
    std::vector<int32_t> kValues = {-99, 99};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{100, 100}, {100, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{199, 100}, {199, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{100, 100}, {100, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 3);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 3);
    EXPECT_EQ(data->input.mergeDimSize, 1);
    EXPECT_EQ(data->input.xRowNum, 100);
    EXPECT_EQ(data->input.xColNum, 100);
    EXPECT_EQ(data->input.diagNum, 199);
    EXPECT_EQ(data->input.maxDiagLen, 100);
    EXPECT_EQ(data->input.k0, -99);
    EXPECT_EQ(data->input.k1, 99);
    EXPECT_EQ(data->xRowFactor, 40);
    EXPECT_EQ(data->xColFactor, 100);
    EXPECT_EQ(data->totalCntPerCore, 1);
}

// 场景：ratio=0.25 使 way=SCATTER 且 isVLFullLoad=true；NoCutTail 优化路径
// GetOptimizeTilingNoCutTail 首轮 sizeTaken<=MIN_PER_UB_SIZE(1024) 触发 break。
// 输入 x{2,12,12} FLOAT，diag{2,3,12}，k{-1,1}。
// 期望：成功；tilingKey=0x102（SCATTER+VL满载）；tilingData 为 MSDV2NoCutTailTilingData：
// input.coreNum=1, mergeDimSize=2, xRowNum=12, xColNum=12, diagNum=3, maxDiagLen=12,
// k0=-1, k1=1, mergeDimNumPerCore=1, ubFactor=2；blockNum=1；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_nocuttail_optimize_break_min_ub)
{
    std::vector<int32_t> kValues = {-1, 1};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 12, 12}, {2, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3, 12}, {2, 3, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 12, 12}, {2, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 1);
    EXPECT_EQ(data->input.mergeDimSize, 2);
    EXPECT_EQ(data->input.xRowNum, 12);
    EXPECT_EQ(data->input.xColNum, 12);
    EXPECT_EQ(data->input.diagNum, 3);
    EXPECT_EQ(data->input.maxDiagLen, 12);
    EXPECT_EQ(data->input.k0, -1);
    EXPECT_EQ(data->input.k1, 1);
    EXPECT_EQ(data->mergeDimNumPerCore, 1);
    EXPECT_EQ(data->ubFactor, 2);
}

// 场景：ratio=0.005<SIMT_RATIO 使 way=SIMT 且 ubSize 正常，直接走
// CalculateValidBufSize SIMT 分支并 FillNoCutTailTilingData（realCoreNum=64 不进优化）。
// 输入 x{64,200,100} FLOAT，diag{64,100}，k{0}。
// 期望：成功；tilingKey=0x003（SIMT）；tilingData 为 MSDV2NoCutTailTilingData：
// input.coreNum=64, mergeDimSize=64, xRowNum=200, xColNum=100, diagNum=1, maxDiagLen=100,
// k0=0, k1=0, mergeDimNumPerCore=1, ubFactor=1；blockNum=64；workspace=0。
TEST_F(MatrixSetDiagV2TilingTest, test_tiling_simt_direct_nocuttail)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{64, 200, 100}, {64, 200, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 100}, {64, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{64, 200, 100}, {64, 200, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000011);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 64);
    EXPECT_EQ(data->input.mergeDimSize, 64);
    EXPECT_EQ(data->input.xRowNum, 200);
    EXPECT_EQ(data->input.xColNum, 100);
    EXPECT_EQ(data->input.diagNum, 1);
    EXPECT_EQ(data->input.maxDiagLen, 100);
    EXPECT_EQ(data->input.k0, 0);
    EXPECT_EQ(data->input.k1, 0);
    EXPECT_EQ(data->mergeDimNumPerCore, 1);
    EXPECT_EQ(data->ubFactor, 1);
}
