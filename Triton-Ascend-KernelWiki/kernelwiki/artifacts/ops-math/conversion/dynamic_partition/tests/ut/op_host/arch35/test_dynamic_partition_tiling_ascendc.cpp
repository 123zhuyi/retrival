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
 * \file test_dynamic_partition_tiling_ascendc.cpp
 * \brief dynamic_partition tiling ut test
 */

#include "../../../../op_host/arch35/dynamic_partition_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DynamicPartitionTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicPartitionTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicPartitionTilingTest TearDown" << std::endl; }
};

TEST_F(DynamicPartitionTilingTest, DynamicPartitionTiling_001)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {64, 245760, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{2, 3, 20, 1000}, {2, 3, 20, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3, 20}, {2, 3, 20}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{120, 1000}, {120, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 64 120000 1880 1560 1880 245760 120 1 1000 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[128,1128] float16，Ascend 950 (20核, 256KB UB)
// totalElements=144384, elePerBlock=16
// mainSize=CeilAlign(CeilDiv(144384,20),16)=CeilAlign(7220,16)=7232
// usedCoreCnt=CeilDiv(144384,7232)=20, tailSize=144384-7232*19=6976
// ubFactor=(availUB/2/blockSize)*elePerBlock=(262048/2/32)*16=65536, min(65536,7232)=7232
// dim0=128, dimNumExtFirst=1, outDimsExtFirst[0]=1128
TEST_F(DynamicPartitionTilingTest, NumPartOne_2D_F16)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{128, 1128}, {128, 1128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128, 1128}, {128, 1128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1}); // num_partitions=1
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 20 144384 7232 6976 7232 262144 128 1 1128 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[64,512] float32
// totalElements=32768, elePerBlock=8
// mainSize=CeilAlign(CeilDiv(32768,20),8)=CeilAlign(1639,8)=1640
// usedCoreCnt=CeilDiv(32768,1640)=20, tailSize=32768-1640*19=1608
// ubFactor=(availUB/2/32)*8=(262048/2/32)*8=32768, min(32768,1640)=1640
// dim0=64, dimNumExtFirst=1, outDimsExtFirst[0]=512
TEST_F(DynamicPartitionTilingTest, NumPartOne_2D_F32)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1});
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 20 32768 1640 1608 1640 262144 64 1 512 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[32,64,16] float16，partitions[32,64]
// After reshape: xShape=[2048,16], totalElements=32768, elePerBlock=16
// mainSize=CeilAlign(CeilDiv(32768,20),16)=CeilAlign(1639,16)=1648
// usedCoreCnt=CeilDiv(32768,1648)=20, tailSize=32768-1648*19=1456
// ubFactor=min((262048/2/32)*16, 1648)=min(65536,1648)=1648
// dim0=2048, dimNumExtFirst=1, outDimsExtFirst[0]=16
TEST_F(DynamicPartitionTilingTest, NumPartOne_3D_F16)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{32, 64, 16}, {32, 64, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 64}, {32, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 64, 16}, {32, 64, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1});
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 20 32768 1648 1456 1648 262144 2048 1 16 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[4096,64] float16（大 H 轴）
// totalElements=262144, elePerBlock=16
// mainSize=CeilAlign(CeilDiv(262144,20),16)=CeilAlign(13108,16)=13120
// usedCoreCnt=CeilDiv(262144,13120)=20, tailSize=262144-13120*19=12864
// ubFactor=min((262048/2/32)*16, 13120)=min(65536,13120)=13120
// dim0=4096, dimNumExtFirst=1, outDimsExtFirst[0]=64
TEST_F(DynamicPartitionTilingTest, NumPartOne_LargeH)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{4096, 64}, {4096, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 64}, {4096, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1});
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 20 262144 13120 12864 13120 262144 4096 1 64 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[1, 1024] float16（小 shape，optBurstElems=512 下限生效）
// CeilAlign(CeilDiv(1024,20),16)=64 < optBurstElems=512 → mainSize=512
// usedCoreCnt=CeilDiv(1024,512)=2, tailSize=512, ubFactor=512
// dim0=1, dimNumExtFirst=1, outDimsExtFirst[0]=1024
TEST_F(DynamicPartitionTilingTest, NumPartOne_SmallShape_F16)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1});
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 2 1024 512 512 512 262144 1 1 1024 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// num_partitions=1 用例：x[1,8] int32（单核场景，totalElements=8 < optBurstElems=256）
// mainSize=max(CeilAlign(CeilDiv(8,20),8),256)=max(8,256)=256
// usedCoreCnt=CeilDiv(8,256)=1, tailSize=8, ubFactor=min(32768,256)=256
// dim0=1, dimNumExtFirst=1, outDimsExtFirst[0]=8
TEST_F(DynamicPartitionTilingTest, NumPartOne_SmallH)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {20, 262144, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                  {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo, {1});
    uint64_t expectTilingKey = 50008;
    string expectTilingData = "50008 1 8 256 8 256 262144 1 1 8 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
