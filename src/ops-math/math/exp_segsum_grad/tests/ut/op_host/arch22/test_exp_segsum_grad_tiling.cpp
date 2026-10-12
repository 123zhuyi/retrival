/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch22/exp_segsum_grad_tiling.h"

using namespace std;
using namespace ge;
using namespace gert;

class ExpSegsumGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExpSegsumGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExpSegsumGradTiling TearDown" << std::endl; }
};

TEST_F(ExpSegsumGradTiling, exp_segsum_grad_tiling_001)
{
    optiling::ExpSegsumGradCompileInfo compileInfo = {1, 10};
    gert::TilingContextPara tilingContextPara("ExpSegsumGrad",
                                              {{{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 1 12 192 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ExpSegsumGradTiling, exp_segsum_grad_tiling_002)
{
    // coreNumPlatform (64) exceeds MAX_CORE_CONT (50): GetNeedCoreNum must clamp it so that
    // batchStart/batchEnd (sized MAX_CORE_CONT) are never written out of bounds.
    optiling::ExpSegsumGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ExpSegsumGrad",
                                              {{{{64, 64, 12, 12}, {64, 64, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{64, 64, 12, 12}, {64, 64, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {{{64, 64, 12, 12}, {64, 64, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // coreNumPlatform=64 is clamped to MAX_CORE_CONT=50: batches=4096 → averageBatches=82 → needCoreNum=50.
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(50));
    ASSERT_TRUE(tilingInfo.tilingData != nullptr);
    // Verify the clamp is what produced needCoreNum=50 and that the batch split loses no data:
    // the last core's batchEnd must cover all batches.
    const int64_t* fields = reinterpret_cast<const int64_t*>(tilingInfo.tilingData.get());
    EXPECT_EQ(fields[0], static_cast<int64_t>(50));   // needCoreNum
    EXPECT_EQ(fields[1], static_cast<int64_t>(4096)); // batches
    const int32_t* batchEnd = reinterpret_cast<const int32_t*>(tilingInfo.tilingData.get() + 32 +
                                                               optiling::MAX_CORE_CONT * sizeof(int32_t));
    EXPECT_EQ(batchEnd[fields[0] - 1], static_cast<int32_t>(fields[1]));
}

TEST_F(ExpSegsumGradTiling, exp_segsum_grad_tiling_003)
{
    // batches == 0: skip GetNeedCoreNum/GetTilingKey and force needCoreNum = 1, since the
    // runtime requires a non-zero block dimension even when there is no output to compute.
    optiling::ExpSegsumGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ExpSegsumGrad",
                                              {{{{0, 2, 12, 12}, {0, 2, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{0, 2, 12, 12}, {0, 2, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {{{0, 2, 12, 12}, {0, 2, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(1));
    EXPECT_EQ(tilingInfo.tilingKey, 1); // SMALL_SIZE_TILING_KEY
}

TEST_F(ExpSegsumGradTiling, exp_segsum_grad_tiling_004)
{
    // tailDimLength == 0: GetTilingKey would divide by zero (calNumAlign == 0), so the
    // zero-size guard must skip it and force needCoreNum = 1.
    optiling::ExpSegsumGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ExpSegsumGrad",
                                              {{{{2, 2, 12, 0}, {2, 2, 12, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 2, 12, 0}, {2, 2, 12, 0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {{{2, 2, 12, 0}, {2, 2, 12, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(1));
    EXPECT_EQ(tilingInfo.tilingKey, 1); // SMALL_SIZE_TILING_KEY
}
