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
 * \file test_acosh_grad_tiling.cpp
 * \brief AcoshGrad tiling UT
 */

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

#include "../../../op_kernel/acosh_grad_tiling_data.h"
#include "../../../op_kernel/acosh_grad_tiling_key.h"

using namespace std;
using namespace ge;

class AcoshGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AcoshGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AcoshGradTiling TearDown" << std::endl; }
};

TEST_F(AcoshGradTiling, acosh_grad_tiling_fp32_small)
{
    struct AcoshGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("AcoshGrad",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(expectTilingKey));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto& td = *reinterpret_cast<const AcoshGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td.totalLength, 64);
    EXPECT_EQ(td.tileLength, 64);
    EXPECT_EQ(td.lastTileLength, 64);
}

TEST_F(AcoshGradTiling, acosh_grad_tiling_fp16)
{
    struct AcoshGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("AcoshGrad",
                                              {
                                                  {{{65}, {65}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{65}, {65}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{65}, {65}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_0);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(expectTilingKey));
    const auto& td = *reinterpret_cast<const AcoshGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td.totalLength, 65);
    EXPECT_EQ(td.tileLength, 65);
}

TEST_F(AcoshGradTiling, acosh_grad_tiling_bf16)
{
    struct AcoshGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("AcoshGrad",
                                              {
                                                  {{{129}, {129}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{129}, {129}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{129}, {129}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_2);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(expectTilingKey));
    const auto& td = *reinterpret_cast<const AcoshGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td.totalLength, 129);
}

TEST_F(AcoshGradTiling, acosh_grad_tiling_fp32_large_multicore)
{
    struct AcoshGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("AcoshGrad",
                                              {
                                                  {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(expectTilingKey));
    EXPECT_GT(tilingInfo.blockNum, 1);
    const auto& td = *reinterpret_cast<const AcoshGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td.totalLength, 16384);
    EXPECT_EQ(td.coreNum, tilingInfo.blockNum);
    EXPECT_EQ(td.blockLength, 2048);
    EXPECT_EQ(td.tailBlockLength, 2048);
    EXPECT_EQ(td.tileLength % 64, 0);
}

TEST_F(AcoshGradTiling, acosh_grad_tiling_fp16_large)
{
    struct AcoshGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("AcoshGrad",
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_0);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(expectTilingKey));
    EXPECT_GT(tilingInfo.blockNum, 1);
    const auto& td = *reinterpret_cast<const AcoshGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td.totalLength, 1023 * 2047);
    EXPECT_EQ(td.coreNum, tilingInfo.blockNum);
}
