/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "math/cumulative_logsumexp/op_kernel/arch35/cumulative_logsumexp_tiling_key.h"

class CumulativeLogsumexpTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CumulativeLogsumexpTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CumulativeLogsumexpTiling TearDown" << std::endl; }
};

TEST_F(CumulativeLogsumexpTiling, fp32_rank3_axis1_inclusive_forward)
{
    int32_t axisValue = 1;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, 8);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT);
    std::string expectTilingData = "24 2 3 4 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 3);
}

TEST_F(CumulativeLogsumexpTiling, fp16_negative_axis_exclusive_reverse)
{
    int64_t axisValue = -1;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo, 64);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16);
    std::string expectTilingData = "105 15 7 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CumulativeLogsumexpTiling, int16_axis_scalar_supported)
{
    int16_t axisValue = 0;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16);
    std::string expectTilingData = "128 1 8 16 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CumulativeLogsumexpTiling, unsupported_x_dtype_failed)
{
    int32_t axisValue = 0;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 4}, {2, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(CumulativeLogsumexpTiling, axis_out_of_range_failed)
{
    int64_t axisValue = 2;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}
