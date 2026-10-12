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
 * \file test_sim_thread_exponential_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/sim_thread_exponential_tiling_arch35.h"

class SimThreadExponentialTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SimThreadExponentialTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SimThreadExponentialTiling TearDown" << std::endl;
    }
};

TEST_F(SimThreadExponentialTiling, sim_thread_exponential_tiling_fp32)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential", {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "64 300000 5 4 0 0 1065353216 0 0 0 1 300000 0 624 159744 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SimThreadExponentialTiling, sim_thread_exponential_tiling_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential", {{{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);

    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "64 300000 5 4 0 0 1065353216 0 0 0 1 300000 0 624 159744 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SimThreadExponentialTiling, sim_thread_exponential_tiling_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential", {{{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);

    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "64 300000 5 4 0 0 1065353216 0 0 0 1 300000 0 624 159744 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
