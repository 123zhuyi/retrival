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
 * \file test_sim_thread_exponential_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch32/sim_thread_exponential_tiling_arch32.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SimThreadExponentialTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SimThreadExponentialTilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SimThreadExponentialTilingTest  TearDown" << std::endl;
    }
};

TEST_F(SimThreadExponentialTilingTest, SimThreadExponentialTilingData_test_FP32_300000_success_case0)
{
    optiling::Tiling4SimThreadExponentialCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential",
        {
            {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "5 1 42949673023 2680059592708 300000 10720238373312 5360320512 4 0 4575657221408423936 1065353216 ";
    std::vector<size_t> expectWorkspaces = {16778000};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SimThreadExponentialTilingTest, SimThreadExponentialTilingData_test_FP16_300000_success_case0)
{
    optiling::Tiling4SimThreadExponentialCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential",
        {
            {{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "5 1 42949673023 2680059592708 300000 10720238373312 5360320512 4 0 4575657221408423936 1065353216 ";
    std::vector<size_t> expectWorkspaces = {16778000};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SimThreadExponentialTilingTest, SimThreadExponentialTilingData_test_bf16_300000_success_case0)
{
    optiling::Tiling4SimThreadExponentialCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "SimThreadExponential",
        {
            {{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(300000)),
         gert::TilingContextPara::OpAttr("lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
         gert::TilingContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "5 1 42949673023 2680059592708 300000 10720238373312 5360320512 4 0 4575657221408423936 1065353216 ";
    std::vector<size_t> expectWorkspaces = {16778000};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}