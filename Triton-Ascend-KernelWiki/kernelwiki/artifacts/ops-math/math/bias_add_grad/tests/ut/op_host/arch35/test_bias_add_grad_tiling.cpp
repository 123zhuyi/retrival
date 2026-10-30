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
#include "../../../../op_host/arch35/bias_add_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;
class BiasAddGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BiasAddGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BiasAddGradTiling TearDown" << std::endl;
    }
};

TEST_F(BiasAddGradTiling, BiasAddGrad_tiling1)
{
    optiling::BiasAddGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "BiasAddGrad",
        {
            {{{1, 1, 4, 2}, {1, 1, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 4, 2}, {1, 1, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    uint64_t expectTilingKey = 5161;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BiasAddGradTiling, BiasAddGrad_tiling2)
{
    optiling::BiasAddGradCompileInfo compileInfo = {64, 262144, 262144};
    gert::TilingContextPara tilingContextPara(
        "BiasAddGrad",
        {
            {{{1999, 1999, 4, 2}, {1999, 1999, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1999, 1999, 4, 2}, {1999, 1999, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    uint64_t expectTilingKey = 141375;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BiasAddGradTiling, BiasAddGrad_tiling3)
{
    optiling::BiasAddGradCompileInfo compileInfo = {64, 262144, 262144, 262144};
    gert::TilingContextPara tilingContextPara(
        "BiasAddGrad",
        {
            {{{1999, 1999, 4, 1}, {1999, 1999, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1999, 1999, 4, 1}, {1999, 1999, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    uint64_t expectTilingKey = 6167;
    std::vector<size_t> expectWorkspaces = {16793600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BiasAddGradTiling, BiasAddGrad_tiling4)
{
    optiling::BiasAddGradCompileInfo compileInfo = {64, 262144, 262144};
    gert::TilingContextPara tilingContextPara(
        "BiasAddGrad",
        {
            {{{2, 4, 128, 128}, {2, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 128, 128}, {2, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    uint64_t expectTilingKey = 11839;
    std::vector<size_t> expectWorkspaces = {16793600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}