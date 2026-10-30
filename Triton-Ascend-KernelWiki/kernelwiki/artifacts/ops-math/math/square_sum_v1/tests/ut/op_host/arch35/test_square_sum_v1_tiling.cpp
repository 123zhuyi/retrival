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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/reduce/reduce_tiling.h"

using namespace std;

class SquareSumV1Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SquareSumV1Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SquareSumV1Tiling TearDown" << std::endl;
    }
};

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_0)
{
    Ops::Base::ReduceOpCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SquareSumV1",
        {
            {{{100, 48}, {100, 48}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48}, {48}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 5143;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_1)
{
    Ops::Base::ReduceOpCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SquareSumV1",
        {
            {{{16, 16, 16, 16}, {16, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2, 3})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 6167;
    std::vector<size_t> expectWorkspaces = {16793600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_2)
{
    Ops::Base::ReduceOpCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SquareSumV1",
        {
            {{{16, 16, 16, 16}, {16, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2, 3})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 6167;
    std::vector<size_t> expectWorkspaces = {16793600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_3)
{
    Ops::Base::ReduceOpCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SquareSumV1",
        {
            {{{16, 16, 16, 16}, {16, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 11817;
    std::vector<size_t> expectWorkspaces = {16842752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
