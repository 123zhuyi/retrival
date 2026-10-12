/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_dynamic_stitch_tiling.cpp
 * \brief dynamic_stitch tiling ut test
 */

#include "../../../../op_host/arch35/dynamic_stitch_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DynamicStitchTilingTest : public testing::Test {
   protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicStitchTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicStitchTilingTest TearDown" << std::endl;
    }
};

TEST_F(DynamicStitchTilingTest, DynamicStitchTiling_tiling_test_1)
{
    optiling::DynamicStitchCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("DynamicStitch",
                                              {
                                                {{{24, 4, 128}, {24, 4, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{24, 4, 128, 62}, {24, 4, 128, 62}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{100, 62}, {100, 62}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 100008;
    std::vector<size_t> expectWorkspaces = {16826768};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}

TEST_F(DynamicStitchTilingTest, DynamicStitchTiling_tiling_test_multi_dim_slice)
{
    optiling::DynamicStitchCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("DynamicStitch",
                                              {
                                                {{{24, 4, 128}, {24, 4, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{24, 4, 128, 62, 31}, {24, 4, 128, 62, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{100, 62, 31}, {100, 62, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 200008;
    std::vector<size_t> expectWorkspaces = {16826768};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}

TEST_F(DynamicStitchTilingTest, DynamicStitchTiling_tiling_test_for_zero_slice)
{
    optiling::DynamicStitchCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("DynamicStitch",
                                              {
                                                {{{24, 4, 128}, {24, 4, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{24, 4, 128}, {24, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 100004;
    std::vector<size_t> expectWorkspaces = {16826768};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}

TEST_F(DynamicStitchTilingTest, DynamicStitchTiling_tiling_test_for_large_slice)
{
    optiling::DynamicStitchCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("DynamicStitch",
                                              {
                                                {{{24, 4, 128}, {24, 4, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{24, 4, 128, 131072}, {24, 4, 128, 131072}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{100, 131072}, {100, 131072}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 200008;
    std::vector<size_t> expectWorkspaces = {16826768};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}