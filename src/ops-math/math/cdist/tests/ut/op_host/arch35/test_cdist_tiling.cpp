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
#include "../../../../op_host/arch35/cdist_tiling_arch35.h"
#include "../../../../op_kernel/cdist_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class CdistTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "CdistTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "CdistTiling TearDown " << endl; }
};

static optiling::CdistCompileInfo MakeCompileInfo()
{
    optiling::CdistCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;
    return compileInfo;
}

static void RunCdistTiling(const std::vector<gert::TilingContextPara::TensorDescription>& inputs,
                           const std::vector<gert::TilingContextPara::TensorDescription>& outputs,
                           ge::graphStatus expectResult = ge::GRAPH_SUCCESS)
{
    optiling::CdistCompileInfo compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara(
        "Cdist", inputs, outputs, {gert::TilingContextPara::OpAttr("p", Ops::Math::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(ok, expectResult == ge::GRAPH_SUCCESS) << "Cdist tiling unexpected result";
}

TEST_F(CdistTiling, tiling_simt_small_m_fp16)
{
    RunCdistTiling({{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{2, 4, 6}, {2, 4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_simt_small_m_fp32)
{
    RunCdistTiling({{{{4, 16, 8}, {4, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                    {{{4, 32, 8}, {4, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                   {{{{4, 16, 32}, {4, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_large_m_fp16)
{
    RunCdistTiling({{{{4, 4, 300}, {4, 4, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{4, 8, 300}, {4, 8, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{4, 4, 8}, {4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_large_m_fp32)
{
    RunCdistTiling({{{{4, 4, 300}, {4, 4, 300}}, ge::DT_FLOAT, ge::FORMAT_ND},
                    {{{4, 8, 300}, {4, 8, 300}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                   {{{{4, 4, 8}, {4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_batch_4d_fp16)
{
    RunCdistTiling({{{{2, 3, 4, 300}, {2, 3, 4, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{2, 3, 8, 300}, {2, 3, 8, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{2, 3, 4, 8}, {2, 3, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_simt_large_batch_fp16)
{
    RunCdistTiling({{{{8, 8, 16, 200}, {8, 8, 16, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{8, 8, 16, 200}, {8, 8, 16, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{8, 8, 16, 16}, {8, 8, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_p_attr_non_default)
{
    optiling::CdistCompileInfo compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara(
        "Cdist",
        {{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 4, 6}, {2, 4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("p", Ops::Math::AnyValue::CreateFrom<float>(1.0f))}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(CdistTiling, tiling_p_attr_negative_failed)
{
    optiling::CdistCompileInfo compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara(
        "Cdist",
        {{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 4, 6}, {2, 4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("p", Ops::Math::AnyValue::CreateFrom<float>(-1.0f))}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(ok);
}

TEST_F(CdistTiling, tiling_dim_less_than_2_failed)
{
    RunCdistTiling({{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_m_not_equal_failed)
{
    RunCdistTiling(
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{6, 10}, {6, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 6}, {4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_x2_dimnum_mismatch_failed)
{
    RunCdistTiling(
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 6}, {4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_y_dimnum_mismatch_failed)
{
    RunCdistTiling({{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{4, 6}, {4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_y_last_two_dims_failed)
{
    RunCdistTiling(
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_batch_mismatch_failed)
{
    RunCdistTiling({{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{3, 6, 8}, {3, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{2, 4, 6}, {2, 4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, ge::GRAPH_FAILED);
}

TEST_F(CdistTiling, tiling_normal_block_tiling_large_p)
{
    RunCdistTiling({{{{1, 100, 300}, {1, 100, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{1, 200, 300}, {1, 200, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{1, 100, 200}, {1, 100, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_block_tiling_large_r)
{
    RunCdistTiling({{{{1, 4, 300}, {1, 4, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{1, 300, 300}, {1, 300, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{1, 4, 300}, {1, 4, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_block_tiling_large_b)
{
    RunCdistTiling({{{{100, 4, 300}, {100, 4, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{100, 8, 300}, {100, 8, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{100, 4, 8}, {100, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_ub_cut_m_fp16)
{
    RunCdistTiling({{{{4, 500, 300}, {4, 500, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                    {{{4, 500, 300}, {4, 500, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                   {{{{4, 500, 500}, {4, 500, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
}

TEST_F(CdistTiling, tiling_normal_ub_cut_m_fp32)
{
    RunCdistTiling({{{{4, 500, 300}, {4, 500, 300}}, ge::DT_FLOAT, ge::FORMAT_ND},
                    {{{4, 500, 300}, {4, 500, 300}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                   {{{{4, 500, 500}, {4, 500, 500}}, ge::DT_FLOAT, ge::FORMAT_ND}});
}
