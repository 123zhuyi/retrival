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
#include "math/atan/op_host/arch35/atan_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;

class AtanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AtanTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AtanTilingTest TearDown" << std::endl;
    }
};

TEST_F(AtanTilingTest, atan_test_tiling_fp16_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 203;
    string expectTilingData = "8192 4 5760 2048 4 1 1 2048 2048 5760 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(AtanTilingTest, atan_test_tiling_bf16_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 205;
    string expectTilingData = "8192 4 5760 2048 4 1 1 2048 2048 5760 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(AtanTilingTest, atan_test_tiling_fp32_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 207;
    string expectTilingData = "8192 8 8704 1024 8 1 1 1024 1024 8704 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(AtanTilingTest, atan_test_tiling_invalid_input_dtype)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, false, expectTilingData, expectWorkspaces);
}

TEST_F(AtanTilingTest, atan_test_tiling_invalid_output_dtype)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, false, expectTilingData, expectWorkspaces);
}

TEST_F(AtanTilingTest, atan_test_tiling_invalid_shape)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 6}, {1, 64, 2, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(
        tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, false, expectTilingData, expectWorkspaces);
}
