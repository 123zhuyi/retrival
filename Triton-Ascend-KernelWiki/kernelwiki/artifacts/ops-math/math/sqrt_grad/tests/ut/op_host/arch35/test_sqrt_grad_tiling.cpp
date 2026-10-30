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
 * \file test_sqrt_grad_tiling.cpp
 * \brief
 */
#include "../../../../op_host/arch35/sqrt_grad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SqrtGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SqrtGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SqrtGradTiling TearDown" << std::endl;
    }
};

TEST_F(SqrtGradTiling, test_tiling_failed_dtype_input_diff_004)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_dtype_input_output_diff_005)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_shape_input_diff_006)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_shape_input_output_diff_007)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_empty_tensor_008)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_unsupport_input_009)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SqrtGradTiling, test_tiling_failed_unsupport_output_010)
{
    optiling::SqrtGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "SqrtGrad",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}