/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../op_host/arch35/log_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class LogTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LogTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LogTilingTest TearDown" << std::endl;
    }
};

TEST_F(LogTilingTest, test_tiling_failed_dtype_input_output_diff_005)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Log",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(LogTilingTest, test_tiling_failed_shape_input_output_diff_007)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Log",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(LogTilingTest, test_tiling_failed_empty_tensor_008)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Log",
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(LogTilingTest, test_tiling_failed_unsupport_input_009)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Log",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(LogTilingTest, test_tiling_failed_unsupport_output_010)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Log",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}