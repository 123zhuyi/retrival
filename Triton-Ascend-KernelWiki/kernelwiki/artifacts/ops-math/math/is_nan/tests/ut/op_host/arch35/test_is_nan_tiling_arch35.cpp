/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/is_nan_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class IsNanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IsNanTilingTest Ascend C SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IsNanTilingTest Ascend C TearDown" << std::endl;
    }
};

TEST_F(IsNanTilingTest, test_tiling_fp16_001)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 93458488360962 4096 2 1 1 4096 4096 21760 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsNanTilingTest, test_tiling_bf16_002)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 93458488360962 4096 2 1 1 4096 4096 21760 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsNanTilingTest, test_tiling_fp32_003)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "8192 46179488366594 4096 2 1 1 4096 4096 10752 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsNanTilingTest, test_tiling_invalid_input_dtype_004)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(IsNanTilingTest, test_tiling_invalid_output_dtype_005)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(IsNanTilingTest, test_tiling_invalid_shape_006)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 6}, {1, 64, 2, 6}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(IsNanTilingTest, test_tiling_empty_tensor_007)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 0, 64}, {1, 64, 0, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 6}, {1, 64, 2, 6}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(IsNanTilingTest, test_tiling_empty_tensor_008)
{
    optiling::IsNanCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "IsNan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 0, 64}, {1, 64, 0, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}