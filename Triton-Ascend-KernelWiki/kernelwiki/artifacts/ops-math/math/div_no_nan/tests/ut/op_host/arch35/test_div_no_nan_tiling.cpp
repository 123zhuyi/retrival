/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../../../../op_host/arch35/div_no_nan_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DivNoNanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DivNoNanTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DivNoNanTilingTest TearDown" << std::endl;
    }
};

TEST_F(DivNoNanTilingTest, test_tiling_int32) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivNoNanTilingTest, test_tiling_fp16) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivNoNanTilingTest, test_tiling_bf16) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivNoNanTilingTest, test_tiling_fp32) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivNoNanTilingTest, test_tiling_uint8) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivNoNanTilingTest, test_tiling_int8) {
    optiling::DivNoNanCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("DivNoNan",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}