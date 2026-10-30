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
#include "../../../../op_host/arch35/neg_tiling_arch35.h"
#include "../../../../op_kernel/arch35/neg_tiling_struct.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;
using namespace ge;

class NegTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NegTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NegTiling TearDown" << std::endl;
  }
};

TEST_F(NegTiling, neg_test_tiling_float16_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "64 1 32768 512 1 1 1 512 64 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_float_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "64 1 16384 512 1 1 1 512 64 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_INT32_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 11;
    string expectTilingData = "64 1 16384 512 1 1 1 512 64 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_int8_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "64 1 65536 512 1 1 1 512 64 65536 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_int64_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 13;
    string expectTilingData = "64 1 8192 512 1 1 1 512 64 8192 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_invalid_shape)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "64 1 32768 512 1 1 1 512 64 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_invalid_dtype)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "64 1 32768 512 1 1 1 512 64 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}