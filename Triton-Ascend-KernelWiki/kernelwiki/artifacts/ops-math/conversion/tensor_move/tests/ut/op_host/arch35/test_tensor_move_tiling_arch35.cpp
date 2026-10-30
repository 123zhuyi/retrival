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
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/tensor_move_tiling_arch35.h"

using namespace std;

class TensorMoveTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorMoveTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorMoveTilingTest TearDown" << std::endl;
  }
};

TEST_F(TensorMoveTilingTest, tensormove_tiling1)
{
    optiling::TensorMoveCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "TensorMove",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "64 1 1 1 512 256 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TensorMoveTilingTest, tensormove_invalid_dtype)
{
    optiling::TensorMoveCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "TensorMove",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "64 1 1 1 512 256 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
