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
#include "../../../../op_host/arch35/ceil_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class CeilTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CeilTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CeilTilingTest TearDown" << std::endl;
  }
};

TEST_F(CeilTilingTest, test_tiling_fp16_001) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_bf16_002) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 5;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}


TEST_F(CeilTilingTest, test_tiling_fp32_003) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 7;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_dtype_input_output_diff_005) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_shape_input_output_diff_007) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64},  {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_empty_tensor_008) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_unsupport_input_009) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_unsupport_output_010) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}