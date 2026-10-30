
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
 * \file test_sin_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/sin_tiling_arch35.h"

using namespace std;
using namespace ge;

class SinTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SinTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SinTiling TearDown" << std::endl;
  }
};

TEST_F(SinTilingTest, test_tiling_fp16_001) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }},
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 3;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_bf16_002) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_BF16, ge::FORMAT_ND }},
        {{ shape, ge::DT_BF16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 5;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_fp32_003) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_failed_dtype_input_output_diff_005) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        {{ shape, ge::DT_BF16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 0 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_failed_shape_input_output_diff_007) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape inShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape outShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ inShape, ge::DT_FLOAT, ge::FORMAT_ND }},
        {{ outShape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 0 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_failed_empty_tensor_008) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 0, 2, 64}, {1, 0, 2, 64}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 0 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_failed_unsupport_input_009) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_DOUBLE, ge::FORMAT_ND }},
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 0 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SinTilingTest, test_tiling_failed_unsupport_output_010) {
    optiling::SinCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara(
        "Sin",
        {{ shape, ge::DT_FLOAT, ge::FORMAT_ND }},
        {{ shape, ge::DT_DOUBLE, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 0 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}