/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sinc_tiling.cpp
 * \brief Sinc算子Tiling策略测试
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/sinc_tiling_arch35.h"

using namespace std;
using namespace ge;

class SincTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SincTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SincTiling TearDown" << std::endl; }
};

// ========== 正常场景测试 - 不同数据类型 ==========

TEST_F(SincTilingTest, test_tiling_fp16_001)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 3;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_fp32_002)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_bf16_003)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{shape, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 5;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// ========== 正常场景测试 - 不同形状维度 ==========

TEST_F(SincTilingTest, test_tiling_1dim_fp32_004)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{256}, {256}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_2dim_fp32_005)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{16, 256}, {16, 256}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_4dim_fp32_006)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// ========== 异常场景测试 - 数据类型不匹配 ==========

TEST_F(SincTilingTest, test_tiling_failed_dtype_input_output_diff_007)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_failed_dtype_fp32_fp16_008)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// ========== 异常场景测试 - 形状不匹配 ==========

TEST_F(SincTilingTest, test_tiling_failed_shape_input_output_diff_009)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape inShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape outShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara("Sinc", {{inShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_failed_shape_dim_diff_010)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape inShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape outShape = {{1, 64, 2}, {1, 64, 2}};
    gert::TilingContextPara tilingContextPara("Sinc", {{inShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// ========== 异常场景测试 - 空Tensor ==========

TEST_F(SincTilingTest, test_tiling_failed_empty_tensor_011)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 0, 2, 64}, {1, 0, 2, 64}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// ========== 异常场景测试 - 不支持的数据类型 ==========

TEST_F(SincTilingTest, test_tiling_failed_unsupport_input_double_012)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_DOUBLE, ge::FORMAT_ND}},
                                              {{shape, ge::DT_DOUBLE, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_failed_unsupport_input_int32_013)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{shape, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_failed_unsupport_output_int32_014)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// ========== 边界场景测试 - 大形状 ==========

TEST_F(SincTilingTest, test_tiling_large_shape_fp32_015)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1024, 1024}, {1024, 1024}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 7;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_tiling_large_shape_fp16_016)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{2048, 2048}, {2048, 2048}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 3;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(SincTilingTest, test_ez0020_output_dtype_int32)
{
    optiling::SincCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::TilingContextPara tilingContextPara("Sinc", {{shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{shape, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}
