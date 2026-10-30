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
 * \file test_abs_grad_tiling_arch35.cpp
 * \brief abs_grad算子tiling单元测试
 */

#include "../../../../op_host/arch35/abs_grad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AbsGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AbsGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AbsGradTilingTest TearDown" << std::endl; }
};

TEST_F(AbsGradTilingTest, test_tiling_fp16_001)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 93458488360964 2048 4 1 1 2048 2048 21760 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp32_002)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 46729244180488 1024 8 1 1 1024 1024 10880 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_bf16_003)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 93458488360964 2048 4 1 1 2048 2048 21760 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_dtype_mismatch_y_dy_004)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_dtype_mismatch_output_005)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_shape_mismatch_006)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 32, 2, 64}, {1, 32, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_unsupported_int32_007)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_unsupported_double_008)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_empty_tensor_009)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp16_1d_shape_010)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 93458488360964 2048 4 1 1 2048 2048 21760 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp32_2d_shape_011)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 46729244180488 1024 8 1 1 1024 1024 10880 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp32_3d_shape_012)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{16, 16, 32}, {16, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16, 32}, {16, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16, 32}, {16, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 46729244180488 1024 8 1 1 1024 1024 10880 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_bf16_4d_shape_013)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 93458488360964 2048 4 1 1 2048 2048 21760 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_unsupported_complex64_014)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_failed_unsupported_bool_015)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp16_small_shape_016)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "16 93458488360961 512 1 1 1 512 16 21760 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsGradTilingTest, test_tiling_fp32_large_shape_017)
{
    optiling::AbsGradCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("AbsGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "1048576 46729244180544 16384 64 2 2 5504 5504 10880 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
