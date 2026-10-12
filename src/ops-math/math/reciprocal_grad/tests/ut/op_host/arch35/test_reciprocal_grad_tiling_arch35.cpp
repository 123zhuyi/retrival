/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_kernel/arch35/reciprocal_grad_tiling_data.h"
#include "../../../../op_host/arch35/reciprocal_grad_tiling_arch35.h"

using namespace std;
using namespace ge;
using optiling::ReciprocalGradCompileInfo;

class ReciprocalGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReciprocalGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReciprocalGradTiling TearDown" << std::endl; }
};

// ─── L0: 各 dtype 基本用例 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L0: 1D / 4D shape ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_1d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_4d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_1d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_4d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L0: 不同 core/UB 配置 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_different_cores)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 16;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 16, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_small_ub)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 32768;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 8, 32768, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L1: 大 shape ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_large)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_large)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L1: 反例测试 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_shape_mismatch_y_dy)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_shape_mismatch_y_z)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_dtype_mismatch_y_dy)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_unsupported_dtype)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
