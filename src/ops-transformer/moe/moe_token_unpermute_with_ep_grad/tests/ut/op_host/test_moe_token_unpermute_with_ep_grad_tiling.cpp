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
 * \file test_moe_token_unpermute_with_ep_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/moe_token_unpermute_with_ep_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

class MoeTokenUnpermuteWithEpGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_bf16_001)
{
    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({1,1})},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({-1,-1})},
                                        {"topk_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 29 10 54 1 0 3 0 256 1 64 3 3 16 261888 29 29 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}



TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_bf16_002)
{

    //////

    //dlog_setlevel(0, 0, 0);
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};
    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 256 1 64 3 3 8 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_none_bf16_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"topk_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 64 1 64 984 984 984 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_bf16_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_BF16, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 16 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_fp16_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 256 1 64 3 3 16 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_none_fp16_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"topk_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 64 1 64 984 984 984 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_fp16_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 16 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_fp16_002)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 8 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_fp32_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 128 1 64 3 3 8 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_none_fp32_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"topk_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 56 2 8 504 504 504 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_fp32_001)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 7168 2 1024 1 3 8 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_fp32_error_indices)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {{30, 1}, {30, 1}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 128 1 64 3 3 8 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_split_h_fp32_002)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {};
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {unpermuted_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                              {permuted_tokens_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {probs_shape, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {permuted_tokens_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 7168 2 1024 1 3 16 261888 0 30 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}
