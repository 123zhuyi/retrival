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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_with_ep_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenPermuteWithEpGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteWithEpGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteWithEpGradTiling TearDown" << std::endl;
  }
};

struct MoeTokenPermuteWithEpGradCompileInfo {
};

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_bf16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({1, 49152});
    // input
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    // output
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEpGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {permuted_token_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
                                                {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {permuted_probs_output_d_shape, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {input_token_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
                                                {input_probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 5; // tilngkey
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2 2340 4680 14040 1 1 1 1 1 0 4680 2 2340 1 1 4 1 5 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16777376}; // workspace
}


TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_fp16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({1, 49152});
    // input
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    // output
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEpGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {permuted_token_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {permuted_probs_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {input_token_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {input_probs_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 5; // tilngkey
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2 2340 4680 14040 1 1 1 1 1 0 4680 2 2340 1 1 4 1 5 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16777376}; // workspace
}


TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_fp32)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({1, 49152});
    // input
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    // output
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEpGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {permuted_token_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {permuted_probs_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {input_token_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {input_probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 5; // tilngkey
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2 2340 4680 14040 1 1 1 1 1 0 4680 2 2340 1 1 4 1 5 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16777376}; // workspace
}