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
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct MoeInitRoutingV2CompileInfo {};

class MoeInitRoutingV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingV2Tiling TearDown" << std::endl;
  }
};


gert::TilingContextPara RunMoeInitRoutingV2Case(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                      int64_t dropPadMode, int64_t countFlag, bool tokenFlag, int64_t quantMode,
                                      int64_t dqFlag, ge::DataType optionalDt, int64_t optionalDtypePosi)
{
    MoeInitRoutingV2CompileInfo compileInfo;
    // 根据 optionalDtypePosi 确定数据类型
    ge::DataType dtScale = optionalDtypePosi == 0 ? optionalDt : ge::DT_FLOAT;
    ge::DataType dtOffset = optionalDtypePosi == 1 ? optionalDt : ge::DT_FLOAT;
    ge::DataType dtDynamic = optionalDtypePosi == 2 ? optionalDt : ge::DT_FLOAT;
    // 计算输入形状
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    // x: [N, H]
    inputs.emplace_back(gert::StorageShape({N, H}, {N, H}), ge::DT_FLOAT16, ge::FORMAT_ND);
    // expert_idx: [N, K]
    inputs.emplace_back(gert::StorageShape({N, K}, {N, K}), ge::DT_INT32, ge::FORMAT_ND);

    // scale 形状根据 quantMode 变化
    if (quantMode == 0) {
        // 静态量化: [1]
        inputs.emplace_back(gert::StorageShape({1}, {1}), dtScale, ge::FORMAT_ND);
    } else {
        if (dqFlag == 0) {
            // 动态量化: [E, H]
            inputs.emplace_back(gert::StorageShape({E, H}, {E, H}), dtScale, ge::FORMAT_ND);
        } else {
            // 动态量化: [1, H]
            inputs.emplace_back(gert::StorageShape({1, H}, {1, H}), dtScale, ge::FORMAT_ND);
        }
    }

    // 计算输出形状
    std::vector<gert::TilingContextPara::TensorDescription> outputs;

    // expanded_x (形状根据 dropPadMode 变化)
    if (dropPadMode == 0) {
        int64_t first_dim = N * K;
        if (activeNum > 0 && activeNum < first_dim) {
            first_dim = activeNum;
        }
        // [first_dim, H]
        outputs.emplace_back(gert::StorageShape({first_dim, H}, {first_dim, H}), ge::DT_FLOAT16, ge::FORMAT_ND);
    } else {
        // [E, C, H]
        outputs.emplace_back(gert::StorageShape({E, C, H}, {E, C, H}), ge::DT_FLOAT16, ge::FORMAT_ND);
    }
    // expanded_row_idx [N*K]
    outputs.emplace_back(gert::StorageShape({N * K}, {N * K}), ge::DT_INT32, ge::FORMAT_ND);
    // expert_tokens_count_or_cumsum [E]
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);

    // expert_tokens_before_capacity [E]
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);
    // dynamic_quant_scale (形状根据 dropPadMode 变化)

    // 设置属性
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(E)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
        {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)}};

    // 创建 TilingContextPara
    return gert::TilingContextPara("MoeInitRoutingV2", inputs, outputs, attrs, &compileInfo);
}
// 单核+drop  
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_01)
{
  auto tilingContextPara = RunMoeInitRoutingV2Case(8, 30, 6, 0, 6, 8, 1, 0, true, 0, 0, ge::DT_FLOAT, 0);
  uint64_t expectTilingKey = 10011;
  string expectTilingData = "64 8 30 6 6 8 1 0 1 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
  std::vector<size_t> expectWorkspaces = {16779264};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+非drop  
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_perf)
{
  auto tilingContextPara = RunMoeInitRoutingV2Case(8, 30, 6, 0, 6, 8, 0, 0, false, 0, 0, ge::DT_FLOAT, 0);
  uint64_t expectTilingKey = 20000;
  string expectTilingData = "64 8 30 6 6 8 0 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
  std::vector<size_t> expectWorkspaces = {16779264};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核++dropless  11000
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_one_core_dropless) {
  auto tilingContextPara = RunMoeInitRoutingV2Case(
    /*N=*/80, /*H=*/3000, /*K=*/60, 
    /*activeNum=*/0, /*C=*/6, /*E=*/8,
    /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
    /*quantMode=*/1, /*dqFlag=*/0, 
    /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0
  );
  
  uint64_t expectTilingKey = 20000;
  string expectTilingData = "64 80 3000 60 6 8 0 1 0 1 4800 1 4800 4800 4800 1 4800 4800 8160 0 2040 64 0 75 75 75 75 75 75 0 0 0 0 0 64 0 75 75 75 75 75 75 1 1 3000 3000 1 64 4800 75 75 75 75 75 75 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {16931328};
  
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+drop  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_drop) {
  auto tilingContextPara = RunMoeInitRoutingV2Case(
    /*N=*/320, /*H=*/3000, /*K=*/56, 
    /*activeNum=*/0, /*C=*/200, /*E=*/32,
    /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
    /*quantMode=*/0, /*dqFlag=*/0, 
    /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0
  );
  
  uint64_t expectTilingKey = 10012;
  string expectTilingData = "64 320 3000 56 200 32 1 0 1 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {17351168};
  
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+dropless  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_dropless) {
  auto tilingContextPara = RunMoeInitRoutingV2Case(
    /*N=*/320, /*H=*/3000, /*K=*/56, 
    /*activeNum=*/0, /*C=*/200, /*E=*/32,
    /*dropPadMode=*/0, /*countFlag=*/0, /*tokenFlag=*/true,
    /*quantMode=*/0, /*dqFlag=*/0, 
    /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0
  );
  
  uint64_t expectTilingKey = 10002;
  string expectTilingData = "64 320 3000 56 200 32 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {17351168};
  
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_expert_id) {
  auto tilingContextPara = RunMoeInitRoutingV2Case(
    /*N=*/320, /*H=*/3000, /*K=*/56, 
    /*activeNum=*/0, /*C=*/200, /*E=*/32,
    /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
    /*quantMode=*/0, /*dqFlag=*/0, 
    /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/0
  );
  tilingContextPara.inputTensorDesc_[1].shape_ = gert::StorageShape({2}, {2});

  // 对于错误用例，不需要检查 tilingKey 和 tilingData
  ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_token) {
  auto tilingContextPara = RunMoeInitRoutingV2Case(
    /*N=*/320, /*H=*/3000, /*K=*/56, 
    /*activeNum=*/0, /*C=*/200, /*E=*/32,
    /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
    /*quantMode=*/0, /*dqFlag=*/0, 
    /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/0
  );
  tilingContextPara.inputTensorDesc_[0].shape_ = gert::StorageShape({2}, {2});

  // 对于错误用例，不需要检查 tilingKey 和 tilingData
  ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

