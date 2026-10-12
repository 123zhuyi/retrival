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
 * \file test_moe_fused_topk_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/moe_fused_topk_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
class MoeFusedTopkTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFusedTopkTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFusedTopkTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeFusedTopkTiling, test_tiling_bf16)
{
    optiling::MoeFusedTopkCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
		                                  {
                                        {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
					                            },
                                      {
                                        {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"group_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"group_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                        {"top_n", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                        {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"activate_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                        {"is_norm", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                        {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                        {"enable_expert_mapping", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                      },
                                      &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "549755814016 34359738496 8589934596 8 4575657221408423937 171798691856 12884907452 8 4398046511104 1024 512 1099511627936 4294967360 34359738376 274877906960 34359738372 17179869186 8589934598 4294967360 34359738496 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 40 * 512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}