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
#include "../../../op_host/transform_bias_rescale_qkv_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

class TransformBiasRescaleQkvTilingData : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransformBiasRescaleQkvTilingData SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransformBiasRescaleQkvTilingData TearDown" << std::endl;
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

TEST_F(TransformBiasRescaleQkvTilingData, TransformBiasRescaleQkvTilingData_01)
{
    gert::StorageShape qkv_shape = {{3, 4, 9}, {3, 4, 9}};
    gert::StorageShape qkv_bias_shape = {{9}, {9}};
    gert::StorageShape q_shape = {{3, 3, 4, 3}, {3, 3, 4, 3}};
    gert::StorageShape k_shape = {{3, 3, 4, 3}, {3, 3, 4, 3}};
    gert::StorageShape v_shape = {{3, 3, 4, 3}, {3, 3, 4, 3}};

    optiling::TransformBiasRescaleQkvCompileInfo compileInfo = {};

    auto num_heads = Ops::Math::AnyValue::CreateFrom<int64_t>(3);

    gert::TilingContextPara tilingContextPara(
        "TransformBiasRescaleQkv", 
        {{qkv_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {qkv_bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{q_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {k_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {v_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("num_heads", num_heads)},
        &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "108 36 3 4 3 3 1 12288 ",  expectWorkspaces);
}