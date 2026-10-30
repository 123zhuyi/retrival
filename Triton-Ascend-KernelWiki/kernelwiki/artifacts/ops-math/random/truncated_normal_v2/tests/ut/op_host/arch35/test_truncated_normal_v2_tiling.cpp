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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/truncated_normal_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

class TruncatedNormalV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TruncatedNormalV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TruncatedNormalV2 TearDown" << std::endl;
  }
};

TEST_F(TruncatedNormalV2Tiling, truncated_normal_v2_tiling_910D_float_001)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int32_t> shape_value = {32, 512};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "TruncatedNormalV2", {{shape_shape, ge::DT_INT32, ge::FORMAT_ND, true, shape_value.data()}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed),
         gert::TilingContextPara::OpAttr("seed2", seed2),
         gert::TilingContextPara::OpAttr("dtype", dtype)},
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData =
        "64 16384 10 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
