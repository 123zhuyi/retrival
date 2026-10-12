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
#include "../../../op_host/hans_encode_tiling.h"

using namespace std;
using namespace ge;

class HansEncodeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansEncode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansEncode TearDown" << std::endl;
  }
};

struct HansEncodeCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(HansEncodeTiling, ascend910B1_test_tiling__001)
{
    HansEncodeCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "HansEncode",
        {
            {{{65536}, {65536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{49152}, {49152}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {gert::TilingContextPara::OpAttr("statistic", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("reshuff", Ops::Math::AnyValue::CreateFrom<bool>(false))},
         &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "2 512 512 32512 32512 65536 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
