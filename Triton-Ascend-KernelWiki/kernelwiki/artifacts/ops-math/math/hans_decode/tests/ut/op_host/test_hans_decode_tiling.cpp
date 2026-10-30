/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/hans_decode_tiling.h"

using namespace std;
using namespace ge;

class HansDecodeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansDecode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansDecode TearDown" << std::endl;
  }
};

struct HansDecodeCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(HansDecodeTiling, ascend910B1_test_tiling_001)
{
    HansDecodeCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "HansDecode",
	      {
            {{{49152}, {49152}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{65536}, {65536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
	      {
          gert::TilingContextPara::OpAttr("reshuff", Ops::Math::AnyValue::CreateFrom<bool>(false))
        },
         &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "196608 65536 262144 1048576 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

