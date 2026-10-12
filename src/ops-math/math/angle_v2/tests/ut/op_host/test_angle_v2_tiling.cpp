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
#include "../../../op_host/angle_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class AngleV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AngleV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AngleV2Tiling TearDown" << std::endl;
  }
};

struct AngleV2CompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};


TEST_F(AngleV2Tiling, test_angle_v2_complex64) {

    AngleV2CompileInfo compileInfo = {64, 262144, true};
    gert::TilingContextPara tilingContextPara(
        "AngleV2",
        {
            {{{16, 16}, {16, 16}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "256 0 4 64 64 8 256 3840 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}