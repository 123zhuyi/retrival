
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
 * \file test_cast_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/cast_tiling.h"

using namespace std;
using namespace ge;

class CastTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CastTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CastTiling TearDown" << std::endl;
  }
};

TEST_F(CastTilingTest, test_tiling_int32_cast_int4_fail) {
    optiling::CastCompileInfo compileInfo = {64, 262144};
    gert::StorageShape shape = {{1, 64, 2, 63}, {1, 64, 2, 63}};
    gert::TilingContextPara tilingContextPara(
        "Cast",
        {{ shape, ge::DT_INT32, ge::FORMAT_ND }},
        {{ shape, ge::DT_INT4, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}
