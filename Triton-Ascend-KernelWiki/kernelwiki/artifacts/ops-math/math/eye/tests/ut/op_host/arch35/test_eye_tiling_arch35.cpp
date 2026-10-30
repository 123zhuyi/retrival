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
#include "../../../../op_host/arch35/eye_tiling_arch35.h"

using namespace std;
using namespace ge;

class EyeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EyeTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EyeTiling TearDown" << std::endl;
  }
};

TEST_F(EyeTiling, eye_test_tiling_001)
{
    optiling::EyeCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Eye",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            gert::TilingContextPara::OpAttr("num_rows", Ops::Math::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("num_columns", Ops::Math::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("batch_shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})),
            gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(1))
        },
        &compileInfo);
    uint64_t expectTilingKey = 1001;
    string expectTilingData = "1 1 1 1 8 8 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}                