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
#include "../../../../op_host/arch35/bias_add_tiling_arch35.h"

using namespace std;
using namespace ge;

class BiasAddTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BiasAddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BiasAddTiling TearDown" << std::endl;
    }
};

TEST_F(BiasAddTiling, BiasAdd_test_tiling_001)
{
    optiling::BiasAddCompileInfo compileInfo = {{}, 0, 0, true};
    int32_t bias = 0;
    gert::TilingContextPara tilingContextPara(
        "BiasAdd",
        {
            {{{3, 1024, 32, 32}, {3, 1024, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, &bias},
        },
        {
            {{{3, 1024, 32, 32}, {3, 1024, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW")),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "12884901888 68719476737 16 64 3 3 192 16384 64 3 1024 1024 0 0 0 0 0 1048576 1024 1 0 0 0 0 0 0 0 0 0 3 1024 "
        "1024 0 0 0 0 0 1 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1048576 1024 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}