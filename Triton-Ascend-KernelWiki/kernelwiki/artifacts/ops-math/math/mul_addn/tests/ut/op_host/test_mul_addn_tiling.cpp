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
 * \file test_mul_addn.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/mul_addn_tiling.h"

using namespace std;

class MulAddnTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MulAddnTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MulAddnTiling TearDown" << std::endl;
    }
};
struct MulAddnCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(MulAddnTiling, test_tiling_float16_case)
{
    MulAddnCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MulAddn",
        {
            {{{1500, 512, 1}, {1500, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1500, 1, 128}, {1500, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1500, 512, 128}, {1500, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("N", Ops::Math::AnyValue::CreateFrom<int64_t>(6))}, &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "6 1500 512 128 128 24 63 12 171 3 170 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
