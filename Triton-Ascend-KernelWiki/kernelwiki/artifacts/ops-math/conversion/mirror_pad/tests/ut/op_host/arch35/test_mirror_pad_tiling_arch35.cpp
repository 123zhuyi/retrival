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

using namespace std;
using namespace ge;

class MirrorPadTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MirrorPadTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MirrorPadTiling TearDown" << std::endl; }
};

TEST_F(MirrorPadTiling, mirror_pad_tiling_test_001)
{
    struct MirrorPadCompileInfo {};
    MirrorPadCompileInfo compileInfo = {};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{1, 1}, {1, 1}};
    int pad_value[2] = {3, 4};

    gert::TilingContextPara tilingContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("REFLECT"))},
        &compileInfo);
    uint64_t expectTilingKey = 21000;
    string
        expectTilingData = "1 0 0 24 3 4 0 0 0 0 0 24 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
