/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "trilu_tiling.h"
#include "../../../../op_kernel/arch35/trilu_tiling_data.h"
#include "../../../../op_kernel/arch35/trilu_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class TriluTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "TriluTilingTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "TriluTilingTest TearDown" << endl; }
};

TEST_F(TriluTilingTest, ascend950_test_tiling_FLOAT_001)
{
    optiling::TriluCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Trilu",
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({4, 4}, {4, 4}), ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({4, 4}, {4, 4}), ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        },
        &compileInfo);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(TRILU_TPL_SCH_MODE_0);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TriluTilingTest, ascend950_test_tiling_INT32_002)
{
    optiling::TriluCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Trilu",
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({3, 5}, {3, 5}), ge::DT_INT32, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({3, 5}, {3, 5}), ge::DT_INT32, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(TRILU_TPL_SCH_MODE_2);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
