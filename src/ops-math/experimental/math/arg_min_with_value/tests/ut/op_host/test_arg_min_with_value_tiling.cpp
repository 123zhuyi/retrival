/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_arg_min_with_value_tiling.cpp
 * \brief ArgMinWithValue tiling unit tests.
 */

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ArgMinWithValueTiling : public testing::Test {};

// ArgMinWithValueCompileInfo is empty (host reads platform info directly per-call), so any non-null
// pointer satisfies the framework's CompileInfo plumbing.
struct ArgMinWithValueCompileInfo {};

TEST_F(ArgMinWithValueTiling, arg_min_with_value_test_tiling_last_mode)
{
    ArgMinWithValueCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ArgMinWithValue",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 1; // GET_TPL_TILING_KEY(ARG_TPL_SCH_LAST)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}
