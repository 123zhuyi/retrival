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
 * \file test_fill_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/fill_tiling_arch35.h"
#include "../../../../op_kernel/fill_struct.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

class FillTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FillTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FillTilingTest TearDown" << std::endl;
    }
};

TEST_F(FillTilingTest, fill_test_0) {
    optiling::FillCompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara("Fill",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {{{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},},
                                              &compile_info);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "1 70368744177665 512 1 1 1 512 1 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}