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
#include <vector>
#include <tuple>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/sinkhorn_tiling.h"


using namespace std;
using namespace ge;

class SinkhornTiling : public testing::Test {
protected:
 static void SetUpTestCase() {
   std::cout << "SinkhornTiling SetUp" << std::endl;
 }

 static void TearDownTestCase() {
   std::cout << "SinkhornTiling TearDown" << std::endl;
 }
};

TEST_F(SinkhornTiling, sinkhorn_tiling_parse) {
    optiling::SinkhornCompileInfo compileInfo = {40, 16 * 1024 * 1024, 196608};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn",
        {
            {{{48, 2}, {48, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10000, 160}, {10000, 160}}, ge::DT_FLOAT, ge::FORMAT_ND},

        },
        {
            {{{48, 2}, {48, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10000, 160}, {10000, 160}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001)),
            gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.00001)),
        }
        ,&compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 48 96 1 48 96 0 0 0 0 0 0 2084 4168 48 2 8 953267991 ";
    std::vector<size_t> expectWorkspaces = {16777496};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}