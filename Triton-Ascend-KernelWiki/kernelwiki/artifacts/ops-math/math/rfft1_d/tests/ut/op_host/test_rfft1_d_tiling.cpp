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

#include "../../../op_host/rfft1_d_tiling.h"
using namespace std;
using namespace ge;

class Rfft1DTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Rfft1DTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Rfft1DTiling TearDown" << std::endl;
    }
};

struct Rfft1DCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(Rfft1DTiling, ascend910B1_test_tiling__001)
{
    Rfft1DCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "Rfft1D",
        {
            {{{16, 64, 1024}, {16, 64, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 64, 513, 2}, {16, 64, 513, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("n", Ops::Math::AnyValue::CreateFrom<int64_t>(1024)),
         gert::TilingContextPara::OpAttr("norm", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),},
         &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4294968320 274877908992 4294967328 274877906945 4398046513152 137438953504 2203318288640 68719477762 66383014526976 1593192348709248 4296451072 4402342535168 4613555085123584 70368744177664 17408 4398046511104 0 4096 ";
    std::vector<size_t> expectWorkspaces = {2164260864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
