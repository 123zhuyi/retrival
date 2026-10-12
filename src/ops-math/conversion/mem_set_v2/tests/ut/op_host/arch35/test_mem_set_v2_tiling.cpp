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
 * \file test_mem_set_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/mem_set_v2_tiling_arch35.h"

using namespace std;

class MemSetV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MemSetV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MemSetV2Tiling TearDown" << std::endl;
    }
};
struct MemSetV2CompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

TEST_F(MemSetV2Tiling, test_tiling_float_case)
{
    MemSetV2CompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MemSetV2",
        {
            {{{1500, 512, 1}, {1500, 512, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1500, 1, 128}, {1500, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1500, 512, 128}, {1500, 512, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1500, 512, 128}, {1500, 512, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 10002;
    string expectTilingData = "64 262144 1 0 0 0 0 0 94742683582464 47 0 57344 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
