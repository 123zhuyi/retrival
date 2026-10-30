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

#include "../../../op_host/add_lora_tiling.h"

class AddLoraTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLoraTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLoraTiling TearDown" << std::endl;
    }
};

struct Tiling4AddLoraCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(AddLoraTiling, ascend910B1_test_tiling__001)
{
    Tiling4AddLoraCompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "AddLora",
        {
            {{{1, 4096}, {1, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 4096, 16}, {2, 1, 4096, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 1, 16, 16}, {2, 1, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4096}, {1, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("layer_idx", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(0.01)),
         gert::TilingContextPara::OpAttr("y_offset", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("y_slice_size", Ops::Math::AnyValue::CreateFrom<int64_t>(4096))},
         &compileInfo);
    uint64_t expectTilingKey = 100001;
    string expectTilingData = "4294967360 68719476737 68719480832 2 1008981770 4096 4294967328 4096 ";
    std::vector<size_t> expectWorkspaces = {17957600};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

