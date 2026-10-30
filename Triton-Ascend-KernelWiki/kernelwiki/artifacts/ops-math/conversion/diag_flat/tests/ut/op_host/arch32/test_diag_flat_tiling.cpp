/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../../diag_v2/op_host/arch32/diag_v2_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class diagFlatTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "diagFlat Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "diagFlat Tiling TearDown" << std::endl;
    }
};

struct DiagFlatCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(diagFlatTiling, test_diag_flat_tiling_fp16_fp16) {
    DiagFlatCompileInfo compileInfo = {48, 196608, false};
    
    gert::TilingContextPara tilingContextPara("DiagFlat",
                                              {{{{8,}, {8,}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{8,}, {8,}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "0 0 0 0 0 0 0 101 0 8 48 1 8 0 163840 16384 0 16778752 ";
    std::vector<size_t> expectWorkspaces = {16778752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}