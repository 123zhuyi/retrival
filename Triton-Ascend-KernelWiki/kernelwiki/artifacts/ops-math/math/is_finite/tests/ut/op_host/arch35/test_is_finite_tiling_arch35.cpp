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
#include "../../../../op_host/arch35/is_finite_tiling_arch35.h"
#include "../../../../op_kernel/is_finite_struct.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace IsFiniteNs;

class IsFiniteTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "IsFiniteTiling SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "IsFiniteTiling TearDown " << endl;
    }
};

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_fp16_001) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "8192 24189255811074 4096 2 1 1 4096 4096 5632 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_bf16_002) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},},
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 10;
    string expectTilingData = "8192 24189255811074 4096 2 1 1 4096 4096 5632 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_fp32_003) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 14;
    string expectTilingData = "8192 24189255811074 4096 2 1 1 4096 4096 5632 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_failed_empty_tensor_004) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_failed_unsupported_input_type_005) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},},
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_failed_unsupported_output_type_006) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(IsFiniteTiling, ascend910D1_test_tiling_failed_shape_input_output_diff_007) {
    optiling::IsFiniteCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("IsFinite",
                                              {{{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 64, 3, 64}, {1, 64, 3, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara);
}
