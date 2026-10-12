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
#include "../../../op_host/matrix_diag_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace MatrixDiagAsc;

class MatrixDiagTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "MatrixDiagTiling SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "MatrixDiagTiling TearDown " << endl;
    }
};

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_100_int8) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{34000, 32}, {34000, 32}}, ge::DT_INT8, ge::FORMAT_ND},},
                                              {{{{34000, 32, 32}, {34000, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 34000 32 62 24 549 9 8 37 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_100) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{320, 64, 40}, {320, 64, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{320, 64, 40, 40}, {320, 64, 40, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 20480 40 38 36 539 9 8 27 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_102) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{30000, 2}, {30000, 2}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              {{{{30000, 2, 2}, {30000, 2, 2}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "30000 2 253952 64 32 938 937 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_103) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{8126464, 1}, {8126464, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{8126464, 1, 1}, {8126464, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "8126464 253952 64 64 126976 126976 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_101) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{1200,64}, {1200,64}}, ge::DT_UINT32, ge::FORMAT_ND},},
                                              {{{{1200,64,64}, {1200,64,64}}, ge::DT_UINT32, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "64 0 64 2 2 600 10 9 24 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_101_int8) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{600,256}, {600,256}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              {{{{600,256,256}, {600,256,256}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "64 0 256 1 1 600 10 9 24 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_104) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{520,900}, {520,900}}, ge::DT_UINT16, ge::FORMAT_ND},},
                                              {{{{520,900,900}, {520,900,900}}, ge::DT_UINT16, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData = "64 520 900 128 4 8 520 0 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tilingkey_104_int8) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{638,700}, {638,700}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              {{{{638,700,700}, {638,700,700}}, ge::DT_UINT8, ge::FORMAT_ND},},
                                              &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData = "64 638 700 256 188 3 90 89 46 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tiling_failed_empty_tensor) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{1, 64, 0, 64}, {1, 64, 0, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 64, 0, 64, 64}, {1, 64, 0, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tiling_failed_scalar) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagTiling, ascend910D1_test_tiling_failed_zero_last_dim) {
    optiling::MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{64, 0}, {64, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{64, 0, 0}, {64, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}