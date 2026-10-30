/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEGG FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "diag_part_tiling.h"
#include "../../../op_kernel/diag_part_tiling_data.h"
#include "../../../op_kernel/diag_part_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class DiagPartTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DiagPartTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "DiagPartTiling TearDown " << endl; }
};

TEST_F(DiagPartTiling, ascend910b_test_tiling_FLOAT16_001)
{
    optiling::DiagPartCompileInfo compileInfo = {40, 196608, false};

    // TensorDescription is nested inside TilingContextPara
    using TD = gert::TilingContextPara::TensorDescription;
    TD inputDesc(gert::StorageShape{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    TD outputDesc(gert::StorageShape{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    std::vector<TD> inputDescs{inputDesc};
    std::vector<TD> outputDescs{outputDesc};

    gert::TilingContextPara tilingContextPara("DiagPart", inputDescs, outputDescs, &compileInfo, 40, 196608, 4096);
    // GET_TPL_TILING_KEY(schMode=0, dtype=FLOAT16=1): schMode index=0, dtype index=1 in [0,1,3] -> key = 0 | (1<<1) = 2
    uint64_t expectTilingKey = 2;
    // sideLength=4, dtype=1(float16), realCoreNum=1, numPerCore=16, tailNum=4, blockSize=0
    string expectTilingData = "4 1 1 16 4 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DiagPartTiling, ascend910b_test_tiling_FLOAT32_001)
{
    optiling::DiagPartCompileInfo compileInfo = {40, 196608, false};

    using TD = gert::TilingContextPara::TensorDescription;
    TD inputDesc(gert::StorageShape{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    TD outputDesc(gert::StorageShape{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND);

    std::vector<TD> inputDescs{inputDesc};
    std::vector<TD> outputDescs{outputDesc};

    gert::TilingContextPara tilingContextPara("DiagPart", inputDescs, outputDescs, &compileInfo, 40, 196608, 4096);
    // GET_TPL_TILING_KEY(schMode=0, dtype=FLOAT=0): schMode index=0, dtype index=0 in [0,1,3] -> key = 0
    uint64_t expectTilingKey = 0;
    // sideLength=4, dtype=0(float32), realCoreNum=1, numPerCore=8, tailNum=4, blockSize=0
    string expectTilingData = "4 0 1 8 4 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DiagPartTiling, ascend910b_test_tiling_INT32_001)
{
    optiling::DiagPartCompileInfo compileInfo = {40, 196608, false};

    using TD = gert::TilingContextPara::TensorDescription;
    TD inputDesc(gert::StorageShape{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND);
    TD outputDesc(gert::StorageShape{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND);

    std::vector<TD> inputDescs{inputDesc};
    std::vector<TD> outputDescs{outputDesc};

    gert::TilingContextPara tilingContextPara("DiagPart", inputDescs, outputDescs, &compileInfo, 40, 196608, 4096);
    // GET_TPL_TILING_KEY(schMode=0, dtype=INT32=3): schMode index=0, dtype index=2 in [0,1,3] -> key = 0 | (2<<1) = 4
    uint64_t expectTilingKey = 4;
    // sideLength=4, dtype=3(int32), realCoreNum=1, numPerCore=8, tailNum=4, blockSize=0
    string expectTilingData = "4 3 1 8 4 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
