
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
 * \file test_chunk_cat.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/chunk_cat_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void chunk_cat(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class chunk_cat_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "chunk_cat_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "chunk_cat_test TearDown\n" << endl;
    }
};

TEST_F(chunk_cat_test, test_case_fp16_001)
{
    size_t xSize = 4 * 16 * sizeof(half);
    size_t ySize = 4 * 16 * sizeof(half);
    size_t tilingDataSize = sizeof(ChunkCatTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    ChunkCatTilingData* tilingData = reinterpret_cast<ChunkCatTilingData*>(tiling);
    tilingData->isAllAlign = false;
    tilingData->isHalfAlign = true;
    tilingData->inputNum = 1;
    tilingData->inUbSize = 87381;
    tilingData->outUbSize = 174762;
    tilingData->blockRowNum = 1;
    tilingData->blockColNum = 1;
    tilingData->ubRowFactor = 4;
    tilingData->ubColFactor = 16;
    tilingData->dim = 0;
    tilingData->numChunk = 4;
    tilingData->outputRow = 4;
    tilingData->outputCol = 8;
    tilingData->blockRowFactor = 4;
    tilingData->blockColFactor = 16;
    tilingData->tailBlockRowFactor = 4;
    tilingData->tailBlockColFactor = 8;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(chunk_cat, numBlocks, x, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
