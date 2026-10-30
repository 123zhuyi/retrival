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
 * \file test_tril.cpp
 * \brief kernel UT for Tril operator
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/tril_apt.cpp"

class TrilTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "tril_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "tril_test TearDown" << std::endl;
    }
};

// ============================================================================
// Test Case 1: OUTPUT_ZERO mode (tiling key 4) - int32
// All elements below diagonal are zeroed; uses simple SplitAll path
// ============================================================================
TEST_F(TrilTest, test_output_zero_int32)
{
    constexpr int64_t ELEM_NUM = 128;

    size_t byteSize = ELEM_NUM * sizeof(int32_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TriangulatorTilingData));

    TriangulatorTilingData* tilingData = reinterpret_cast<TriangulatorTilingData*>(tiling);
    tilingData->bufferSize = ELEM_NUM * sizeof(int32_t);
    tilingData->usedCoreNum = 1;
    tilingData->normalCoreProcessNum = ELEM_NUM;
    tilingData->tailCoreProcessNum = ELEM_NUM;

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(tril, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
