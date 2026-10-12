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
 * \file test_tensor_equal.cpp
 * \brief kernel UT for TensorEqual operator
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/tensor_equal_apt.cpp"

class TensorEqualTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "tensor_equal_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "tensor_equal_test TearDown" << std::endl;
    }
};

// ============================================================================
// NORMAL mode (tiling key 121) - float32, 1D
// input_x: (64,), input_y: (64,) => output_z: scalar bool
// ============================================================================
TEST_F(TensorEqualTest, test_normal_float32)
{
    constexpr int64_t ELEM_NUM = 64;

    size_t inputByteSize = ELEM_NUM * sizeof(float);
    size_t outputByteSize = sizeof(int8_t);

    uint8_t* input_x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* input_y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output_z = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TensorEqualTilingData));

    TensorEqualTilingData* tilingData = reinterpret_cast<TensorEqualTilingData*>(tiling);
    tilingData->inputShapeSize = ELEM_NUM;
    tilingData->inputDtypeSize = sizeof(float);
    tilingData->usedCoreNum = 1;
    tilingData->ubSize = ELEM_NUM * sizeof(float);
    tilingData->ubFactor = ELEM_NUM;
    tilingData->perCoreLoopTimes = 1;
    tilingData->tailCoreLoopTimes = 1;
    tilingData->perCoreTailFactor = ELEM_NUM;
    tilingData->tailCoreTailFactor = ELEM_NUM;
    tilingData->tilingKey = 121;

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(tensor_equal, numBlocks, input_x, input_y, output_z, workspace, tiling);

    AscendC::GmFree(input_x);
    AscendC::GmFree(input_y);
    AscendC::GmFree(output_z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
