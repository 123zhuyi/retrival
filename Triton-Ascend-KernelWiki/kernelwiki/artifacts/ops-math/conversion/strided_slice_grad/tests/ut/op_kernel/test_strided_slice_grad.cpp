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
 * \file test_strided_slice_grad.cpp
 * \brief kernel UT for StridedSliceGrad operator
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/strided_slice_grad_apt.cpp"

class StridedSliceGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceGradTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "StridedSliceGradTest TearDown" << std::endl;
    }
};

// ============================================================================
// Test Case 1: ALL_CLEAR_EMPTY_TENSOR_INT32 (tiling key 104)
// Empty tensor path: only clears output, no data copy needed
// ============================================================================
TEST_F(StridedSliceGradTest, test_empty_tensor_int32)
{
    constexpr int64_t OUT_ELEMS = 128;

    uint8_t* shape   = (uint8_t*)AscendC::GmAlloc(2 * sizeof(int32_t));
    uint8_t* begin   = (uint8_t*)AscendC::GmAlloc(2 * sizeof(int32_t));
    uint8_t* end     = (uint8_t*)AscendC::GmAlloc(2 * sizeof(int32_t));
    uint8_t* strides = (uint8_t*)AscendC::GmAlloc(2 * sizeof(int32_t));
    uint8_t* dy      = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));  // empty
    uint8_t* output  = (uint8_t*)AscendC::GmAlloc(OUT_ELEMS * sizeof(int32_t));

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling    = (uint8_t*)AscendC::GmAlloc(sizeof(StridedSliceGradTilingData));

    StridedSliceGradTilingData* td = reinterpret_cast<StridedSliceGradTilingData*>(tiling);
    td->usedCoreNumForClear          = 1;
    td->normalCoreProcessNumForClear = OUT_ELEMS;
    td->tailCoreProcessNumForClear   = OUT_ELEMS;
    td->normalCoreProcessNum         = 0;
    td->tailCoreProcessNum           = 0;
    td->tailAxisOuter                = 0;
    td->tailAxisInner                = 0;
    td->tailAxisTail                 = 0;
    td->inputDimNum                  = 2;
    td->usedCoreNum                  = 1;
    td->totalCoreNum                 = 1;
    td->bufferSize                   = OUT_ELEMS * sizeof(int32_t);
    td->splitUbAxisNum               = 0;
    td->bytesForOneData              = sizeof(int32_t);
    td->tilingKey                    = 104;
    td->workspaceSize                = workspaceSize;

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(104);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(strided_slice_grad, numBlocks, shape, begin, end, strides, dy, output, workspace, tiling);

    AscendC::GmFree(shape);
    AscendC::GmFree(begin);
    AscendC::GmFree(end);
    AscendC::GmFree(strides);
    AscendC::GmFree(dy);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
