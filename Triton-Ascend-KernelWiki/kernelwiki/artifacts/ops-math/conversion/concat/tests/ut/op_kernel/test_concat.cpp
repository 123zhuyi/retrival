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
 * \file test_concat.cpp
 * \brief kernel UT for Concat operator
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"

struct ConcatTilingDataNoArrArrays {
    int64_t preLoadDim1[2];
    uint64_t strideList[32];
    uint64_t concatDimList[32];
};

struct ConcatTilingDataNoArray {
    int16_t ubSplitDim1;
    int16_t dim;
    int16_t tensorNum;
    int16_t dtypeSize;
    int16_t isNonContiguous;
    int32_t ubFactorDim0;
    int32_t ubFactorDim1;
    int32_t tailUbFactorDim0;
    int32_t tailUbFactorDim1;
    int32_t bufferSize;
    int32_t dataPtrOffset;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t uoDim0;
    int64_t uoDim1;
    int64_t catDim1;
    int64_t sameShapeTensorDim1;
    int16_t isFP4Type;
    ConcatTilingDataNoArrArrays arrays;
};

struct ConcatTilingDataForSimtArrays {
    int32_t tensorColsOffset[128];
};

struct ConcatTilingDataForSimt {
    int32_t tensorNumPerCore;
    int32_t tensorNum;
    int32_t catDim0;
    int32_t catDim1;
    ConcatTilingDataForSimtArrays arrays;
};

#include <algorithm>
using std::min;

#include "../../../op_kernel/concat_apt.cpp"

class ConcatTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "concat_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "concat_test TearDown" << std::endl;
    }
};

// ============================================================================
// PURE_COPY_SPLIT_DIM1 mode (tiling key 20002) - float32
// 3 inputs of shape (2, 4), concat on dim1 => output (2, 12)
// ============================================================================
TEST_F(ConcatTest, test_pure_copy_split_dim1_float32)
{
    constexpr int64_t BATCH = 2;
    constexpr int64_t DIM1_PER_TENSOR = 4;
    constexpr int64_t TENSOR_NUM = 3;
    constexpr int64_t OUT_DIM1 = DIM1_PER_TENSOR * TENSOR_NUM;

    size_t inputByteSize = BATCH * DIM1_PER_TENSOR * TENSOR_NUM * sizeof(float);
    size_t outputByteSize = BATCH * OUT_DIM1 * sizeof(float);
    size_t dimByteSize = sizeof(int32_t);

    uint8_t* dimBuf = (uint8_t*)AscendC::GmAlloc(dimByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ConcatTilingData));

    ConcatTilingData* tilingData = reinterpret_cast<ConcatTilingData*>(tiling);
    tilingData->ubSplitDim1 = 1;
    tilingData->dim = 1;
    tilingData->tensorNum = TENSOR_NUM;
    tilingData->dtypeSize = sizeof(float);
    tilingData->isNonContiguous = 0;
    tilingData->ubFactorDim0 = static_cast<int32_t>(BATCH);
    tilingData->ubFactorDim1 = static_cast<int32_t>(OUT_DIM1);
    tilingData->tailUbFactorDim0 = static_cast<int32_t>(BATCH);
    tilingData->tailUbFactorDim1 = static_cast<int32_t>(OUT_DIM1);
    tilingData->bufferSize = static_cast<int32_t>(BATCH * OUT_DIM1 * sizeof(float));
    tilingData->dataPtrOffset = 0;
    tilingData->blockFactor = 1;
    tilingData->tailBlockFactor = 1;
    tilingData->uoDim0 = BATCH;
    tilingData->uoDim1 = OUT_DIM1;
    tilingData->catDim1 = OUT_DIM1;
    tilingData->sameShapeTensorDim1 = DIM1_PER_TENSOR;
    tilingData->isFP4Type = 0;
    for (int i = 0; i < 2; i++) {
        tilingData->arrays.preLoadDim1[i] = DIM1_PER_TENSOR;
    }
    for (int i = 0; i < 32; i++) {
        tilingData->arrays.strideList[i] = 0;
        tilingData->arrays.concatDimList[i] = (i < TENSOR_NUM) ? DIM1_PER_TENSOR : 0;
    }

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(20002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(concat, numBlocks, dimBuf, x, y, workspace, tiling);

    AscendC::GmFree(dimBuf);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
