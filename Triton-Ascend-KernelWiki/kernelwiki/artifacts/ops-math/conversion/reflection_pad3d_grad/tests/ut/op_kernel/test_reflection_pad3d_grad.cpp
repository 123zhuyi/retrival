/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reflection_pad3d_grad.cpp
 * \brief
 */

#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/reflection_pad3d_grad_tiling.h"

constexpr uint8_t DIM3_NUM = 3;
constexpr uint8_t KERNEL_DIM_IDX = 2;
constexpr uint8_t PAD_DOUBLE_DIM = 2;
constexpr uint8_t C0_B16 = 16;
constexpr uint8_t BF16_SIZE = 2;
constexpr uint8_t FLOAT16_SIZE = 2;
constexpr uint8_t FLOAT_SIZE = 4;
static std::vector<int> KEY_MAP = {4, 2, 2};

extern "C" __global__ __aicore__ void reflection_pad3d_grad(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class reflection_pad3d_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "reflection_pad3d_grad_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "reflection_pad3d_grad_test TearDown\n" << std::endl;
    }
};

size_t VectorReduceMul(const std::vector<uint32_t>& vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<size_t>());
}

void CallSimpleKernel(
    const std::vector<uint32_t>& inputShape, const std::vector<uint32_t>& padding, uint32_t tiling_key = 0)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    const uint32_t numBlocks = 1;
    std::vector<uint32_t> outputShape(inputShape);
    outputShape[2] -= (padding[4] + padding[5]);
    outputShape[3] -= (padding[6] + padding[7]);
    outputShape[4] -= (padding[8] + padding[9]);
    int dtype = tiling_key / 100 - 1;
    size_t inputByteSize = VectorReduceMul(inputShape) * KEY_MAP[dtype];
    size_t outputByteSize = VectorReduceMul(outputShape) * KEY_MAP[dtype];
    size_t tilingDataSize = sizeof(ReflectionPad3dGradTilingData);
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* padvalues = (uint8_t*)AscendC::GmAlloc(10 * 4);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReflectionPad3dGradTilingData));
    memset(workspace, 0, 16 * 1024 * 1024);
    memset(padvalues, 0, 10 * 4);

    char* path_ = get_current_dir_name();
    std::string path(path_);

    ReflectionPad3dGradTilingData* tilingDataFromBin = reinterpret_cast<ReflectionPad3dGradTilingData*>(tiling);
    tilingDataFromBin->batch = inputShape[0];
    tilingDataFromBin->channel = inputShape[1];
    tilingDataFromBin->depth = inputShape[2];
    tilingDataFromBin->height = inputShape[3];
    tilingDataFromBin->width = inputShape[4];
    tilingDataFromBin->alignHeight = ((inputShape[3] + 15) / 16) * 16;
    tilingDataFromBin->alignWidth = ((inputShape[4] + 15) / 16) * 16;
    tilingDataFromBin->outDepth = outputShape[2];
    tilingDataFromBin->outHeight = outputShape[3];
    tilingDataFromBin->outWidth = outputShape[4];
    tilingDataFromBin->dPad1 = padding[4];
    tilingDataFromBin->dPad2 = padding[5];
    tilingDataFromBin->hPad1 = padding[6];
    tilingDataFromBin->hPad2 = padding[7];
    tilingDataFromBin->wPad1 = padding[8];
    tilingDataFromBin->wPad2 = padding[9];
    tilingDataFromBin->blockNum = 1;
    tilingDataFromBin->ubFactorElement = (2 * 1024) / KEY_MAP[dtype];
    tilingDataFromBin->ncPerCore = inputShape[0] * inputShape[1];
    tilingDataFromBin->tailNC = 0;
    tilingDataFromBin->tilingKey = tiling_key;

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        reflection_pad3d_grad, tilingDataFromBin->blockNum, input, padvalues, output, workspace,
        (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(reflection_pad3d_grad_test, test_reflect_0)
// {
//     std::vector<uint32_t> inputShape = {2, 128, 18, 66, 66};        // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 100);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_1)
// {
//     std::vector<uint32_t> inputShape = {2, 64, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 101);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_2)
// {
//     std::vector<uint32_t> inputShape = {2, 256, 6, 18, 18};         // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 102);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_3)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 200);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_4)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 202);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_5)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 203);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_6)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 300);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_7)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 302);
// }

// TEST_F(reflection_pad3d_grad_test, test_reflect_8)
// {
//     std::vector<uint32_t> inputShape = {2, 32, 18, 130, 130};       // NCDHW
//     std::vector<uint32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}; // 10
//     CallSimpleKernel(inputShape, padding, 303);
// }
