/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/accumulate_nv2_tiling_data.h"

// Skip tiling_key.h (uses ASCENDC_TPL macros incompatible with tikicpulib)
#define __ACCUMULATE_NV2_TILING_KEY_H__
#include "../../../op_kernel/arch35/accumulate_nv2.h"

using namespace NsAccumulateNV2;

void kernel_accumulate_nv2_float_single(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AccumulateNV2TilingData, tilingData, tiling);
    AccumulateNV2<float, 0> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

void kernel_accumulate_nv2_float_double(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AccumulateNV2TilingData, tilingData, tiling);
    AccumulateNV2<float, 1> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

void kernel_accumulate_nv2_half_single(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AccumulateNV2TilingData, tilingData, tiling);
    AccumulateNV2<half, 0> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

void kernel_accumulate_nv2_int32_single(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AccumulateNV2TilingData, tilingData, tiling);
    AccumulateNV2<int32_t, 0> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

class AccumulateNV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AccumulateNV2KernelTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AccumulateNV2KernelTest TearDown" << std::endl;
    }
};

struct TensorListBuilder {
    std::vector<uint8_t*> tensors;
    uint8_t* tensorListMem = nullptr;

    uint8_t* Build(int32_t inputNum)
    {
        size_t headerSize = sizeof(uint64_t) * (1 + inputNum);
        tensorListMem = (uint8_t*)AscendC::GmAlloc(headerSize);
        uint64_t* header = reinterpret_cast<uint64_t*>(tensorListMem);
        header[0] = sizeof(uint64_t);
        for (int32_t i = 0; i < inputNum; i++) {
            header[1 + i] = reinterpret_cast<uint64_t>(tensors[i]);
        }
        return tensorListMem;
    }

    void Free()
    {
        for (auto* t : tensors) {
            AscendC::GmFree(t);
        }
        if (tensorListMem) {
            AscendC::GmFree(tensorListMem);
        }
    }
};

// float32, 3 inputs of 32 elements, single buffer
TEST_F(AccumulateNV2KernelTest, test_float32_3inputs_single_buffer)
{
    constexpr int32_t INPUT_NUM = 3;
    constexpr int64_t TOTAL_NUM = 32;
    constexpr size_t ELEM_SIZE = sizeof(float);

    TensorListBuilder builder;
    for (int32_t i = 0; i < INPUT_NUM; i++) {
        float* tensor = (float*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
        for (int64_t j = 0; j < TOTAL_NUM; j++) {
            tensor[j] = static_cast<float>(i + 1);
        }
        builder.tensors.push_back((uint8_t*)tensor);
    }
    uint8_t* x = builder.Build(INPUT_NUM);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    memset(y, 0, TOTAL_NUM * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNV2TilingData));
    AccumulateNV2TilingData* td = reinterpret_cast<AccumulateNV2TilingData*>(tiling);
    td->totalNum = TOTAL_NUM;
    td->blockFactor = TOTAL_NUM;
    td->ubFactor = TOTAL_NUM;
    td->inputNum = INPUT_NUM;

    ICPU_SET_TILING_KEY(210010000UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_accumulate_nv2_float_single, 1, x, y, workspace, tiling);

    builder.Free();
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// float32, 1 input of 32 elements, single buffer (passthrough path)
TEST_F(AccumulateNV2KernelTest, test_float32_1input)
{
    constexpr int32_t INPUT_NUM = 1;
    constexpr int64_t TOTAL_NUM = 32;
    constexpr size_t ELEM_SIZE = sizeof(float);

    TensorListBuilder builder;
    float* tensor = (float*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    for (int64_t j = 0; j < TOTAL_NUM; j++) {
        tensor[j] = static_cast<float>(j);
    }
    builder.tensors.push_back((uint8_t*)tensor);
    uint8_t* x = builder.Build(INPUT_NUM);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    memset(y, 0, TOTAL_NUM * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNV2TilingData));
    AccumulateNV2TilingData* td = reinterpret_cast<AccumulateNV2TilingData*>(tiling);
    td->totalNum = TOTAL_NUM;
    td->blockFactor = TOTAL_NUM;
    td->ubFactor = TOTAL_NUM;
    td->inputNum = INPUT_NUM;

    ICPU_SET_TILING_KEY(210010000UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_accumulate_nv2_float_single, 1, x, y, workspace, tiling);

    builder.Free();
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// float32, 2 inputs of 2048 elements, double buffer
TEST_F(AccumulateNV2KernelTest, test_float32_2inputs_double_buffer)
{
    constexpr int32_t INPUT_NUM = 2;
    constexpr int64_t TOTAL_NUM = 2048;
    constexpr size_t ELEM_SIZE = sizeof(float);

    TensorListBuilder builder;
    for (int32_t i = 0; i < INPUT_NUM; i++) {
        float* tensor = (float*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
        for (int64_t j = 0; j < TOTAL_NUM; j++) {
            tensor[j] = static_cast<float>(i + 1);
        }
        builder.tensors.push_back((uint8_t*)tensor);
    }
    uint8_t* x = builder.Build(INPUT_NUM);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    memset(y, 0, TOTAL_NUM * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNV2TilingData));
    AccumulateNV2TilingData* td = reinterpret_cast<AccumulateNV2TilingData*>(tiling);
    td->totalNum = TOTAL_NUM;
    td->blockFactor = TOTAL_NUM;
    td->ubFactor = 256;
    td->inputNum = INPUT_NUM;

    ICPU_SET_TILING_KEY(210000000UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_accumulate_nv2_float_double, 1, x, y, workspace, tiling);

    builder.Free();
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// float16, 2 inputs of 32 elements, single buffer
TEST_F(AccumulateNV2KernelTest, test_float16_2inputs_single_buffer)
{
    constexpr int32_t INPUT_NUM = 2;
    constexpr int64_t TOTAL_NUM = 32;
    constexpr size_t ELEM_SIZE = sizeof(half);

    TensorListBuilder builder;
    for (int32_t i = 0; i < INPUT_NUM; i++) {
        half* tensor = (half*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
        for (int64_t j = 0; j < TOTAL_NUM; j++) {
            tensor[j] = static_cast<half>(static_cast<float>(i + 1));
        }
        builder.tensors.push_back((uint8_t*)tensor);
    }
    uint8_t* x = builder.Build(INPUT_NUM);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    memset(y, 0, TOTAL_NUM * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNV2TilingData));
    AccumulateNV2TilingData* td = reinterpret_cast<AccumulateNV2TilingData*>(tiling);
    td->totalNum = TOTAL_NUM;
    td->blockFactor = TOTAL_NUM;
    td->ubFactor = TOTAL_NUM;
    td->inputNum = INPUT_NUM;

    ICPU_SET_TILING_KEY(210010000UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_accumulate_nv2_half_single, 1, x, y, workspace, tiling);

    builder.Free();
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// int32, 3 inputs of 32 elements, single buffer
TEST_F(AccumulateNV2KernelTest, test_int32_3inputs_single_buffer)
{
    constexpr int32_t INPUT_NUM = 3;
    constexpr int64_t TOTAL_NUM = 32;
    constexpr size_t ELEM_SIZE = sizeof(int32_t);

    TensorListBuilder builder;
    for (int32_t i = 0; i < INPUT_NUM; i++) {
        int32_t* tensor = (int32_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
        for (int64_t j = 0; j < TOTAL_NUM; j++) {
            tensor[j] = (i + 1) * 10;
        }
        builder.tensors.push_back((uint8_t*)tensor);
    }
    uint8_t* x = builder.Build(INPUT_NUM);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL_NUM * ELEM_SIZE);
    memset(y, 0, TOTAL_NUM * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNV2TilingData));
    AccumulateNV2TilingData* td = reinterpret_cast<AccumulateNV2TilingData*>(tiling);
    td->totalNum = TOTAL_NUM;
    td->blockFactor = TOTAL_NUM;
    td->ubFactor = TOTAL_NUM;
    td->inputNum = INPUT_NUM;

    ICPU_SET_TILING_KEY(210010000UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_accumulate_nv2_int32_single, 1, x, y, workspace, tiling);

    builder.Free();
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
