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
 * \file test_accumulate_nv2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/accumulate_nv2.cpp"

using namespace std;

class AccumulateNv2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "accumulate_nv2_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./accumulate_nv2_data/");
    }
    static void TearDownTestCase() { std::cout << "accumulate_nv2_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AccumulateNv2Test::rootPath = "../../../../experimental/";
const std::string AccumulateNv2Test::dataPath = rootPath + "math/accumulate_nv2/tests/ut/op_kernel/accumulate_nv2_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(AccumulateNv2Test, test_case_float32_1)
{
    system("pwd");
    system("ls -la accumulate_nv2_data 2>/dev/null || echo 'accumulate_nv2_data not found'");

    uint32_t blockDim = 1;
    system("cd ./accumulate_nv2_data/ && python3 gen_data.py '(2, 90)' 'float32'");

    uint32_t dataCount = 90;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string x1fileName = "./accumulate_nv2_data/float32_input_accumulate_nv2_x1.bin";
    std::string x2fileName = "./accumulate_nv2_data/float32_input_accumulate_nv2_x2.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(x1fileName, inputByteSize, x1, inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(x2fileName, inputByteSize, x2, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AccumulateNv2TilingData));

    AccumulateNv2TilingData* tilingData = reinterpret_cast<AccumulateNv2TilingData*>(tiling);

    tilingData->smallCoreDataNum = 96;
    tilingData->bigCoreDataNum = 104;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 12272;
    tilingData->smallTailDataNum = 96;
    tilingData->bigTailDataNum = 104;
    tilingData->tailBlockNum = 0;
    tilingData->num = 2;

    // 构造符合 AccumulateNv2 内部逻辑的内存布局
    const int tensor_count = 2;
    // 布局：[Offset_Bytes, TensorAddr_0, TensorAddr_1, ...]
    // 需要 1 个头 + 2 个地址
    size_t count_total = 1 + tensor_count;
    size_t byte_size = count_total * sizeof(uint64_t);

    uint64_t* h_data = (uint64_t*)malloc(byte_size);

    // 第一步：填入偏移量 (Offset)
    // 解释：GetTensorAddr 读取 *dataAddr 作为字节偏移，右移 3 位变成元素个数偏移
    // 我们希望 tensorPtr 指向第二个元素 (index 1)，所以字节偏移应该是 sizeof(uint64_t) = 8
    h_data[0] = (uint64_t)sizeof(uint64_t);

    // 第二步：填入真实的 Tensor 地址
    h_data[1] = (uint64_t)(uintptr_t)x1;
    h_data[2] = (uint64_t)(uintptr_t)x2;

    // 第三步：分配到 GM 并拷贝
    uint64_t* d_data = (uint64_t*)AscendC::GmAlloc(byte_size);
    std::memcpy(d_data, h_data, byte_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = accumulate_nv2<ELEMENTWISE_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, reinterpret_cast<uint8_t*>(d_data), y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./accumulate_nv2_data/float32_output_accumulate_nv2.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(d_data));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(h_data);
    system("cd ./accumulate_nv2_data/ && python3 compare_data.py 'float32'");
}