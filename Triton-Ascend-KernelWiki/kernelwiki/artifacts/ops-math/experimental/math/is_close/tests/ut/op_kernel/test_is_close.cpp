/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/is_close.cpp"

using namespace std;

namespace {
template <typename T>
constexpr uint32_t TplDtype()
{
    if constexpr (std::is_same_v<T, float>) {
        return IS_CLOSE_TPL_FP32 - 1;
    } else if constexpr (std::is_same_v<T, half>) {
        return IS_CLOSE_TPL_FP16 - 1;
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return IS_CLOSE_TPL_BF16 - 1;
    } else {
        return IS_CLOSE_TPL_INT32 - 1;
    }
}

template <uint32_t BROADCAST_MODE>
constexpr uint32_t TplKey()
{
    return BROADCAST_MODE * IS_CLOSE_TPL_DTYPE_COUNT;
}
} // namespace

class IsCloseKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "is_close kernel test SetUp" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(system(cmd.c_str()), 0);
        ASSERT_EQ(system("chmod -R 755 ./is_close_data/"), 0);
    }

    static void TearDownTestCase()
    {
        cout << "is_close kernel test TearDown" << endl;
    }

    template <typename T, uint32_t BROADCAST_MODE>
    void RunIsCloseCase(
        const string& dtype, const string& caseName, uint32_t x1Count, uint32_t x2Count, uint32_t yCount, float rtol,
        float atol, uint32_t equalNan, uint32_t rank, const vector<uint64_t>& outShape,
        const vector<uint64_t>& x1Stride, const vector<uint64_t>& x2Stride)
    {
        constexpr uint32_t tileBufferLen = 256;
        constexpr uint32_t blockDim = 1;
        const string prefix = dtype + "_" + caseName;
        const string genCmd = "cd ./is_close_data/ && python3 gen_data.py '" + dtype + "' '" + caseName + "'";
        ASSERT_EQ(system(genCmd.c_str()), 0);

        size_t x1ByteSize = x1Count * sizeof(T);
        size_t x2ByteSize = x2Count * sizeof(T);
        uint8_t* x1 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(x1ByteSize, 32)));
        uint8_t* x2 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(x2ByteSize, 32)));
        ASSERT_NE(x1, nullptr);
        ASSERT_NE(x2, nullptr);

        ASSERT_TRUE(ReadFile("./is_close_data/" + prefix + "_input_x1_is_close.bin", x1ByteSize, x1, x1ByteSize));
        ASSERT_TRUE(ReadFile("./is_close_data/" + prefix + "_input_x2_is_close.bin", x2ByteSize, x2, x2ByteSize));

        size_t outputByteSize = yCount * sizeof(int8_t);
        uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
        uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
        uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(IsCloseTilingData)));
        ASSERT_NE(y, nullptr);
        ASSERT_NE(workspace, nullptr);
        ASSERT_NE(tiling, nullptr);

        auto* tilingData = reinterpret_cast<IsCloseTilingData*>(tiling);
        std::memset(tilingData, 0, sizeof(IsCloseTilingData));
        tilingData->formerCoreNum = 1;
        tilingData->tailCoreNum = 0;
        tilingData->formerCoreDataNum = yCount;
        tilingData->tailCoreDataNum = 0;
        tilingData->formerCoreLoopCount = (yCount + tileBufferLen - 1) / tileBufferLen;
        tilingData->formerCoreFormerDataNum = yCount > tileBufferLen ? tileBufferLen : yCount;
        tilingData->formerCoreTailDataNum =
            yCount % tileBufferLen == 0 ? tileBufferLen : yCount % tileBufferLen;
        tilingData->tailCoreLoopCount = 0;
        tilingData->tailCoreFormerDataNum = 0;
        tilingData->tailCoreTailDataNum = 0;
        tilingData->tileBufferLen = tileBufferLen;
        tilingData->rtol = rtol;
        tilingData->atol = atol;
        tilingData->equalNan = equalNan;
        tilingData->rank = rank;
        tilingData->broadcastMode = BROADCAST_MODE;
        tilingData->totalLength = yCount;
        for (uint32_t i = 0; i < rank; ++i) {
            tilingData->outShape[i] = outShape[i];
            tilingData->x1Stride[i] = x1Stride[i];
            tilingData->x2Stride[i] = x2Stride[i];
        }

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        auto func = is_close<TplKey<BROADCAST_MODE>() + TplDtype<T>()>;
        ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, reinterpret_cast<uint8_t*>(tilingData));

        ASSERT_TRUE(WriteFile("./is_close_data/" + prefix + "_output_y_is_close.bin", y, outputByteSize));

        AscendC::GmFree(reinterpret_cast<void*>(x1));
        AscendC::GmFree(reinterpret_cast<void*>(x2));
        AscendC::GmFree(reinterpret_cast<void*>(y));
        AscendC::GmFree(reinterpret_cast<void*>(workspace));
        AscendC::GmFree(reinterpret_cast<void*>(tiling));

        const string cmpCmd = "cd ./is_close_data/ && python3 compare_data.py '" + dtype + "' '" + caseName + "'";
        ASSERT_EQ(system(cmpCmd.c_str()), 0);
    }

private:
    template <typename T1, typename T2>
    static T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }

    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string IsCloseKernelTest::rootPath = "../../../../";
const std::string IsCloseKernelTest::dataPath =
    IsCloseKernelTest::rootPath + "experimental/math/is_close/tests/ut/op_kernel/is_close_data";

TEST_F(IsCloseKernelTest, Float32EqualNan)
{
    RunIsCloseCase<float, IS_CLOSE_BROADCAST_MODE_CONTIGUOUS>(
        "float32", "equal", 64, 64, 64, 1e-5f, 1e-8f, 1U, 1,
        vector<uint64_t>{64}, vector<uint64_t>{1}, vector<uint64_t>{1});
}
