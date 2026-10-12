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
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/right_shift.cpp"

using namespace std;

namespace {
template <typename T1, typename T2>
T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
T GoldenRightShift(T x, T y)
{
    constexpr uint32_t bitWidth = sizeof(T) * 8;
    if constexpr (std::is_signed<T>::value) {
        if (y < static_cast<T>(0) || y >= static_cast<T>(bitWidth)) {
            return x < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(0);
        }
        return static_cast<T>(static_cast<int64_t>(x) >> static_cast<uint32_t>(y));
    }

    if (y >= static_cast<T>(bitWidth)) {
        return static_cast<T>(0);
    }
    return static_cast<T>(static_cast<uint64_t>(x) >> static_cast<uint32_t>(y));
}

uint64_t CalcHostOffset(uint64_t outIndex, const std::vector<uint64_t>& outShape, const std::vector<uint64_t>& stride)
{
    uint64_t offset = 0;
    uint64_t remain = outIndex;
    for (int32_t i = static_cast<int32_t>(outShape.size()) - 1; i >= 0; --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        uint64_t coord = remain % outShape[idx];
        remain /= outShape[idx];
        offset += coord * stride[idx];
    }
    return offset;
}

uint64_t CalcElementCount(const std::vector<uint64_t>& outShape)
{
    uint64_t count = 1;
    for (uint64_t dim : outShape) {
        count *= dim;
    }
    return count;
}

void FillTilingData(RightShiftTilingData* tilingData, uint64_t totalLength, uint32_t mode,
                    const std::vector<uint64_t>& outShape, const std::vector<uint64_t>& xStride,
                    const std::vector<uint64_t>& yStride)
{
    constexpr uint32_t tileBufferLen = 256;
    std::memset(tilingData, 0, sizeof(RightShiftTilingData));
    tilingData->formerCoreNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->formerCoreDataNum = totalLength;
    tilingData->tailCoreDataNum = 0;
    tilingData->tileBufferLen = tileBufferLen;
    tilingData->totalLength = totalLength;
    tilingData->rank = static_cast<uint32_t>(outShape.size());
    tilingData->mode = mode;
    for (size_t i = 0; i < outShape.size(); ++i) {
        tilingData->outShape[i] = outShape[i];
        tilingData->xStride[i] = xStride[i];
        tilingData->yStride[i] = yStride[i];
    }
}

template <typename T>
void ExpectValueEq(T actual, T expected, uint32_t index)
{
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        EXPECT_EQ(static_cast<int32_t>(actual), static_cast<int32_t>(expected)) << "index: " << index;
    } else {
        EXPECT_EQ(actual, expected) << "index: " << index;
    }
}

template <typename T, uint32_t BROADCAST_MODE, uint32_t DTYPE_MODE>
void RunKernelCase(const std::vector<T>& xHost, const std::vector<T>& yHost, const std::vector<uint64_t>& outShape,
                   const std::vector<uint64_t>& xStride, const std::vector<uint64_t>& yStride)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t tplKey = BROADCAST_MODE * RIGHT_SHIFT_TPL_DTYPE_COUNT + DTYPE_MODE - 1;
    uint64_t dataCount = CalcElementCount(outShape);
    ASSERT_EQ(xStride.size(), outShape.size());
    ASSERT_EQ(yStride.size(), outShape.size());

    std::vector<T> expectHost;
    expectHost.reserve(dataCount);
    for (uint64_t i = 0; i < dataCount; ++i) {
        uint64_t xOffset = CalcHostOffset(i, outShape, xStride);
        uint64_t yOffset = CalcHostOffset(i, outShape, yStride);
        ASSERT_LT(xOffset, xHost.size());
        ASSERT_LT(yOffset, yHost.size());
        expectHost.push_back(GoldenRightShift(xHost[xOffset], yHost[yOffset]));
    }

    size_t xByteSize = xHost.size() * sizeof(T);
    size_t yByteSize = yHost.size() * sizeof(T);
    size_t zByteSize = dataCount * sizeof(T);
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(xByteSize, 32)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(yByteSize, 32)));
    uint8_t* z = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(zByteSize, 32)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(RightShiftTilingData)));
    auto freeBuffer = [](uint8_t*& buffer) {
        if (buffer != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(buffer));
            buffer = nullptr;
        }
    };
    auto freeAllBuffers = [&]() {
        freeBuffer(x);
        freeBuffer(y);
        freeBuffer(z);
        freeBuffer(workspace);
        freeBuffer(tiling);
    };
    if (x == nullptr || y == nullptr || z == nullptr || workspace == nullptr || tiling == nullptr) {
        freeAllBuffers();
        FAIL() << "GmAlloc failed.";
    }

    std::memcpy(x, xHost.data(), xByteSize);
    std::memcpy(y, yHost.data(), yByteSize);
    std::memset(z, 0, zByteSize);

    auto* tilingData = reinterpret_cast<RightShiftTilingData*>(tiling);
    FillTilingData(tilingData, dataCount, BROADCAST_MODE, outShape, xStride, yStride);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = right_shift<tplKey>;
    ICPU_RUN_KF(func, blockDim, x, y, z, workspace, reinterpret_cast<uint8_t*>(tilingData));

    auto* zHost = reinterpret_cast<T*>(z);
    for (uint64_t i = 0; i < dataCount; ++i) {
        ExpectValueEq(zHost[i], expectHost[i], static_cast<uint32_t>(i));
    }

    freeAllBuffers();
}
} // namespace

class RightShiftKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "right_shift kernel test SetUp" << endl; }

    static void TearDownTestCase() { cout << "right_shift kernel test TearDown" << endl; }
};

TEST_F(RightShiftKernelTest, Int32Contiguous)
{
    const std::vector<int32_t> xHost = {-16, -8, -1, 0, 1, 8, 16, 32};
    const std::vector<int32_t> yHost = {0, 1, 2, 3, -1, 32, 4, 5};
    RunKernelCase<int32_t, RIGHT_SHIFT_MODE_CONTIGUOUS, RIGHT_SHIFT_TPL_INT32>(xHost, yHost, {8}, {1}, {1});
}

TEST_F(RightShiftKernelTest, Uint32YScalar)
{
    const std::vector<uint32_t> xHost = {0, 1, 7, 8, 16, 1024, 0x80000000U, 0xFFFFFFFFU};
    const std::vector<uint32_t> yHost = {3};
    RunKernelCase<uint32_t, RIGHT_SHIFT_MODE_Y_SCALAR, RIGHT_SHIFT_TPL_UINT32>(xHost, yHost, {8}, {1}, {0});
}

TEST_F(RightShiftKernelTest, Int64XScalar)
{
    const std::vector<int64_t> xHost = {-1024};
    const std::vector<int64_t> yHost = {0, 1, 7, 8, 31, 32, 63, 64, -1};
    RunKernelCase<int64_t, RIGHT_SHIFT_MODE_X_SCALAR, RIGHT_SHIFT_TPL_INT64>(xHost, yHost, {9}, {0}, {1});
}

TEST_F(RightShiftKernelTest, Int32TailContiguous)
{
    const std::vector<int32_t> xHost = {-256, -128, -64, -32, -16, -8, -4, -2, 2, 4, 8, 16, 32, 64, 128, 256};
    const std::vector<int32_t> yHost = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 31, 32, -1, 0};
    RunKernelCase<int32_t, RIGHT_SHIFT_MODE_TAIL_CONTIGUOUS, RIGHT_SHIFT_TPL_INT32>(xHost, yHost, {2, 2, 8}, {8, 0, 1},
                                                                                    {0, 8, 1});
}

TEST_F(RightShiftKernelTest, Int8GeneralBroadcast)
{
    const std::vector<int8_t> xHost = {-64, -8, -1, 0, 32, 127};
    const std::vector<int8_t> yHost = {0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 1, 2};
    RunKernelCase<int8_t, RIGHT_SHIFT_MODE_GENERAL, RIGHT_SHIFT_TPL_INT8>(xHost, yHost, {2, 3, 4}, {3, 1, 0},
                                                                          {0, 4, 1});
}
