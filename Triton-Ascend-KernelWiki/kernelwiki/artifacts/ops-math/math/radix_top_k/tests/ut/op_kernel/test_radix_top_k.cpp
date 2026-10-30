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
 * \file test_radix_top_k.cpp
 * \brief RadixTopK op_kernel UT
 */

#include <iostream>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

extern "C" __global__ __aicore__ void radix_top_k(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling);

class RadixTopKKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RadixTopKKernelTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RadixTopKKernelTest TearDown" << std::endl;
    }
};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

struct RadixTopKCompileInfo {
    int32_t totalCoreNum;
    uint64_t ubSizePlatForm;
};

namespace {

bool RunRadixTopKTiling(
    const gert::StorageShape &xShape,
    ge::DataType xDtype,
    int32_t kValue,
    bool sorted,
    bool largest,
    TilingInfo &tilingInfo)
{
    gert::StorageShape kShape({1}, {1});

    RadixTopKCompileInfo compileInfo;
    compileInfo.totalCoreNum = 48;
    compileInfo.ubSizePlatForm = 196608;

    auto* compileInfoPtr = new RadixTopKCompileInfo(compileInfo);

    gert::TilingContextPara tilingCtx(
        "RadixTopK",
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {kShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(&kValue)},
        },
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {xShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(sorted)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(largest)),
            gert::TilingContextPara::OpAttr("indices_dtype",
                Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        },
        compileInfoPtr);

    return ExecuteTiling(tilingCtx, tilingInfo);
}

}  // namespace

TEST_F(RadixTopKKernelTest, ub_fp16_1d_128_k5_sorted_largest)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunRadixTopKTiling({{128}, {128}}, ge::DT_FLOAT16, 5, true, true, tilingInfo));

    uint32_t kVal = 5;
    size_t xByteSize = CeilAlign(128 * sizeof(half), 32);
    size_t kByteSize = CeilAlign(sizeof(int32_t), 32);
    size_t valueByteSize = CeilAlign(kVal * sizeof(half), 32);
    size_t indexByteSize = CeilAlign(kVal * sizeof(int32_t), 32);
    size_t wsByteSize = tilingInfo.workspaceSizes.empty() ? 0 :
        CeilAlign(tilingInfo.workspaceSizes[0], 32);

    uint8_t* xGm = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kGm = (uint8_t*)AscendC::GmAlloc(kByteSize);
    ASSERT_NE(kGm, nullptr);
    uint8_t* valuesGm = (uint8_t*)AscendC::GmAlloc(valueByteSize);
    uint8_t* indicesGm = (uint8_t*)AscendC::GmAlloc(indexByteSize);
    uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(wsByteSize);
    uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    *reinterpret_cast<int32_t*>(kGm) = kVal;
    std::memcpy(tilingGm, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(radix_top_k, tilingInfo.blockNum,
        xGm, kGm, valuesGm, indicesGm, workspaceGm, tilingGm);

    AscendC::GmFree((void*)xGm);
    AscendC::GmFree((void*)kGm);
    AscendC::GmFree((void*)valuesGm);
    AscendC::GmFree((void*)indicesGm);
    AscendC::GmFree((void*)workspaceGm);
    AscendC::GmFree((void*)tilingGm);
}

TEST_F(RadixTopKKernelTest, ub_fp16_2d_32x512_k10_sorted_largest)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunRadixTopKTiling({{32, 512}, {32, 512}}, ge::DT_FLOAT16, 10, true, true, tilingInfo));

    uint32_t batchSize = 32;
    uint32_t sortLen = 512;
    uint32_t kVal = 10;
    size_t xByteSize = CeilAlign(batchSize * sortLen * sizeof(half), 32);
    size_t kByteSize = CeilAlign(sizeof(int32_t), 32);
    size_t valueByteSize = CeilAlign(batchSize * kVal * sizeof(half), 32);
    size_t indexByteSize = CeilAlign(batchSize * kVal * sizeof(int32_t), 32);
    size_t wsByteSize = tilingInfo.workspaceSizes.empty() ? 0 :
        CeilAlign(tilingInfo.workspaceSizes[0], 32);

    uint8_t* xGm = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kGm = (uint8_t*)AscendC::GmAlloc(kByteSize);
    uint8_t* valuesGm = (uint8_t*)AscendC::GmAlloc(valueByteSize);
    uint8_t* indicesGm = (uint8_t*)AscendC::GmAlloc(indexByteSize);
    uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(wsByteSize);
    uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    *reinterpret_cast<int32_t*>(kGm) = kVal;
    std::memcpy(tilingGm, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(radix_top_k, tilingInfo.blockNum,
        xGm, kGm, valuesGm, indicesGm, workspaceGm, tilingGm);

    AscendC::GmFree((void*)xGm);
    AscendC::GmFree((void*)kGm);
    AscendC::GmFree((void*)valuesGm);
    AscendC::GmFree((void*)indicesGm);
    AscendC::GmFree((void*)workspaceGm);
    AscendC::GmFree((void*)tilingGm);
}
