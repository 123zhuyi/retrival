/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_kernel/right_shift_tiling_data.h"
#include "../../../op_kernel/right_shift_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

namespace {
constexpr uint64_t DEFAULT_CORE_NUM = 64;
constexpr uint64_t DEFAULT_UB_SIZE = 262144;
constexpr size_t EXPECT_WORKSPACE_SIZE = 16777216;

struct RightShiftCompileInfo {};
RightShiftCompileInfo g_compileInfo{};

gert::StorageShape Shape(std::initializer_list<int64_t> dims) { return {dims, dims}; }

gert::TilingContextPara BuildTilingContext(std::initializer_list<int64_t> xShape, std::initializer_list<int64_t> yShape,
                                           std::initializer_list<int64_t> zShape, ge::DataType xType,
                                           ge::DataType yType)
{
    return gert::TilingContextPara("RightShift",
                                   {
                                       {Shape(xShape), xType, ge::FORMAT_ND},
                                       {Shape(yShape), yType, ge::FORMAT_ND},
                                   },
                                   {
                                       {Shape(zShape), xType, ge::FORMAT_ND},
                                   },
                                   {}, &g_compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, sizeof(RightShiftTilingData));
}

RightShiftTilingData GetTilingData(const TilingInfo& tilingInfo)
{
    RightShiftTilingData tilingData{};
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(RightShiftTilingData));
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(RightShiftTilingData));
    return tilingData;
}

uint64_t ExpectedTilingKey(ge::DataType dtype, uint32_t mode)
{
    uint32_t dtypeTemplate = RIGHT_SHIFT_TPL_INT32 - 1;
    if (dtype == ge::DT_INT8) {
        dtypeTemplate = RIGHT_SHIFT_TPL_INT8 - 1;
    } else if (dtype == ge::DT_UINT8) {
        dtypeTemplate = RIGHT_SHIFT_TPL_UINT8 - 1;
    } else if (dtype == ge::DT_INT16) {
        dtypeTemplate = RIGHT_SHIFT_TPL_INT16 - 1;
    } else if (dtype == ge::DT_UINT16) {
        dtypeTemplate = RIGHT_SHIFT_TPL_UINT16 - 1;
    } else if (dtype == ge::DT_UINT32) {
        dtypeTemplate = RIGHT_SHIFT_TPL_UINT32 - 1;
    } else if (dtype == ge::DT_INT64) {
        dtypeTemplate = RIGHT_SHIFT_TPL_INT64 - 1;
    } else if (dtype == ge::DT_UINT64) {
        dtypeTemplate = RIGHT_SHIFT_TPL_UINT64 - 1;
    }
    return GET_TPL_TILING_KEY(mode * RIGHT_SHIFT_TPL_DTYPE_COUNT + dtypeTemplate);
}

void ExpectCommonTiling(const TilingInfo& tilingInfo, ge::DataType dtype, uint32_t mode, uint64_t totalLength,
                        uint32_t rank, uint32_t blockNum, RightShiftTilingData& tilingData)
{
    EXPECT_EQ(tilingInfo.tilingKey, ExpectedTilingKey(dtype, mode));
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), 1);
    if (tilingInfo.workspaceSizes.size() == 1) {
        EXPECT_EQ(tilingInfo.workspaceSizes[0], EXPECT_WORKSPACE_SIZE);
    }
    EXPECT_EQ(tilingInfo.blockNum, blockNum);

    tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalLength, totalLength);
    EXPECT_EQ(tilingData.rank, rank);
    EXPECT_EQ(tilingData.mode, mode);
}
} // namespace

class RightShiftTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "RightShiftTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "RightShiftTiling TearDown" << endl; }
};

TEST_F(RightShiftTiling, Int32Contiguous)
{
    auto tilingContextPara = BuildTilingContext({2, 3}, {2, 3}, {2, 3}, ge::DT_INT32, ge::DT_INT32);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_INT32, RIGHT_SHIFT_MODE_CONTIGUOUS, 6, 1, 1, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.tailCoreNum, 0);
    EXPECT_EQ(tilingData.formerCoreDataNum, 6);
    EXPECT_EQ(tilingData.tailCoreDataNum, 0);
    EXPECT_EQ(tilingData.outShape[0], 6);
    EXPECT_EQ(tilingData.xStride[0], 1);
    EXPECT_EQ(tilingData.yStride[0], 1);
}

TEST_F(RightShiftTiling, Uint8YScalar)
{
    auto tilingContextPara = BuildTilingContext({2, 3}, {1}, {2, 3}, ge::DT_UINT8, ge::DT_UINT8);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_UINT8, RIGHT_SHIFT_MODE_Y_SCALAR, 6, 1, 1, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.formerCoreDataNum, 6);
    EXPECT_EQ(tilingData.outShape[0], 6);
    EXPECT_EQ(tilingData.xStride[0], 1);
    EXPECT_EQ(tilingData.yStride[0], 0);
}

TEST_F(RightShiftTiling, Int64XScalar)
{
    auto tilingContextPara = BuildTilingContext({1}, {2, 3}, {2, 3}, ge::DT_INT64, ge::DT_INT64);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_INT64, RIGHT_SHIFT_MODE_X_SCALAR, 6, 1, 1, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.formerCoreDataNum, 6);
    EXPECT_EQ(tilingData.outShape[0], 6);
    EXPECT_EQ(tilingData.xStride[0], 0);
    EXPECT_EQ(tilingData.yStride[0], 1);
}

TEST_F(RightShiftTiling, Int16TailContiguous)
{
    auto tilingContextPara = BuildTilingContext({2, 1, 4}, {1, 3, 4}, {2, 3, 4}, ge::DT_INT16, ge::DT_INT16);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_INT16, RIGHT_SHIFT_MODE_TAIL_CONTIGUOUS, 24, 3, 1, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.formerCoreDataNum, 24);
    EXPECT_EQ(tilingData.outShape[0], 2);
    EXPECT_EQ(tilingData.outShape[1], 3);
    EXPECT_EQ(tilingData.outShape[2], 4);
    EXPECT_EQ(tilingData.xStride[0], 4);
    EXPECT_EQ(tilingData.xStride[1], 0);
    EXPECT_EQ(tilingData.xStride[2], 1);
    EXPECT_EQ(tilingData.yStride[0], 0);
    EXPECT_EQ(tilingData.yStride[1], 4);
    EXPECT_EQ(tilingData.yStride[2], 1);
}

TEST_F(RightShiftTiling, Uint32GeneralBroadcast)
{
    auto tilingContextPara = BuildTilingContext({2, 3, 1}, {1, 3, 4}, {2, 3, 4}, ge::DT_UINT32, ge::DT_UINT32);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_UINT32, RIGHT_SHIFT_MODE_GENERAL, 24, 3, 1, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.formerCoreDataNum, 24);
    EXPECT_EQ(tilingData.outShape[0], 2);
    EXPECT_EQ(tilingData.outShape[1], 3);
    EXPECT_EQ(tilingData.outShape[2], 4);
    EXPECT_EQ(tilingData.xStride[0], 3);
    EXPECT_EQ(tilingData.xStride[1], 1);
    EXPECT_EQ(tilingData.xStride[2], 0);
    EXPECT_EQ(tilingData.yStride[0], 0);
    EXPECT_EQ(tilingData.yStride[1], 4);
    EXPECT_EQ(tilingData.yStride[2], 1);
}

TEST_F(RightShiftTiling, Int16YScalarMultiCore)
{
    auto tilingContextPara = BuildTilingContext({1024}, {1}, {1024}, ge::DT_INT16, ge::DT_INT16);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    RightShiftTilingData tilingData{};
    ExpectCommonTiling(tilingInfo, ge::DT_INT16, RIGHT_SHIFT_MODE_Y_SCALAR, 1024, 1, 32, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 32);
    EXPECT_EQ(tilingData.tailCoreNum, 0);
    EXPECT_EQ(tilingData.formerCoreDataNum, 32);
    EXPECT_EQ(tilingData.tailCoreDataNum, 0);
    EXPECT_EQ(tilingData.outShape[0], 1024);
    EXPECT_EQ(tilingData.xStride[0], 1);
    EXPECT_EQ(tilingData.yStride[0], 0);
}

TEST_F(RightShiftTiling, DtypeMismatchFailed)
{
    auto tilingContextPara = BuildTilingContext({2, 3}, {2, 3}, {2, 3}, ge::DT_INT32, ge::DT_INT64);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(RightShiftTiling, UnbroadcastableShapeFailed)
{
    auto tilingContextPara = BuildTilingContext({2, 3}, {4, 3}, {2, 3}, ge::DT_INT32, ge::DT_INT32);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
