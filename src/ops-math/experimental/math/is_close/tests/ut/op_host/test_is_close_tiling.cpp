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
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_kernel/is_close_tiling_data.h"
#include "../../../op_kernel/is_close_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

namespace {
constexpr uint64_t DEFAULT_CORE_NUM = 64;
constexpr uint64_t DEFAULT_UB_SIZE = 262144;
constexpr size_t EXPECT_WORKSPACE_SIZE = 16777216;

struct IsCloseCompileInfo {};
IsCloseCompileInfo g_compileInfo {};

gert::StorageShape Shape(std::initializer_list<int64_t> dims)
{
    return {dims, dims};
}

std::vector<gert::TilingContextPara::OpAttr> Attrs(float rtol, float atol, bool equalNan)
{
    return {
        {"rtol", Ops::Math::AnyValue::CreateFrom<float>(rtol)},
        {"atol", Ops::Math::AnyValue::CreateFrom<float>(atol)},
        {"equal_nan", Ops::Math::AnyValue::CreateFrom<bool>(equalNan)},
    };
}

gert::TilingContextPara BuildTilingContext(
    std::initializer_list<int64_t> selfShape, std::initializer_list<int64_t> otherShape,
    std::initializer_list<int64_t> outShape, ge::DataType selfType, ge::DataType otherType,
    const std::vector<gert::TilingContextPara::OpAttr>& attrs = {})
{
    return gert::TilingContextPara(
        "IsClose",
        {
            {Shape(selfShape), selfType, ge::FORMAT_ND},
            {Shape(otherShape), otherType, ge::FORMAT_ND},
        },
        {
            {Shape(outShape), ge::DT_BOOL, ge::FORMAT_ND},
        },
        attrs,
        &g_compileInfo,
        DEFAULT_CORE_NUM,
        DEFAULT_UB_SIZE,
        sizeof(IsCloseTilingData));
}

IsCloseTilingData GetTilingData(const TilingInfo& tilingInfo)
{
    IsCloseTilingData tilingData {};
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(IsCloseTilingData));
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(IsCloseTilingData));
    return tilingData;
}

uint64_t ExpectedTilingKey(ge::DataType dtype, uint32_t broadcastMode)
{
    uint32_t dtypeTemplate = IS_CLOSE_TPL_FP32 - 1;
    if (dtype == ge::DT_FLOAT16) {
        dtypeTemplate = IS_CLOSE_TPL_FP16 - 1;
    } else if (dtype == ge::DT_BF16) {
        dtypeTemplate = IS_CLOSE_TPL_BF16 - 1;
    } else if (dtype == ge::DT_INT32) {
        dtypeTemplate = IS_CLOSE_TPL_INT32 - 1;
    }
    return GET_TPL_TILING_KEY(broadcastMode * IS_CLOSE_TPL_DTYPE_COUNT + dtypeTemplate);
}

void ExpectBroadcastInfo(
    const IsCloseTilingData& tilingData, uint32_t rank, uint32_t mode, const std::vector<uint64_t>& outShape,
    const std::vector<uint64_t>& x1Stride, const std::vector<uint64_t>& x2Stride)
{
    EXPECT_EQ(tilingData.rank, rank);
    EXPECT_EQ(tilingData.broadcastMode, mode);
    ASSERT_EQ(outShape.size(), rank);
    ASSERT_EQ(x1Stride.size(), rank);
    ASSERT_EQ(x2Stride.size(), rank);
    for (uint32_t i = 0; i < rank; ++i) {
        EXPECT_EQ(tilingData.outShape[i], outShape[i]);
        EXPECT_EQ(tilingData.x1Stride[i], x1Stride[i]);
        EXPECT_EQ(tilingData.x2Stride[i], x2Stride[i]);
    }
}
} // namespace

class IsCloseTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "IsCloseTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "IsCloseTiling TearDown" << endl;
    }
};

TEST_F(IsCloseTiling, Float32SingleCore)
{
    auto tilingContextPara = BuildTilingContext({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, ExpectedTilingKey(ge::DT_FLOAT, IS_CLOSE_BROADCAST_MODE_CONTIGUOUS));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], EXPECT_WORKSPACE_SIZE);
    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.formerCoreNum, 1);
    EXPECT_EQ(tilingData.tailCoreNum, 0);
    EXPECT_EQ(tilingData.formerCoreDataNum, 6);
    EXPECT_EQ(tilingData.formerCoreLoopCount, 1);
    EXPECT_EQ(tilingData.formerCoreFormerDataNum, 6);
    EXPECT_EQ(tilingData.formerCoreTailDataNum, 6);
    EXPECT_FLOAT_EQ(tilingData.rtol, 1e-5f);
    EXPECT_FLOAT_EQ(tilingData.atol, 1e-8f);
    EXPECT_EQ(tilingData.equalNan, 0U);
    EXPECT_EQ(tilingData.totalLength, 6);
    ExpectBroadcastInfo(
        tilingData, 1, IS_CLOSE_BROADCAST_MODE_CONTIGUOUS, std::vector<uint64_t>{6}, std::vector<uint64_t>{1},
        std::vector<uint64_t>{1});
}
