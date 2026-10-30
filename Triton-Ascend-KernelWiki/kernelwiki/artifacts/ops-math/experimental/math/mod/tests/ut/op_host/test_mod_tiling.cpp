/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_kernel/mod_tiling_data.h"

namespace optiling {
struct ModCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};
} // namespace optiling

namespace ModNs {
bool operator==(const ModTilingData& lhs, const ModTilingData& rhs)
{
    if (lhs.usableUbSize != rhs.usableUbSize || lhs.needCoreNum != rhs.needCoreNum ||
        lhs.totalDataCount != rhs.totalDataCount || lhs.perCoreDataCount != rhs.perCoreDataCount ||
        lhs.tailDataCoreNum != rhs.tailDataCoreNum || lhs.lastCoreDataCount != rhs.lastCoreDataCount ||
        lhs.isInput2Scalar != rhs.isInput2Scalar || lhs.isInput2SameShape != rhs.isInput2SameShape ||
        lhs.dimNum != rhs.dimNum) {
        return false;
    }
    for (uint32_t i = 0; i < 8; ++i) {
        if (lhs.input1Shape[i] != rhs.input1Shape[i] || lhs.input2Shape[i] != rhs.input2Shape[i] ||
            lhs.input2Stride[i] != rhs.input2Stride[i]) {
            return false;
        }
    }
    return true;
}
} // namespace ModNs

namespace {
constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;

void ExpectTiling(
    const gert::TilingContextPara& tilingContextPara, uint64_t expectTilingKey,
    const ModNs::ModTilingData& expectTilingData)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(reinterpret_cast<std::vector<size_t>&>(tilingInfo.workspaceSizes), std::vector<size_t>({WORKSPACE_SIZE}));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    EXPECT_EQ(*reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get()), expectTilingData);
}
} // namespace

class ModTiling : public testing::Test {};

TEST_F(ModTiling, same_shape_float)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    ModNs::ModTilingData expect{};
    expect.usableUbSize = 3968;
    expect.needCoreNum = 8;
    expect.totalDataCount = 8192;
    expect.perCoreDataCount = 1024;
    expect.tailDataCoreNum = 0;
    expect.lastCoreDataCount = 1024;
    expect.isInput2Scalar = false;
    expect.isInput2SameShape = true;
    expect.dimNum = 1;
    expect.input1Shape[0] = 8192;
    expect.input2Shape[0] = 8192;
    expect.input2Stride[0] = 1;
    for (uint32_t i = 1; i < 8; ++i) {
        expect.input1Shape[i] = 1;
        expect.input2Shape[i] = 1;
        expect.input2Stride[i] = 0;
    }

    ExpectTiling(tilingContextPara, 1973790, expect);
}

TEST_F(ModTiling, broadcast_stride_float16)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1315860);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_FALSE(tilingData->isInput2Scalar);
    EXPECT_FALSE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->dimNum, 3U);
    EXPECT_EQ(tilingData->input2Stride[0], 0U);
    EXPECT_EQ(tilingData->input2Stride[1], 1U);
    EXPECT_EQ(tilingData->input2Stride[2], 0U);
}

TEST_F(ModTiling, scalar_int32)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 2631720);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(tilingData->isInput2Scalar);
    EXPECT_FALSE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->totalDataCount, 128U);
    EXPECT_EQ(tilingData->needCoreNum, 1U);
}
