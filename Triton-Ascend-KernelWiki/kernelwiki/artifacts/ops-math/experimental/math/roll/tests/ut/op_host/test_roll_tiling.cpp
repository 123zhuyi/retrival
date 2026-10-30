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
#include "../../../op_kernel/roll_tiling_data.h"

namespace {
struct RollCompileInfoForTest {
    int32_t coreNum = 64;
};
} // namespace

class RollTiling : public testing::Test {};

TEST_F(RollTiling, basic_last_dim_roll)
{
    RollCompileInfoForTest compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
            gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
        },
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_GT(tilingInfo.blockNum, 0U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    ASSERT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(RollTilingData));

    auto* data = reinterpret_cast<const RollTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalNum, 6);
    EXPECT_EQ(data->dimNum, 2);
    EXPECT_EQ(data->activeDimCount, 1);
    EXPECT_EQ(data->activeDim, 1);
    EXPECT_EQ(data->dimSize, 3);
    EXPECT_EQ(data->innerSize, 1);
    EXPECT_EQ(data->activeShift, 1);
}

TEST_F(RollTiling, flatten_roll_when_dims_empty)
{
    RollCompileInfoForTest compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5})),
            gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})),
        },
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* data = reinterpret_cast<const RollTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->dimNum, 1);
    EXPECT_EQ(data->totalNum, 24);
    EXPECT_EQ(data->shapes[0], 24);
    EXPECT_EQ(data->strides[0], 1);
    EXPECT_EQ(data->shifts[0], 5);
}
