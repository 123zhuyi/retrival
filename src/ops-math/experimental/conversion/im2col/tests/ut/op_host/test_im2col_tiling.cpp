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
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

#include "../../../op_kernel/im2col_tiling_data.h"

namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr size_t EXPECTED_WORKSPACE_COUNT = 1U;
constexpr uint64_t TEST_CORE_NUM = 40;
constexpr uint64_t TEST_UB_SIZE = 262144;
struct Im2colCompileInfo {};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara BuildCase(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& outputShape,
                                  ge::DataType dtype, const std::vector<int64_t>& kernel,
                                  const std::vector<int64_t>& strides, const std::vector<int64_t>& dilations,
                                  const std::vector<int64_t>& pads, const std::string& paddingMode = "CALCULATED")
{
    const gert::StorageShape inputStorageShape = MakeStorageShape(inputShape);
    const gert::StorageShape outputStorageShape = MakeStorageShape(outputShape);
    const std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {inputStorageShape, dtype, ge::FORMAT_NCHW},
    };
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputStorageShape, dtype, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(kernel)},
        {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
        {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
        {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>(paddingMode)},
        {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
    };
    static Im2colCompileInfo compileInfo;
    return gert::TilingContextPara("Im2col", inputs, outputs, attrs, &compileInfo, TEST_CORE_NUM, TEST_UB_SIZE,
                                   sizeof(Im2colTilingData));
}

const Im2colTilingData* GetTilingData(const TilingInfo& tilingInfo)
{
    return reinterpret_cast<const Im2colTilingData*>(tilingInfo.tilingData.get());
}

class Im2colTilingTest : public testing::Test {};

TEST_F(Im2colTilingTest, selects_channel_identity)
{
    auto context = BuildCase({1, 2, 4, 4}, {1, 2, 16}, ge::DT_FLOAT16, {1, 1}, {1, 1}, {1, 1}, {0, 0, 0, 0});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Im2colTilingHeader));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->fastChannel, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->channelIdentity, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->totalChannels, 2);
    EXPECT_EQ(data->totalOutputElements, 32);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), EXPECTED_WORKSPACE_COUNT);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
}

TEST_F(Im2colTilingTest, selects_bool_channel_template)
{
    auto context = BuildCase({1, 4, 8, 8}, {1, 36, 64}, ge::DT_BOOL, {3, 3}, {1, 1}, {1, 1}, {1, 1, 1, 1});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->fastChannel, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->channelIdentity, IM2COL_TILING_FLAG_DISABLED);
    EXPECT_EQ(data->channelFlatGather, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->channelIndexTemplateValid, IM2COL_CHANNEL_INDEX_TEMPLATE_UINT32);
    EXPECT_GT(data->channelIndexTemplateElements, 0U);
    EXPECT_GT(tilingInfo.tilingDataSize, sizeof(Im2colTilingHeader));
}

TEST_F(Im2colTilingTest, aligns_int16_channel_template_copy)
{
    auto context = BuildCase({1, 2, 16, 16}, {1, 16, 5}, ge::DT_FLOAT, {1, 8}, {16, 2}, {1, 1}, {0, 0, 0, 0});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->fastChannel, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->channelIndexTemplateValid, IM2COL_CHANNEL_INDEX_TEMPLATE_INT16);
    EXPECT_EQ(data->outputChannelStrideElements, 40);
    EXPECT_EQ(data->channelIndexTemplateElements, 48U);
    EXPECT_EQ(data->channelIndexTemplateElements * sizeof(uint16_t) % DATA_BLOCK_BYTES, 0U);
}

TEST_F(Im2colTilingTest, selects_group_batch)
{
    auto context = BuildCase({1, 1, 32, 32}, {1, 9, 1024}, ge::DT_FLOAT, {3, 3}, {1, 1}, {1, 1}, {1, 1, 1, 1});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->fastChannel, IM2COL_TILING_FLAG_DISABLED);
    EXPECT_EQ(data->fastGroup, IM2COL_TILING_FLAG_ENABLED);
    EXPECT_EQ(data->totalGroups, 9);
    EXPECT_GE(data->batchRows, 1);
}

TEST_F(Im2colTilingTest, falls_back_to_row_tiles)
{
    auto context = BuildCase({1, 1, 2, 100000}, {1, 4, 99999}, ge::DT_FLOAT, {2, 2}, {1, 1}, {1, 1}, {0, 0, 0, 0});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->fastChannel, IM2COL_TILING_FLAG_DISABLED);
    EXPECT_EQ(data->fastGroup, IM2COL_TILING_FLAG_DISABLED);
    EXPECT_GT(data->tileElements, 0);
    EXPECT_LE(data->tileElements, data->outW);
}

TEST_F(Im2colTilingTest, rejects_unsupported_dtype)
{
    auto context = BuildCase({1, 1, 4, 4}, {1, 1, 16}, ge::DT_INT32, {1, 1}, {1, 1}, {1, 1}, {0, 0, 0, 0});
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST_F(Im2colTilingTest, rejects_negative_padding)
{
    auto context = BuildCase({1, 1, 4, 4}, {1, 1, 16}, ge::DT_FLOAT16, {1, 1}, {1, 1}, {1, 1}, {-1, 0, 0, 0});
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST_F(Im2colTilingTest, rejects_shape_product_overflow)
{
    auto context = BuildCase({1, 3037000500LL, 3037000500LL, 1}, {1, 1, 1}, ge::DT_FLOAT16, {1, 1}, {1, 1}, {1, 1},
                             {0, 0, 0, 0});
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

} // namespace
