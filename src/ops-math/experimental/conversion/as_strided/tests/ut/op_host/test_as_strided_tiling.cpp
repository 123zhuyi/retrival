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
 * \file test_as_strided_tiling.cpp
 * \brief AsStrided tiling UT.
 */

#include <gtest/gtest.h>
#include <vector>

#include "../../../op_kernel/as_strided_tiling_data.h"
#include "../../../op_kernel/as_strided_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t TEST_CORE_NUM = 64;
constexpr uint64_t TEST_UB_SIZE = 262144;
constexpr uint64_t TEST_TILING_DATA_SIZE = 4096;

struct AsStridedTestCompileInfo {
    uint32_t ubSizePlatform;
    uint32_t maxCoreNum;
};

AsStridedTestCompileInfo g_compileInfo = {
    static_cast<uint32_t>(TEST_UB_SIZE),
    static_cast<uint32_t>(TEST_CORE_NUM),
};

const AsStridedTilingData* GetTilingData(const TilingInfo& tilingInfo)
{
    return reinterpret_cast<const AsStridedTilingData*>(tilingInfo.tilingData.get());
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape storageShape;
    for (auto dim : dims) {
        storageShape.MutableOriginShape().AppendDim(dim);
        storageShape.MutableStorageShape().AppendDim(dim);
    }
    return storageShape;
}

gert::TilingContextPara BuildTilingCase(const gert::StorageShape& xShape, ge::DataType xDtype,
                                        const gert::StorageShape& yShape, int64_t rank, ge::DataType indexDtype,
                                        void* sizeValue, void* strideValue, void* storageOffsetValue)
{
    return gert::TilingContextPara("AsStrided",
                                   {
                                       {xShape, xDtype, ge::FORMAT_ND},
                                       {{{rank}, {rank}}, indexDtype, ge::FORMAT_ND, true, sizeValue},
                                       {{{rank}, {rank}}, indexDtype, ge::FORMAT_ND, true, strideValue},
                                       {{{1}, {1}}, indexDtype, ge::FORMAT_ND, true, storageOffsetValue},
                                   },
                                   {
                                       {yShape, xDtype, ge::FORMAT_ND},
                                   },
                                   &g_compileInfo, TEST_CORE_NUM, TEST_UB_SIZE, TEST_TILING_DATA_SIZE);
}
} // namespace

class AsStridedTiling : public testing::Test {};

TEST_F(AsStridedTiling, tiling_contiguous_path)
{
    std::vector<int64_t> sizeValue = {16};
    std::vector<int64_t> strideValue = {1};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({64}), ge::DT_INT32, MakeStorageShape({16}), 1,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(AsStridedTilingData));

    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_CONTIGUOUS);
    EXPECT_EQ(data->totalOutputElements, 16);
    EXPECT_EQ(data->inputElementCount, 64);
    EXPECT_EQ(data->storageOffset, 0);
    EXPECT_EQ(data->outputDimNum, 1);
    EXPECT_EQ(data->lastDimSize, 16);
    EXPECT_EQ(data->lastDimStride, 1);
    EXPECT_EQ(data->inputSpanElements, 16);
    EXPECT_EQ(data->outSize[0], 16);
    EXPECT_EQ(data->outStride[0], 1);
}

TEST_F(AsStridedTiling, tiling_small_aligned_contiguous_path)
{
    std::vector<int64_t> sizeValue = {256};
    std::vector<int64_t> strideValue = {1};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({256}), ge::DT_INT32, MakeStorageShape({256}), 1,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED);
    EXPECT_EQ(data->totalOutputElements, 256);
    EXPECT_EQ(data->blockElements, 8);
}

TEST_F(AsStridedTiling, tiling_stride1_row_batch_path)
{
    std::vector<int64_t> sizeValue = {16, 4};
    std::vector<int64_t> strideValue = {8, 1};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({256}), ge::DT_FLOAT16, MakeStorageShape({16, 4}), 2,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH);
    EXPECT_EQ(data->axis0Elements, 16);
    EXPECT_EQ(data->lastDimSize, 4);
    EXPECT_EQ(data->lastDimStride, 1);
}

TEST_F(AsStridedTiling, tiling_general_small_span_path)
{
    std::vector<int64_t> sizeValue = {16, 4};
    std::vector<int64_t> strideValue = {16, 2};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({256}), ge::DT_FLOAT16, MakeStorageShape({16, 4}), 2,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN);
    EXPECT_EQ(data->lastDimStride, 2);
    EXPECT_EQ(data->inputSpanElements, 247);
}

TEST_F(AsStridedTiling, tiling_broadcast_path)
{
    std::vector<int64_t> sizeValue = {8, 8};
    std::vector<int64_t> strideValue = {8, 0};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({64}), ge::DT_INT32, MakeStorageShape({8, 8}), 2,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_BROADCAST);
    EXPECT_EQ(data->lastDimStride, 0);
    EXPECT_EQ(data->axis0Elements, 8);
}

TEST_F(AsStridedTiling, tiling_rank1_stride_span_path)
{
    std::vector<int64_t> sizeValue = {128};
    std::vector<int64_t> strideValue = {2};
    std::vector<int64_t> storageOffsetValue = {1};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({256}), ge::DT_FLOAT16, MakeStorageShape({128}), 1,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN);
    EXPECT_EQ(data->storageOffset, 1);
    EXPECT_EQ(data->inputSpanElements, 255);
}

TEST_F(AsStridedTiling, tiling_rank1_stride_path)
{
    std::vector<int64_t> sizeValue = {1000};
    std::vector<int64_t> strideValue = {20};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({20000}), ge::DT_FLOAT16, MakeStorageShape({1000}), 1,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_RANK1_STRIDE);
    EXPECT_EQ(data->lastDimStride, 20);
    EXPECT_EQ(data->inputSpanElements, 19981);
}

TEST_F(AsStridedTiling, tiling_compact_suffix_path)
{
    std::vector<int64_t> sizeValue = {64, 32};
    std::vector<int64_t> strideValue = {64, 1};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({4096}), ge::DT_FLOAT16, MakeStorageShape({64, 32}), 2,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_COMPACT_SUFFIX);
    EXPECT_EQ(data->totalOutputElements, 2048);
    EXPECT_EQ(data->inputSpanElements, 4064);
    EXPECT_EQ(data->suffixStartDim, 0);
    EXPECT_GT(data->suffixElements, 0);
    EXPECT_GT(data->suffixOuterElements, 1);
}

TEST_F(AsStridedTiling, tiling_empty_output_uses_scalar_path)
{
    std::vector<int64_t> sizeValue = {0, 4};
    std::vector<int64_t> strideValue = {4, 1};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({16}), ge::DT_FLOAT16, MakeStorageShape({0, 4}), 2,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    const auto* data = GetTilingData(tilingInfo);
    EXPECT_EQ(data->tilingKey, AS_STRIDED_TILING_KEY_SCALAR);
    EXPECT_EQ(data->totalOutputElements, 0);
    EXPECT_EQ(data->usedCoreNum, 1);
    EXPECT_EQ(data->perCoreElements, 0);
}

TEST_F(AsStridedTiling, tiling_rejects_rank_mismatch)
{
    std::vector<int64_t> sizeValue = {4, 4};
    std::vector<int64_t> strideValue = {1};
    std::vector<int64_t> storageOffsetValue = {0};

    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, storageOffsetValue.data()},
        },
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &g_compileInfo, TEST_CORE_NUM, TEST_UB_SIZE, TEST_TILING_DATA_SIZE);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(AsStridedTiling, tiling_rejects_out_of_range_view)
{
    std::vector<int64_t> sizeValue = {4};
    std::vector<int64_t> strideValue = {4};
    std::vector<int64_t> storageOffsetValue = {0};

    auto tilingContextPara = BuildTilingCase(MakeStorageShape({8}), ge::DT_INT32, MakeStorageShape({4}), 1,
                                             ge::DT_INT64, sizeValue.data(), strideValue.data(),
                                             storageOffsetValue.data());

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
