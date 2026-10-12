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
 * \file test_square_sum_all_tiling.cpp
 * \brief SquareSumAll arch35 tiling tests.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/square_sum_all_tiling_data.h"

namespace {
constexpr uint64_t TEST_CORE_NUM = 56;
constexpr uint64_t TEST_UB_SIZE = 256 * 1024;
constexpr uint64_t TEST_TILING_CAPACITY = 4096;
constexpr size_t TEST_SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024;
constexpr int64_t TILE_ELEMENTS = 4096;
constexpr int64_t WORKSPACE_BYTES_PER_CORE = 64;
constexpr int64_t GPU_ALIGNED_MIN_ELEMENTS = 8192;
constexpr int64_t GPU_ALIGNED_MAX_ELEMENTS_FOR_VECTOR_CHUNKS = 14495293440;
constexpr uint64_t GPU_ALIGNED_MIN_UB_BYTES = 137728;
constexpr size_t GPU_ALIGNED_MIN_WORKSPACE_BYTES = 4096;
constexpr size_t GPU_ALIGNED_MAX_WORKSPACE_BYTES = 110592;
constexpr uint64_t LEGACY_TILING_KEY = 0;
constexpr uint64_t GPU_ALIGNED_TILING_KEY = 1;

struct SquareSumAllCompileInfo {};
SquareSumAllCompileInfo g_compileInfo;

struct ContextOptions {
    ge::DataType x1Dtype = ge::DT_FLOAT;
    ge::DataType x2Dtype = ge::DT_FLOAT;
    ge::DataType y1Dtype = ge::DT_FLOAT;
    ge::DataType y2Dtype = ge::DT_FLOAT;
    ge::Format x1Format = ge::FORMAT_ND;
    ge::Format x2Format = ge::FORMAT_ND;
    ge::Format y1Format = ge::FORMAT_ND;
    ge::Format y2Format = ge::FORMAT_ND;
    std::vector<int64_t> y1Shape;
    std::vector<int64_t> y2Shape;
    uint64_t ubSize = TEST_UB_SIZE;
    uint64_t tilingCapacity = TEST_TILING_CAPACITY;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape().SetDimNum(dimensions.size());
    storageShape.MutableStorageShape().SetDimNum(dimensions.size());
    for (size_t index = 0; index < dimensions.size(); ++index) {
        storageShape.MutableOriginShape().SetDim(index, dimensions[index]);
        storageShape.MutableStorageShape().SetDim(index, dimensions[index]);
    }
    return storageShape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& x1Shape, const std::vector<int64_t>& x2Shape,
                                    const ContextOptions& options = {})
{
    return gert::TilingContextPara("SquareSumAll",
                                   {
                                       {MakeStorageShape(x1Shape), options.x1Dtype, options.x1Format},
                                       {MakeStorageShape(x2Shape), options.x2Dtype, options.x2Format},
                                   },
                                   {
                                       {MakeStorageShape(options.y1Shape), options.y1Dtype, options.y1Format},
                                       {MakeStorageShape(options.y2Shape), options.y2Dtype, options.y2Format},
                                   },
                                   &g_compileInfo, TEST_CORE_NUM, options.ubSize, options.tilingCapacity);
}

const SquareSumAllTilingData& GetTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_GE(tilingInfo.tilingDataSize, sizeof(SquareSumAllTilingData));
    return *reinterpret_cast<const SquareSumAllTilingData*>(tilingInfo.tilingData.get());
}

void CheckCommonResult(const TilingInfo& tilingInfo, int64_t expectedCores,
                       uint64_t expectedTilingKey = LEGACY_TILING_KEY)
{
    EXPECT_EQ(tilingInfo.tilingKey, expectedTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(expectedCores));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_GE(tilingInfo.workspaceSizes[0], expectedCores * WORKSPACE_BYTES_PER_CORE);
}
} // namespace

class SquareSumAllTilingTest : public testing::Test {};

TEST_F(SquareSumAllTilingTest, SingleElementUsesOneCore)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({1}, {1}), tilingInfo));
    CheckCommonResult(tilingInfo, 1);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalElements, 1);
    EXPECT_EQ(tilingData.usedCoreNum, 1);
    EXPECT_EQ(tilingData.baseCoreElements, 1);
    EXPECT_EQ(tilingData.extraCoreCount, 0);
    EXPECT_EQ(tilingData.tileElements, TILE_ELEMENTS);
}

TEST_F(SquareSumAllTilingTest, PrimeLengthAvoidsOverPartitioning)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({4099}, {4099}), tilingInfo));
    CheckCommonResult(tilingInfo, 1);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalElements, 4099);
    EXPECT_EQ(tilingData.usedCoreNum, 1);
    EXPECT_EQ(tilingData.baseCoreElements, 4099);
    EXPECT_EQ(tilingData.extraCoreCount, 0);
    EXPECT_EQ(tilingData.tileElements, TILE_ELEMENTS);
}

TEST_F(SquareSumAllTilingTest, MultiTileLengthKeepsBalancedPartition)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({262147}, {262147}), tilingInfo));
    CheckCommonResult(tilingInfo, 56, GPU_ALIGNED_TILING_KEY);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalElements, 262147);
    EXPECT_EQ(tilingData.usedCoreNum, 56);
    EXPECT_EQ(tilingData.baseCoreElements, 4681);
    EXPECT_EQ(tilingData.extraCoreCount, 11);
    EXPECT_EQ(tilingData.tileElements, TILE_ELEMENTS);
}

TEST_F(SquareSumAllTilingTest, BelowGpuAlignedThresholdUsesLegacyKernel)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({GPU_ALIGNED_MIN_ELEMENTS - 1}, {GPU_ALIGNED_MIN_ELEMENTS - 1}), tilingInfo));
    CheckCommonResult(tilingInfo, 1, LEGACY_TILING_KEY);
}

TEST_F(SquareSumAllTilingTest, GpuAlignedThresholdUsesTwoCores)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({GPU_ALIGNED_MIN_ELEMENTS}, {GPU_ALIGNED_MIN_ELEMENTS}), tilingInfo));
    CheckCommonResult(tilingInfo, 2, GPU_ALIGNED_TILING_KEY);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], TEST_SYSTEM_WORKSPACE_BYTES + GPU_ALIGNED_MIN_WORKSPACE_BYTES);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.baseCoreElements, 4096);
    EXPECT_EQ(tilingData.extraCoreCount, 0);
}

TEST_F(SquareSumAllTilingTest, GpuAlignedPseudoBlockCapBoundaryStaysOnKeyOne)
{
    for (const int64_t elementCount : {221184, 221185}) {
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(MakeContext({elementCount}, {elementCount}), tilingInfo));
        CheckCommonResult(tilingInfo, 54, GPU_ALIGNED_TILING_KEY);
        EXPECT_EQ(tilingInfo.workspaceSizes[0], TEST_SYSTEM_WORKSPACE_BYTES + GPU_ALIGNED_MAX_WORKSPACE_BYTES);
    }
}

TEST_F(SquareSumAllTilingTest, GpuAlignedPathFitsExactMinimumUb)
{
    ContextOptions options;
    options.ubSize = GPU_ALIGNED_MIN_UB_BYTES;
    TilingInfo tilingInfo;
    ASSERT_TRUE(
        ExecuteTiling(MakeContext({GPU_ALIGNED_MIN_ELEMENTS}, {GPU_ALIGNED_MIN_ELEMENTS}, options), tilingInfo));
    CheckCommonResult(tilingInfo, 2, GPU_ALIGNED_TILING_KEY);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], TEST_SYSTEM_WORKSPACE_BYTES + GPU_ALIGNED_MIN_WORKSPACE_BYTES);
}

TEST_F(SquareSumAllTilingTest, GpuAlignedVectorChunkBoundaryFallsBackSafely)
{
    TilingInfo fitInfo;
    ASSERT_TRUE(ExecuteTiling(
        MakeContext({GPU_ALIGNED_MAX_ELEMENTS_FOR_VECTOR_CHUNKS}, {GPU_ALIGNED_MAX_ELEMENTS_FOR_VECTOR_CHUNKS}),
        fitInfo));
    CheckCommonResult(fitInfo, 56, GPU_ALIGNED_TILING_KEY);

    TilingInfo fallbackInfo;
    ASSERT_TRUE(ExecuteTiling(
        MakeContext({GPU_ALIGNED_MAX_ELEMENTS_FOR_VECTOR_CHUNKS + 1}, {GPU_ALIGNED_MAX_ELEMENTS_FOR_VECTOR_CHUNKS + 1}),
        fallbackInfo));
    CheckCommonResult(fallbackInfo, 56, LEGACY_TILING_KEY);
}

TEST_F(SquareSumAllTilingTest, GpuAlignedPathFallsBackWhenUbIsOneByteShort)
{
    ContextOptions options;
    options.ubSize = GPU_ALIGNED_MIN_UB_BYTES - 1;
    TilingInfo tilingInfo;
    ASSERT_TRUE(
        ExecuteTiling(MakeContext({GPU_ALIGNED_MIN_ELEMENTS}, {GPU_ALIGNED_MIN_ELEMENTS}, options), tilingInfo));
    CheckCommonResult(tilingInfo, 2, LEGACY_TILING_KEY);
}

TEST_F(SquareSumAllTilingTest, Int64MaxElementCountFallsBackWithoutOverflow)
{
    const int64_t elementCount = std::numeric_limits<int64_t>::max();
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({elementCount}, {elementCount}), tilingInfo));
    CheckCommonResult(tilingInfo, 56, LEGACY_TILING_KEY);
    EXPECT_EQ(GetTilingData(tilingInfo).totalElements, elementCount);
}

TEST_F(SquareSumAllTilingTest, RankEightIsFlattenedToElementCount)
{
    const std::vector<int64_t> shape = {2, 2, 2, 2, 2, 2, 2, 3};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext(shape, shape), tilingInfo));
    CheckCommonResult(tilingInfo, 1);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalElements, 384);
    EXPECT_EQ(tilingData.usedCoreNum, 1);
    EXPECT_EQ(tilingData.baseCoreElements, 384);
    EXPECT_EQ(tilingData.extraCoreCount, 0);
}

// rank-0 标量输入 = 1 个元素，与 canndev 对齐（见 infershape UT 同名用例的注释）。
TEST_F(SquareSumAllTilingTest, AcceptsRankZeroScalarInput)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({}, {}), tilingInfo));
    CheckCommonResult(tilingInfo, 1);

    const auto& tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalElements, 1);
    EXPECT_EQ(tilingData.usedCoreNum, 1);
    EXPECT_EQ(tilingData.baseCoreElements, 1);
    EXPECT_EQ(tilingData.extraCoreCount, 0);
}

TEST_F(SquareSumAllTilingTest, RejectsRankNine)
{
    const std::vector<int64_t> shape(9, 1);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext(shape, shape), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsEmptyTensor)
{
    const std::vector<std::vector<int64_t>> emptyShapes = {
        {0}, {0, 2, 3}, {2, 0, 3}, {2, 3, 0}, {0, 2, 0}, {0, 0},
    };
    for (const auto& shape : emptyShapes) {
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(MakeContext(shape, shape), tilingInfo));
    }
}

TEST_F(SquareSumAllTilingTest, RejectsDifferentShapes)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({2, 3}, {3, 2}), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsUnsupportedDtype)
{
    ContextOptions options;
    options.x1Dtype = ge::DT_FLOAT16;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsMixedInputDtypes)
{
    ContextOptions options;
    options.x2Dtype = ge::DT_FLOAT16;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsUnsupportedOutputDtype)
{
    ContextOptions options;
    options.y2Dtype = ge::DT_FLOAT16;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, AcceptsNdFormatTuple)
{
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeContext({64}, {64}), tilingInfo));
}

// 私有格式未在 Ascend 950 OpDef 中注册；Host Tiling 仍需防御异常描述绕过注册层。
TEST_F(SquareSumAllTilingTest, RejectsUnsupportedPrivateFormatTuples)
{
    for (const ge::Format format : {ge::FORMAT_FRACTAL_Z, ge::FORMAT_C1HWNCoC0, ge::FORMAT_NC1HWC0}) {
        ContextOptions options;
        options.x1Format = format;
        options.x2Format = format;
        options.y1Format = format;
        options.y2Format = format;
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo))
            << "format " << static_cast<int32_t>(format) << " should be rejected";
    }
}

TEST_F(SquareSumAllTilingTest, AcceptsNchwInputsWithNdOutputs)
{
    ContextOptions options;
    options.x1Format = ge::FORMAT_NCHW;
    options.x2Format = ge::FORMAT_NCHW;
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({2, 3, 4, 5}, {2, 3, 4, 5}, options), tilingInfo));
    EXPECT_EQ(GetTilingData(tilingInfo).totalElements, 120);
}

TEST_F(SquareSumAllTilingTest, AcceptsNhwcInputsWithNdOutputs)
{
    ContextOptions options;
    options.x1Format = ge::FORMAT_NHWC;
    options.x2Format = ge::FORMAT_NHWC;
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({2, 4, 5, 3}, {2, 4, 5, 3}, options), tilingInfo));
    EXPECT_EQ(GetTilingData(tilingInfo).totalElements, 120);
}

TEST_F(SquareSumAllTilingTest, RejectsPublicFormatInputWithNonFourDimensionalShape)
{
    for (const ge::Format format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC}) {
        ContextOptions options;
        options.x1Format = format;
        options.x2Format = format;
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(MakeContext({2, 3, 4}, {2, 3, 4}, options), tilingInfo));
    }
}

TEST_F(SquareSumAllTilingTest, RejectsMixedPublicInputFormats)
{
    ContextOptions options;
    options.x1Format = ge::FORMAT_NCHW;
    options.x2Format = ge::FORMAT_NHWC;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({2, 3, 4, 5}, {2, 3, 4, 5}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsPublicFormatOutputs)
{
    ContextOptions options;
    options.x1Format = ge::FORMAT_NCHW;
    options.x2Format = ge::FORMAT_NCHW;
    options.y1Format = ge::FORMAT_NCHW;
    options.y2Format = ge::FORMAT_NCHW;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({2, 3, 4, 5}, {2, 3, 4, 5}, options), tilingInfo));
}

// 未登记的 format 仍应拒绝，避免校验被放开成"什么都收"。
TEST_F(SquareSumAllTilingTest, RejectsUnregisteredFormat)
{
    ContextOptions options;
    options.x1Format = ge::FORMAT_NCDHW;
    options.x2Format = ge::FORMAT_NCDHW;
    options.y1Format = ge::FORMAT_NCDHW;
    options.y2Format = ge::FORMAT_NCDHW;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsNonScalarOutput)
{
    ContextOptions options;
    options.y1Shape = {2};
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, AcceptsPhysicalLengthOneOutputs)
{
    ContextOptions options;
    options.y1Shape = {1};
    options.y2Shape = {1};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsElementCountOverflow)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({3037000500, 3037000500}, {3037000500, 3037000500}), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsInsufficientTilingCapacity)
{
    ContextOptions options;
    options.tilingCapacity = sizeof(SquareSumAllTilingData) - 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsInsufficientUb)
{
    ContextOptions options;
    options.ubSize = 8 * 1024;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}

TEST_F(SquareSumAllTilingTest, RejectsUbThatCannotHoldOneVector)
{
    ContextOptions options;
    options.ubSize = 9 * 1024;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(MakeContext({64}, {64}, options), tilingInfo));
}
