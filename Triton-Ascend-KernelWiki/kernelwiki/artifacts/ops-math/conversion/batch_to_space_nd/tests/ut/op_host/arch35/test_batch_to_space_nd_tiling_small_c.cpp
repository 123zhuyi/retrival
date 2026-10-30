/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/batch_to_space_nd/op_kernel/arch35/batch_to_space_nd_tiling_data.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using ::testing::Contains;
using ::testing::ElementsAreArray;
using ::testing::Not;

namespace {
B2SNDCompileInfo compileInfo;
}

class BatchToSpaceNDTilingSmallCTest : public testing::Test {
    static constexpr uint32_t SMALL_C_RESERVE_BUFFER_SIZE = 256U;

protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceNDTilingSmallCTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceNDTilingSmallCTest TearDown" << std::endl; }

    template <typename T>
    inline T CeilAlignBlockSize(T size)
    {
        return (size + 31) / 32 * 32;
    }

    std::vector<uint64_t> Shape2Vec(uint64_t shape[], size_t size)
    {
        return std::vector<uint64_t>(shape, shape + size);
    }

    std::array<size_t, MAX_EXPAND_RANK> ComputeY2X(uint32_t rank, uint32_t blockShapeDimNum)
    {
        std::array<size_t, MAX_EXPAND_RANK> yAxisPerm{};
        // batch
        yAxisPerm[0] = blockShapeDimNum;
        // remain
        yAxisPerm[rank - 1] = rank - 1;
        for (size_t i = 0; i < blockShapeDimNum; ++i) {
            // block shape
            yAxisPerm[2 + i * 2] = i;
            // space
            yAxisPerm[1 + i * 2] = blockShapeDimNum + 1 + i;
        }
        return yAxisPerm;
    }

    std::array<size_t, MAX_EXPAND_RANK> ComputeX2Y(uint32_t rank, const std::array<size_t, MAX_EXPAND_RANK>& y2x)
    {
        std::array<size_t, MAX_EXPAND_RANK> x2y{};
        for (size_t i = 0; i < rank; ++i) {
            x2y[y2x[i]] = i;
        }
        return x2y;
    }

    uint64_t ComputeNeedBuffSize(int dSize, B2SNDSmallCTilingData* data, const std::set<size_t>& inner,
                                 const std::set<size_t>& outter, uint32_t cutAxisFactor, uint32_t otherSideFactor,
                                 bool needAlignOnSomeAxis = false)
    {
        // 计算内轴，要Block对齐
        uint64_t needBufSize = dSize;
        bool needAlign = needAlignOnSomeAxis;
        int16_t compactCnt = 1;
        for (auto it = inner.rbegin(); it != inner.rend(); ++it) {
            needBufSize *= data->croppedInShape[*it];
            // 轴被crop，需要对齐
            if (needAlign && data->croppedInShape[*it] != data->oriInShape[*it]) {
                if (compactCnt > 0) {
                    compactCnt--;
                    continue;
                }
                needBufSize = CeilAlignBlockSize(needBufSize);
                // 对齐一次即可
                needAlign = false;
            }
        }
        needBufSize *= cutAxisFactor;
        needBufSize = CeilAlignBlockSize(needBufSize);
        // UB外轴
        needBufSize *= (data->inUbAxis == data->outUbAxis) ? 1 : otherSideFactor;
        for (auto i : outter) {
            needBufSize *= data->croppedInShape[i];
        }
        return needBufSize;
    }

    void CheckBuffSize(B2SNDSmallCTilingData* data, uint32_t blockShapeDimNum, ge::DataType dataType)
    {
        uint32_t rank = blockShapeDimNum + blockShapeDimNum + 2;

        // 坐标转换关系
        auto yAxisPerm = ComputeY2X(rank, blockShapeDimNum);
        auto xAxisToYAxis = ComputeX2Y(rank, yAxisPerm);

        // 计算内外轴
        std::set<size_t> xInner{};
        std::set<size_t> xOutter{};
        std::set<size_t> yInner{};
        std::set<size_t> yOutter{};
        for (size_t i = data->inUbAxis; i < rank; ++i) {
            xInner.insert(i);
        }
        for (size_t i = xAxisToYAxis[data->outUbAxis]; i < rank; ++i) {
            yInner.insert(yAxisPerm[i]);
        }
        std::set_difference(yInner.begin(), yInner.end(), xInner.begin(), xInner.end(),
                            std::inserter(xOutter, xOutter.begin()));
        std::set_difference(xInner.begin(), xInner.end(), yInner.begin(), yInner.end(),
                            std::inserter(yOutter, yOutter.begin()));

        // 切同一根轴，则切的大小要相等
        if (data->inUbAxis == data->outUbAxis) {
            EXPECT_EQ(data->inUbFactor, data->outUbFactor);
        } else {
            // 检查切分轴不是另一侧的内轴
            // 切分轴是0轴，且满切的情况除外
            if (!(xAxisToYAxis[data->outUbAxis] == 0 && data->outUbFactor == data->croppedInShape[data->outUbAxis])) {
                EXPECT_THAT(xInner, Not(Contains(data->outUbAxis)));
            }
            if (!(data->inUbAxis == 0 && data->inUbFactor == data->croppedInShape[data->inUbAxis])) {
                EXPECT_THAT(yInner, Not(Contains(data->inUbAxis)));
            }
        }

        // 切分轴单独处理，剔除
        xInner.erase(data->inUbAxis);
        xInner.erase(data->outUbAxis);
        xOutter.erase(data->inUbAxis);
        xOutter.erase(data->outUbAxis);
        yInner.erase(data->inUbAxis);
        yInner.erase(data->outUbAxis);
        yOutter.erase(data->inUbAxis);
        yOutter.erase(data->outUbAxis);

        int dSize = ge::GetSizeByDataType(dataType);
        // 计算输入需要bufsize
        uint64_t xNeedBufSize = ComputeNeedBuffSize(dSize, data, xInner, xOutter, data->inUbFactor, data->outUbFactor,
                                                    true);
        EXPECT_GE(data->ubTileSize - SMALL_C_RESERVE_BUFFER_SIZE, xNeedBufSize);
        // 计算输出需要bufsize
        uint64_t yNeedBufSize = ComputeNeedBuffSize(dSize, data, yInner, yOutter, data->outUbFactor, data->inUbFactor);
        EXPECT_GE(data->ubTileSize - SMALL_C_RESERVE_BUFFER_SIZE, yNeedBufSize);
    }

    void TestCheckBuffer(ge::DataType dataType, const std::initializer_list<int64_t>& xShape,
                         const std::vector<int32_t>& blockShapeValues, const std::vector<int32_t>& cropsValues)
    {
        int64_t bsDimNum = blockShapeValues.size();
        gert::TilingContextPara tilingContextPara(
            "BatchToSpaceND",
            {
                {{xShape, xShape}, dataType, ge::FORMAT_ND},
                {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, (void*)blockShapeValues.data()},
                {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, (void*)cropsValues.data()},
            },
            {
                {{{}, {}}, dataType, ge::FORMAT_ND},
            },
            &compileInfo);
        TilingInfo tilingInfo;
        auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_TRUE(tilingRet);
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
        B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
        CheckBuffSize(data, blockShapeValues.size(), dataType);
    }
};

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_precrop_nocrop)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {1, 1, 1, 2};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 3, 5, 4}, {30, 3, 5, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 4, 12, 4}, {5, 4, 12, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_precrop_cropspace)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {1, 3, 4, 6};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 3, 5, 4}, {30, 3, 5, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 2, 5, 4}, {5, 2, 5, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({2, 3, 5, 2, 2, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_precrop_cropblockshape)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {1, 4, 9, 4};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 3, 5, 4}, {30, 3, 5, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 1, 2, 4}, {5, 1, 2, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({1, 2, 5, 1, 1, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_nocrop_fullload)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 3, 5, 4}, {30, 3, 5, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 6, 15, 4}, {5, 6, 15, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({2, 3, 5, 3, 5, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 1);
    EXPECT_EQ(data->inUbAxis, 0);
    EXPECT_EQ(data->outUbAxis, 2);
    EXPECT_EQ(data->inUbFactor, 2);
    EXPECT_EQ(data->outUbFactor, 5);
    EXPECT_EQ(data->ubTotalCount, 1);
    EXPECT_EQ(data->ubPerCount, 1);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_nocrop_cut_common)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 300, 5000, 4}, {30, 300, 5000, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 600, 15000, 4}, {5, 600, 15000, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 300, 5000, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({2, 3, 5, 300, 5000, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 64);
    EXPECT_EQ(data->inUbAxis, 4);
    EXPECT_EQ(data->outUbAxis, 4);
    EXPECT_EQ(data->inUbFactor, 1360);
    EXPECT_EQ(data->outUbFactor, 1360);
    EXPECT_EQ(data->ubTotalCount, 12000);
    EXPECT_EQ(data->ubPerCount, 188);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_nocrop_cut_xs0_bs0_2d)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {2, 3};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{30, 300, 300, 4}, {30, 300, 300, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{5, 600, 900, 4}, {5, 600, 900, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({2, 3, 5, 300, 300, 4}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({2, 3, 5, 300, 300, 4}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 63);
    EXPECT_EQ(data->inUbAxis, 3);
    EXPECT_EQ(data->outUbAxis, 0);
    EXPECT_EQ(data->inUbFactor, 4);
    EXPECT_EQ(data->outUbFactor, 1);
    EXPECT_EQ(data->ubTotalCount, 750);
    EXPECT_EQ(data->ubPerCount, 12);
    EXPECT_EQ(tilingInfo.blockNum, 63);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_nocrop_cut_xs0_bs0_1d)
{
    auto dataType = ge::DT_INT16;
    std::vector<int32_t> blockShapeValues = {24};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{576, 55, 115}, {576, 55, 115}}, dataType, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{24, 1320, 115}, {24, 1320, 115}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000001'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 4), ElementsAreArray({24, 24, 55, 115}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 4), ElementsAreArray({24, 24, 55, 115}));
    EXPECT_THAT(Shape2Vec(*data->crops, 2), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 64);
    EXPECT_EQ(data->inUbAxis, 2);
    EXPECT_EQ(data->outUbAxis, 0);
    EXPECT_EQ(data->inUbFactor, 17);
    EXPECT_EQ(data->outUbFactor, 16);
    EXPECT_EQ(data->ubTotalCount, 192);
    EXPECT_EQ(data->ubPerCount, 3);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_nocrop_cut_n)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {3, 5};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{855, 85, 1, 6}, {855, 85, 1, 6}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{57, 255, 5, 6}, {57, 255, 5, 6}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({3, 5, 57, 85, 1, 6}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({3, 5, 57, 85, 1, 6}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 57);
    EXPECT_EQ(data->inUbAxis, 2);
    EXPECT_EQ(data->outUbAxis, 2);
    EXPECT_EQ(data->inUbFactor, 1);
    EXPECT_EQ(data->outUbFactor, 1);
    EXPECT_EQ(data->ubTotalCount, 57);
    EXPECT_EQ(data->ubPerCount, 1);
    EXPECT_EQ(tilingInfo.blockNum, 57);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_crop_cut_xs0_bs0_2d)
{
    auto dataType = ge::DT_INT64;
    std::vector<int32_t> blockShapeValues = {25, 8};
    std::vector<int32_t> cropsValues = {731, 551, 7, 4};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{5800, 54, 68, 10}, {5800, 54, 68, 10}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{28, 68, 533, 10}, {28, 68, 533, 10}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({25, 8, 29, 54, 68, 10}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({25, 8, 29, 3, 68, 10}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 64);
    EXPECT_EQ(data->inUbAxis, 3);
    EXPECT_EQ(data->outUbAxis, 0);
    EXPECT_EQ(data->inUbFactor, 1);
    EXPECT_EQ(data->outUbFactor, 1);
    EXPECT_EQ(data->ubTotalCount, 2175);
    EXPECT_EQ(data->ubPerCount, 34);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_crop_cut_n_bs0)
{
    auto dataType = ge::DT_FLOAT;
    std::vector<int32_t> blockShapeValues = {42, 1};
    std::vector<int32_t> cropsValues = {391, 149, 1, 5};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{294, 21, 8, 22}, {294, 21, 8, 22}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{7, 342, 2, 22}, {7, 342, 2, 22}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({42, 1, 7, 21, 8, 22}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({42, 1, 7, 9, 2, 22}));
    EXPECT_THAT(Shape2Vec(*data->crops, 4), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 35);
    EXPECT_EQ(data->inUbAxis, 2);
    EXPECT_EQ(data->outUbAxis, 0);
    EXPECT_EQ(data->inUbFactor, 1);
    EXPECT_EQ(data->outUbFactor, 9);
    EXPECT_EQ(data->ubTotalCount, 35);
    EXPECT_EQ(data->ubPerCount, 1);
    EXPECT_EQ(tilingInfo.blockNum, 35);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_crop_cut_alignaxis)
{
    auto dataType = ge::DT_UINT16;
    std::vector<int32_t> blockShapeValues = {2, 1, 317};
    std::vector<int32_t> cropsValues = {0, 1, 825, 97, 105, 414};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{634, 2, 1597, 2, 3}, {634, 2, 1597, 2, 3}}, dataType, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 3, 675, 115, 3}, {1, 3, 675, 115, 3}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000011'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 8), ElementsAreArray({2, 1, 317, 1, 2, 1597, 2, 3}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 8), ElementsAreArray({2, 1, 115, 1, 2, 675, 1, 3}));
    EXPECT_THAT(Shape2Vec(*data->crops, 6), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 56);
    EXPECT_EQ(data->inUbAxis, 5);
    EXPECT_EQ(data->outUbAxis, 2);
    EXPECT_EQ(data->inUbFactor, 106);
    EXPECT_EQ(data->outUbFactor, 101);
    EXPECT_EQ(data->ubTotalCount, 56);
    EXPECT_EQ(data->ubPerCount, 1);
    EXPECT_EQ(tilingInfo.blockNum, 56);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_crop_almost_full)
{
    auto dataType = ge::DT_INT16;
    std::vector<int32_t> blockShapeValues = {1, 1, 1};
    std::vector<int32_t> cropsValues = {992, 530, 11, 0, 1, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 3310, 13, 2, 2}, {2, 3310, 13, 2, 2}}, dataType, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 1788, 2, 1}, {2, 1788, 2, 1}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000011'00000010);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(Shape2Vec(data->oriInShape, 6), ElementsAreArray({1, 1, 1, 2, 3310, 13}));
    EXPECT_THAT(Shape2Vec(data->croppedInShape, 6), ElementsAreArray({1, 1, 1, 2, 1788, 2}));
    EXPECT_THAT(Shape2Vec(*data->crops, 6), ElementsAreArray(cropsValues));
    EXPECT_EQ(data->coreNum, 16);
    EXPECT_EQ(data->inUbAxis, 4);
    EXPECT_EQ(data->outUbAxis, 4);
    EXPECT_EQ(data->inUbFactor, 255);
    EXPECT_EQ(data->outUbFactor, 255);
    EXPECT_EQ(data->ubTotalCount, 16);
    EXPECT_EQ(data->ubPerCount, 1);
    EXPECT_EQ(tilingInfo.blockNum, 16);
    CheckBuffSize(data, blockShapeValues.size(), dataType);
}

TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_check_buffer)
{
    TestCheckBuffer(ge::DT_UINT16, {2, 2346, 2060, 2, 2}, {1, 1, 2}, {541, 224, 631, 1046, 2, 0});
    TestCheckBuffer(ge::DT_UINT16, {2, 2346, 2060, 2, 2}, {1, 1}, {1, 0, 1, 0});
}

// Asymmetric block_shape {2,1} with crops on second dim; expect SmallC tiling
TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_asymmetric_block_shape_cut)
{
    auto dataType = ge::DT_FLOAT16;
    std::vector<int32_t> blockShapeValues = {2, 1};
    std::vector<int32_t> cropsValues = {0, 0, 1, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 3, 5, 4}, {2, 3, 5, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 4, 4}, {1, 6, 4, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
}

// Large spatial axis with small block_shape dims; expect SmallC tiling
TEST_F(BatchToSpaceNDTilingSmallCTest, BatchToSpaceNDTilingSmallCTest_large_space_small_blockshape)
{
    auto dataType = ge::DT_FLOAT16;
    std::vector<int32_t> blockShapeValues = {3, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{6, 2000, 3, 4}, {6, 2000, 3, 4}}, dataType, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6000, 6, 4}, {1, 6000, 6, 4}}, dataType, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSmallCTilingData));
    B2SNDSmallCTilingData* data = reinterpret_cast<B2SNDSmallCTilingData*>(tilingInfo.tilingData.get());
    // Verify tiling completed successfully with SmallC mode
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000010'00000010);
    EXPECT_GT(data->ubTotalCount, 0);
}
