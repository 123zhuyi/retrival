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

using ::testing::ElementsAreArray;

class BatchToSpaceNDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceNDTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceNDTilingTest TearDown" << std::endl; }

    std::vector<uint64_t> Shape2Vec(uint64_t shape[], size_t size)
    {
        return std::vector<uint64_t>(shape, shape + size);
    }
};

namespace {
B2SNDCompileInfo compileInfo;
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_x_rank_lt_2)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_block_shape_rank_ne_1)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{8, 2, 2}, {8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_block_shape_dim_lt_1)
{
    std::vector<int32_t> blockShapeValues = {};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_block_shape_dim_ge_x_rank)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 1};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_block_shape_value_le_0)
{
    std::vector<int32_t> blockShapeValues = {0};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_block_shape_not_divisible_by_batch)
{
    std::vector<int32_t> blockShapeValues = {3};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{5, 2, 2}, {5, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_crops_rank_ne_2)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_crops_shape_not_match_block_shape)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_crops_dim2_ne_2)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 3}, {1, 3}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_crops_value_lt_0)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {-1, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_crops_gt_padded_shape)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {5, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_ParamCheck_invalid_dtype)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 2, 2}, {4, 2, 2}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_MergeInput_merge_all_blockshape_3d)
{
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 10, 130}, {2, 10, 130}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 10, 130}, {2, 10, 130}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDLargeCTilingData));
    B2SNDLargeCTilingData* data = reinterpret_cast<B2SNDLargeCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.rank, 3);
    EXPECT_THAT(Shape2Vec(data->input.inShape, 3), ElementsAreArray({2, 1, 1300}));
    EXPECT_THAT(Shape2Vec(data->input.blockShape, 1), ElementsAreArray({1}));
    EXPECT_THAT(Shape2Vec(*(data->input.crops), 2), ElementsAreArray({0, 0}));
}

TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_MergeInput_merge_all_blockshape_4d)
{
    std::vector<int32_t> blockShapeValues = {1, 1};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 3, 4, 200}, {2, 3, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 3, 4, 200}, {2, 3, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDLargeCTilingData));
    B2SNDLargeCTilingData* data = reinterpret_cast<B2SNDLargeCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.rank, 3);
    EXPECT_THAT(Shape2Vec(data->input.inShape, 3), ElementsAreArray({2, 1, 2400}));
    EXPECT_THAT(Shape2Vec(data->input.blockShape, 1), ElementsAreArray({1}));
    EXPECT_THAT(Shape2Vec(*(data->input.crops), 2), ElementsAreArray({0, 0}));
}

// Block_shape first dim is 1 and next dim crops are all zero, spatial dims merge; expect LargeC tiling
TEST_F(BatchToSpaceNDTilingTest, BatchToSpaceNDTilingTest_MergeInput_blockshape1_next_crops_zero)
{
    std::vector<int32_t> blockShapeValues = {1, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 3, 5, 256}, {4, 3, 5, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 3, 10, 256}, {2, 3, 10, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    // LargeC mode (mode=1), blockShapeDimNum=0 for LargeC, isBigShape=0
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDLargeCTilingData));
    B2SNDLargeCTilingData* data = reinterpret_cast<B2SNDLargeCTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.rank, 3);
    // block_shape[0]=1 with next crops both 0 merges spatial dim 1 into dim 2
    EXPECT_THAT(Shape2Vec(data->input.inShape, 3), ElementsAreArray({4, 15, 256}));
    EXPECT_THAT(Shape2Vec(data->input.blockShape, 1), ElementsAreArray({2}));
    EXPECT_THAT(Shape2Vec(*(data->input.crops), 2), ElementsAreArray({0, 0}));
}
