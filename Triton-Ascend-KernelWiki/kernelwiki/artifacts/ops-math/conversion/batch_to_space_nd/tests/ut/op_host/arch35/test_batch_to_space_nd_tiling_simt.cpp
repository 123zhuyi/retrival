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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class BatchToSpaceNDTilingSimtTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceNDTilingSimtTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceNDTilingSimtTest TearDown" << std::endl; }
};

namespace {
B2SNDCompileInfo compileInfo;
}

TEST_F(BatchToSpaceNDTilingSimtTest, BatchToSpaceNDTilingSimtTest_emptyOutput)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {2, 4};
    size_t bsDimNum = blockShapeValues.size();
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 0, 4}, {1, 0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSimtTilingData));
    B2SNDSimtTilingData* data = reinterpret_cast<B2SNDSimtTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalBlock, 1);
    EXPECT_EQ(data->mainCoreBlock, 1);
    EXPECT_EQ(data->needCoreNum, 1);
    EXPECT_EQ(data->mainCoreNum, 1);
    EXPECT_EQ(data->blockSize, 16384);
    EXPECT_EQ(data->tailBlockSize, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(BatchToSpaceNDTilingSimtTest, BatchToSpaceNDTilingSimtTest_4d_smallshape)
{
    std::vector<int32_t> blockShapeValues = {2, 3, 5, 7};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t bsDimNum = blockShapeValues.size();
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{420, 3, 5, 7, 2, 3}, {420, 3, 5, 7, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSimtTilingData));
    B2SNDSimtTilingData* data = reinterpret_cast<B2SNDSimtTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalBlock, 17);
    EXPECT_EQ(data->mainCoreBlock, 1);
    EXPECT_EQ(data->needCoreNum, 17);
    EXPECT_EQ(data->mainCoreNum, 17);
    EXPECT_EQ(data->blockSize, 16384);
    EXPECT_EQ(data->tailBlockSize, 2456);
    EXPECT_EQ(tilingInfo.blockNum, 17);
}

TEST_F(BatchToSpaceNDTilingSimtTest, BatchToSpaceNDTilingSimtTest_4d_bigshape)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t bsDimNum = blockShapeValues.size();
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{32, 2, 2, 2, 2, 1LL << 32}, {32, 2, 2, 2, 2, 1LL << 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 4, 4, 4, 4, 1LL << 32}, {2, 4, 4, 4, 4, 1LL << 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSimtTilingData));
    B2SNDSimtTilingData* data = reinterpret_cast<B2SNDSimtTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalBlock, 134217728);
    EXPECT_EQ(data->mainCoreBlock, 2097152);
    EXPECT_EQ(data->needCoreNum, 64);
    EXPECT_EQ(data->mainCoreNum, 64);
    EXPECT_EQ(data->blockSize, 16384);
    EXPECT_EQ(data->tailBlockSize, 16384);
    EXPECT_EQ(tilingInfo.blockNum, 64);
}

// Bool dtype with small remain axis in 4D shape; expect SIMT tiling
TEST_F(BatchToSpaceNDTilingSimtTest, BatchToSpaceNDTilingSimtTest_dsize1_bool_4d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    size_t bsDimNum = blockShapeValues.size();
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 3, 5, 3}, {4, 3, 5, 3}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 10, 3}, {1, 6, 10, 3}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSimtTilingData));
}

// Int8 dtype with small remain axis in 3D shape; expect SIMT tiling
TEST_F(BatchToSpaceNDTilingSimtTest, BatchToSpaceNDTilingSimtTest_dsize1_int8_3d)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    size_t bsDimNum = blockShapeValues.size();
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{6, 3, 4}, {6, 3, 4}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{bsDimNum}, {bsDimNum}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{bsDimNum, 2}, {bsDimNum, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{3, 6, 4}, {3, 6, 4}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(B2SNDSimtTilingData));
}
