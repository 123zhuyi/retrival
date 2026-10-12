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

class BatchToSpaceNDTilingLargeCTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceNDTilingLargeCTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceNDTilingLargeCTest TearDown" << std::endl; }
};

namespace {
B2SNDCompileInfo compileInfo;
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_3d_fullload)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{6, 3, 256}, {6, 3, 256}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{3, 6, 256}, {3, 6, 256}}, ge::DT_INT64, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 1);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 0);
    EXPECT_EQ(data->ubFactor, 3);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_3d_cut_c)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 3, 9000}, {2, 3, 9000}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 9000}, {1, 6, 9000}}, ge::DT_INT64, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 12);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 2);
    EXPECT_EQ(data->ubFactor, 8192);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 12);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_4d_cut_c)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 3, 5, 36000}, {4, 3, 5, 36000}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 10, 36000}, {1, 6, 10, 36000}}, ge::DT_INT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 180);
    EXPECT_EQ(data->perCoreCount, 3);
    EXPECT_EQ(data->ubAxis, 3);
    EXPECT_EQ(data->ubFactor, 16384);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 60);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_5d_cut_c)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{8, 2, 3, 2, 70000}, {8, 2, 3, 2, 70000}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 4, 6, 4, 70000}, {1, 4, 6, 4, 70000}}, ge::DT_BOOL, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 192);
    EXPECT_EQ(data->perCoreCount, 3);
    EXPECT_EQ(data->ubAxis, 4);
    EXPECT_EQ(data->ubFactor, 65536);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 64);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_3d_cut_w)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{2, 300, 256}, {2, 300, 256}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 600, 256}, {1, 600, 256}}, ge::DT_INT64, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 19);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 1);
    EXPECT_EQ(data->ubFactor, 32);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 19);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_4d_cut_w)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 3, 500, 256}, {4, 3, 500, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 1000, 256}, {1, 6, 1000, 256}}, ge::DT_INT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 96);
    EXPECT_EQ(data->perCoreCount, 2);
    EXPECT_EQ(data->ubAxis, 2);
    EXPECT_EQ(data->ubFactor, 64);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 48);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_5d_cut_w)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{8, 2, 3, 230, 257}, {8, 2, 3, 230, 257}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 4, 6, 460, 257}, {1, 4, 6, 460, 257}}, ge::DT_BOOL, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 72);
    EXPECT_EQ(data->perCoreCount, 2);
    EXPECT_EQ(data->ubAxis, 3);
    EXPECT_EQ(data->ubFactor, 226);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 36);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_3d_cut_n)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> cropsValues = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{200, 3, 256}, {200, 3, 256}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{100, 6, 256}, {100, 6, 256}}, ge::DT_INT64, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 20);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 0);
    EXPECT_EQ(data->ubFactor, 5);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 20);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_4d_cut_h)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 300, 5, 256}, {4, 300, 5, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 600, 10, 256}, {1, 600, 10, 256}}, ge::DT_INT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 100);
    EXPECT_EQ(data->perCoreCount, 2);
    EXPECT_EQ(data->ubAxis, 1);
    EXPECT_EQ(data->ubFactor, 6);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 50);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_5d_cut_h)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{8, 2, 300, 2, 257}, {8, 2, 300, 2, 257}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 4, 600, 4, 257}, {1, 4, 600, 4, 257}}, ge::DT_BOOL, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 44);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 2);
    EXPECT_EQ(data->ubFactor, 56);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 44);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_nocrops_4d_cut_neg4)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{4, 3, 5, 256}, {4, 3, 5, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{1, 6, 10, 256}, {1, 6, 10, 256}}, ge::DT_INT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 1);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 1);
    EXPECT_EQ(data->ubFactor, 6);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_cut_w_with_head_crop)
{
    std::vector<int32_t> blockShapeValues = {3};
    std::vector<int32_t> cropsValues = {10, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{12, 20, 256}, {12, 20, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{4, 50, 256}, {4, 50, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 4);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 0);
    EXPECT_EQ(data->ubFactor, 1);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 4);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_cut_w_with_tail_crop)
{
    std::vector<int32_t> blockShapeValues = {4};
    std::vector<int32_t> cropsValues = {0, 8};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{16, 25, 128}, {16, 25, 128}}, ge::DT_UINT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{4, 92, 128}, {4, 92, 128}}, ge::DT_UINT16, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 2);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 0);
    EXPECT_EQ(data->ubFactor, 2);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 2);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_cut_w_with_head_tail_smallbs)
{
    std::vector<int32_t> blockShapeValues = {5};
    std::vector<int32_t> cropsValues = {2, 1};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{10, 50, 130}, {10, 50, 130}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 247, 130}, {2, 247, 130}}, ge::DT_BF16, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 8);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 1);
    EXPECT_EQ(data->ubFactor, 225);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 8);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_cut_w_with_head_tail_bigbs)
{
    std::vector<int32_t> blockShapeValues = {10};
    std::vector<int32_t> cropsValues = {3, 11};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{20, 7, 5000}, {20, 7, 5000}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 56, 5000}, {2, 56, 5000}}, ge::DT_UINT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 44);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 1);
    EXPECT_EQ(data->ubFactor, 3);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 44);
}

TEST_F(BatchToSpaceNDTilingLargeCTest, BatchToSpaceNDTilingLargeCTest_cut_h_with_onlyhead)
{
    std::vector<int32_t> blockShapeValues = {1, 31, 2};
    std::vector<int32_t> cropsValues = {1, 0, 22, 6, 4, 7};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpaceND",
        {
            {{{124, 2, 1, 30, 92}, {124, 2, 1, 30, 92}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{2, 1, 3, 49, 92}, {2, 1, 3, 49, 92}}, ge::DT_UINT32, ge::FORMAT_ND},
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
    EXPECT_EQ(data->totalCount, 2);
    EXPECT_EQ(data->perCoreCount, 1);
    EXPECT_EQ(data->ubAxis, 2);
    EXPECT_EQ(data->ubFactor, 3);
    EXPECT_EQ(data->outputBufferSize, 65536);
    EXPECT_EQ(tilingInfo.blockNum, 2);
}
