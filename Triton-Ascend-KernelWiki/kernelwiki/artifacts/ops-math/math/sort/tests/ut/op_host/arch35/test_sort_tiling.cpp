/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "math/sort/op_host/arch35/sort_tiling.h"
#include "math/sort/op_host/arch35/sort_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/sort_tiling_data.h"

using namespace std;
using namespace ge;

class SortTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

namespace {
constexpr size_t WORK_SPACE_SIZE = 16777216;
optiling::SortCompileInfo g_compileInfo = {64, 1, 0, 4};

gert::TilingContextPara MakeSortTilingContext(const gert::StorageShape& storageShape, ge::DataType xDtype,
                                              ge::DataType y2Dtype, int64_t dim = -1, bool descending = false,
                                              bool stable = false)
{
    return gert::TilingContextPara(
        "Sort",
        {
            {storageShape, xDtype, ge::FORMAT_ND},
        },
        {
            {storageShape, xDtype, ge::FORMAT_ND},
            {storageShape, y2Dtype, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(descending)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(stable)),
        },
        &g_compileInfo);
}
} // namespace

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_basic_2x1024)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_1d_512)
{
    auto tilingContextPara = MakeSortTilingContext({{512}, {512}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_bf16_32x1024)
{
    auto tilingContextPara = MakeSortTilingContext({{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::DT_INT64);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_3d_8x16x1024)
{
    auto tilingContextPara = MakeSortTilingContext({{8, 16, 1024}, {8, 16, 1024}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_outidx_int32_2x1024)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::DT_INT32);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_desc_2x1024)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::DT_INT64, -1, true);

    uint64_t expectTilingKey = 65792;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort32_small_axis_fp32_2x32)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 32}, {2, 32}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 264;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_axis_one_copy_fp32_outidx_int64_4096x1)
{
    auto tilingContextPara = MakeSortTilingContext({{4096, 1}, {4096, 1}}, ge::DT_FLOAT, ge::DT_INT64, 1);

    uint64_t expectTilingKey = 263;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_axis_one_copy_int8_outidx_int32_desc_8192x1)
{
    auto tilingContextPara = MakeSortTilingContext({{8192, 1}, {8192, 1}}, ge::DT_INT8, ge::DT_INT32, 1, true);

    uint64_t expectTilingKey = 65799;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_sort_fp32_boundary_4096)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 4096}, {2, 4096}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_big_size_fp32_boundary_4097)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 259;
    std::vector<size_t> expectWorkspaces = {16941096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_radix_one_core_precedes_merge_intra_core_fp32_128x8192)
{
    auto tilingContextPara = MakeSortTilingContext({{128, 8192}, {128, 8192}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_intra_core_fp32_64x16384)
{
    auto tilingContextPara = MakeSortTilingContext({{64, 16384}, {64, 16384}}, ge::DT_FLOAT, ge::DT_INT64);

    uint64_t expectTilingKey = 260;
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_int64_outidx_int64_256x12)
{
    auto tilingContextPara = MakeSortTilingContext({{256, 12}, {256, 12}}, ge::DT_INT64, ge::DT_INT64);

    uint64_t expectTilingKey = 261;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_int64_outidx_int64_desc_448x12)
{
    auto tilingContextPara = MakeSortTilingContext({{448, 12}, {448, 12}}, ge::DT_INT64, ge::DT_INT64, -1, true);

    uint64_t expectTilingKey = 65797;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_fp16_boundary_3072x8)
{
    auto tilingContextPara = MakeSortTilingContext({{3072, 8}, {3072, 8}}, ge::DT_FLOAT16, ge::DT_INT32);

    uint64_t expectTilingKey = 261;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_bf16_boundary_1024x4)
{
    auto tilingContextPara = MakeSortTilingContext({{1024, 4}, {1024, 4}}, ge::DT_BF16, ge::DT_INT32);

    uint64_t expectTilingKey = 261;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_int32_boundary_256x11)
{
    auto tilingContextPara = MakeSortTilingContext({{256, 11}, {256, 11}}, ge::DT_INT32, ge::DT_INT32);

    uint64_t expectTilingKey = 261;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_insertion_uint32_boundary_256x11)
{
    auto tilingContextPara = MakeSortTilingContext({{256, 11}, {256, 11}}, ge::DT_UINT32, ge::DT_INT32);

    uint64_t expectTilingKey = 261;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_int32_boundary_256x13)
{
    auto tilingContextPara = MakeSortTilingContext({{256, 13}, {256, 13}}, ge::DT_INT32, ge::DT_INT32);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_fp16_boundary_4096x54)
{
    auto tilingContextPara = MakeSortTilingContext({{4096, 54}, {4096, 54}}, ge::DT_FLOAT16, ge::DT_INT32);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_uint64_boundary_512x128)
{
    auto tilingContextPara = MakeSortTilingContext({{512, 128}, {512, 128}}, ge::DT_UINT64, ge::DT_INT64);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_uint16_boundary_1024x192)
{
    auto tilingContextPara = MakeSortTilingContext({{1024, 192}, {1024, 192}}, ge::DT_UINT16, ge::DT_INT32);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_rank_inverse_int64_outidx_int32_4096x16)
{
    auto tilingContextPara = MakeSortTilingContext({{4096, 16}, {4096, 16}}, ge::DT_INT64, ge::DT_INT32);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_rank_inverse_int64_outidx_int64_desc_4096x16)
{
    auto tilingContextPara = MakeSortTilingContext({{4096, 16}, {4096, 16}}, ge::DT_INT64, ge::DT_INT64, -1, true);

    uint64_t expectTilingKey = 65798;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_no_rank_inverse_int8_outidx_int32_1024x96)
{
    auto tilingContextPara = MakeSortTilingContext({{1024, 96}, {1024, 96}}, ge::DT_INT8, ge::DT_INT32);

    uint64_t expectTilingKey = 262;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_two_stage_uint8_7524x127_balances_batch_size)
{
    auto tilingContextPara = MakeSortTilingContext({{7524, 127}, {7524, 127}}, ge::DT_UINT8, ge::DT_INT64);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 262);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->numTileDataSize, 127);
    EXPECT_EQ(tilingData->keyParams0, 20);
    EXPECT_EQ(tilingData->keyParams1, 377);
    EXPECT_EQ(tilingData->keyParams2, 0);
    EXPECT_EQ(tilingData->keyParams4, 0);
    EXPECT_EQ(tilingData->lastAxisNum, 127);
    EXPECT_EQ(tilingData->unsortedDimNum, 7524);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_small_axis_fallback_radix_one_core_int8_outidx_int32_768x96)
{
    auto tilingContextPara = MakeSortTilingContext({{768, 96}, {768, 96}}, ge::DT_INT8, ge::DT_INT32);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_small_axis_fallback_radix_one_core_int64_outidx_int64_2x16)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 16}, {2, 16}}, ge::DT_INT64, ge::DT_INT64);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_large_batch_falls_back_when_small_axis_batching_overflows)
{
    int64_t largeBatch = (static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1) * static_cast<int64_t>(32);
    auto tilingContextPara = MakeSortTilingContext({{largeBatch, 512}, {largeBatch, 512}}, ge::DT_INT64, ge::DT_INT64);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {WORK_SPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_radix_counter_range_boundary)
{
    constexpr int64_t radixUint32ValueMax = 0x3fffffff;

    EXPECT_TRUE(optiling::IsRadixUint32CounterRange(radixUint32ValueMax));
    EXPECT_FALSE(optiling::IsRadixUint32CounterRange(radixUint32ValueMax + 1));
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_insertion_fp32_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::DT_INT64, 0);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 261);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->keyParams3, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_two_stage_int32_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{8, 13, 64}, {8, 13, 64}}, ge::DT_INT32, ge::DT_INT32, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 262);
    EXPECT_EQ(tilingInfo.blockNum, 8);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->keyParams3, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_int32_desc_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{4, 128, 17}, {4, 128, 17}}, ge::DT_INT32, ge::DT_INT32, 1, true);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 65802);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 17);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_fp32_unaligned_axis_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 31, 5}, {2, 31, 5}}, ge::DT_FLOAT, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 265);
    EXPECT_EQ(tilingInfo.blockNum, 10);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 5);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_uint16_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{3, 192, 9}, {3, 192, 9}}, ge::DT_UINT16, ge::DT_INT32, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 27);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 9);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_uint8_inner_smaller_than_preferred_chunk)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 7, 3}, {2, 7, 3}}, ge::DT_UINT8, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 6);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 3);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_uint8_axis2048_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 2048, 1}, {2, 2048, 1}}, ge::DT_UINT8, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 2);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_rejects_non_last_uint8_axis2049_for_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 2049, 1}, {2, 2049, 1}}, ge::DT_UINT8, ge::DT_INT64, 1);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_uint16_axis2048_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 2048, 1}, {2, 2048, 1}}, ge::DT_UINT16, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 2);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_rejects_non_last_uint16_axis2049_for_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 2049, 1}, {2, 2049, 1}}, ge::DT_UINT16, ge::DT_INT64, 1);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_fp16_axis1840_uses_radix)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 2, 1840, 1610, 2, 2}, {2, 2, 1840, 1610, 2, 2}}, ge::DT_FLOAT16,
                                                   ge::DT_INT64, 2, true, true);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 65802);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_GT(tilingData->innerChunk, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_bf16_uses_merge_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 16, 32}, {2, 16, 32}}, ge::DT_BF16, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 265);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 32);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_bf16_axis1840_uses_radix)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1840, 1}, {2, 1840, 1}}, ge::DT_BF16, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 2);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_int64_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 16, 32}, {2, 16, 32}}, ge::DT_INT64, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 32);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_non_last_small_axis_uint64_uses_no_transpose)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 16, 32}, {2, 16, 32}}, ge::DT_UINT64, ge::DT_INT64, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(SortRegBaseTilingData));
    const auto* tilingData = reinterpret_cast<const SortRegBaseTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->innerChunk, 1);
    EXPECT_EQ(tilingData->innerLoopNum, 32);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(SortTilingTest, test_sort_rejects_non_last_axis_above_no_transpose_threshold)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 4096, 8}, {2, 4096, 8}}, ge::DT_FLOAT, ge::DT_INT64, 1);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SortTilingTest, test_sort_rejects_invalid_index_output_dtype)
{
    auto tilingContextPara = MakeSortTilingContext({{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::DT_FLOAT);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SortTilingTest, test_sort_rejects_empty_input_shape)
{
    auto tilingContextPara = MakeSortTilingContext({{0, 1024}, {0, 1024}}, ge::DT_FLOAT, ge::DT_INT64);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
