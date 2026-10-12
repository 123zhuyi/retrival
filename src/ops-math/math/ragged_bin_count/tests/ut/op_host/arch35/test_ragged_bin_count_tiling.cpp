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
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/ragged_bin_count_precision_policy.h"
#include "../../../../op_kernel/arch35/ragged_bin_count_tiling_data.h"

namespace {
constexpr uint64_t DEFAULT_CORE_NUM = 64U;
constexpr uint64_t DEFAULT_UB_SIZE = 262144U;
constexpr uint64_t DEFAULT_TILING_DATA_SIZE = 4096U;
constexpr size_t EXPECTED_SYSTEM_WORKSPACE = 16U * 1024U * 1024U;
constexpr size_t EXPECTED_USER_WORKSPACE = NsRaggedBinCount::USER_WORKSPACE_HEADER_BYTES;
constexpr size_t EXPECTED_TOTAL_WORKSPACE = EXPECTED_SYSTEM_WORKSPACE + EXPECTED_USER_WORKSPACE;
constexpr uint64_t UINT32_MAX_VALUE = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint64_t THREAD_NUM_U32 = 1024U;
constexpr uint64_t MAX_SUPPORTED_CORES = 128U;
constexpr uint64_t UINT32_INDEX_LIMIT = UINT32_MAX_VALUE - THREAD_NUM_U32 * MAX_SUPPORTED_CORES;

bool UseUint32IndexModel(uint64_t numSplits, uint64_t numValues, uint64_t outputElements)
{
    return numSplits <= UINT32_INDEX_LIMIT && numValues <= UINT32_INDEX_LIMIT && outputElements <= UINT32_INDEX_LIMIT;
}
// usedCoreNum and privateHistElems are the two uint32 fields sharing the tiling data's last eight
// bytes, and the executor prints the raw buffer as int64, so the pair arrives as one packed number.
// Spelling it out here keeps the expectations readable instead of hard-coding values like 51539607553.
std::string CoreAndPrivateHistogram(uint32_t usedCoreNum, uint32_t privateHistElems)
{
    const uint64_t packed = static_cast<uint64_t>(usedCoreNum) | (static_cast<uint64_t>(privateHistElems) << 32U);
    return std::to_string(packed) + ' ';
}

size_t PreciseValueWorkspace(uint32_t usedCoreNum, uint32_t outputElements)
{
    return EXPECTED_SYSTEM_WORKSPACE + EXPECTED_USER_WORKSPACE +
           static_cast<size_t>(usedCoreNum) * outputElements *
               static_cast<size_t>(NsRaggedBinCount::PRECISE_VALUE_PARTITIONS_PER_CORE) *
               static_cast<size_t>(NsRaggedBinCount::PRECISE_VALUE_PARTIAL_BYTES);
}

struct RaggedBinCountCompileInfo {};

template <typename SIZE_TYPE>
void ExecuteNativeDtypeKeyCase(ge::DataType valueDtype, uint32_t key)
{
    RaggedBinCountCompileInfo compileInfo;
    constexpr int64_t numBins = 4;
    const bool valueMapping = (key & 0x4U) != 0U;
    const bool binaryOutput = (key & 0x2U) != 0U;
    const bool hasWeights = (key & 0x1U) != 0U;
    const int64_t numRows = valueMapping ? 1 : 3;
    const int64_t numSplits = numRows + 1;
    const int64_t numValues = valueMapping ? 4096 : 10;
    const int64_t outputElements = numRows * numBins;
    const uint32_t expectedCoreNum = valueMapping ? 4U : 1U;
    SIZE_TYPE sizeData[1] = {static_cast<SIZE_TYPE>(numBins)};
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (binaryOutput) {
        attrs.emplace_back("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true));
    }
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{numSplits}, {numSplits}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{numValues}, {numValues}}, valueDtype, ge::FORMAT_ND},
            {{{1}, {1}}, valueDtype, ge::FORMAT_ND, true, sizeData},
            {{{hasWeights ? numValues : 0}, {hasWeights ? numValues : 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{numRows, numBins}, {numRows, numBins}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        attrs, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    std::ostringstream expectedTilingData;
    // Every shape here is small enough that the host privatises, so privateHistElems == outputElements.
    expectedTilingData << numRows << ' ' << numSplits << ' ' << numValues << ' ' << numBins << ' ' << outputElements
                       << ' ' << CoreAndPrivateHistogram(expectedCoreNum, static_cast<uint32_t>(outputElements));
    SCOPED_TRACE("native dtype=" + std::to_string(static_cast<int32_t>(valueDtype)) + ", key=" + std::to_string(key));
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, key, expectedTilingData.str(), {EXPECTED_TOTAL_WORKSPACE});
}
} // namespace

class RaggedBinCountTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RaggedBinCountTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RaggedBinCountTilingTest TearDown" << std::endl; }
};

TEST_F(RaggedBinCountTilingTest, test_row_mapping_with_weights)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 1U, "3 4 10 4 12 " + CoreAndPrivateHistogram(1U, 12U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_launch_and_precise_workspace_use_scheduled_core_count)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(RaggedBinCountTilingData));
    const auto* tilingData = reinterpret_cast<const RaggedBinCountTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->usedCoreNum, 1U);

    int64_t valueSizeData[1] = {1};
    gert::TilingContextPara valueContext("RaggedBinCount",
                                         {
                                             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                             {{{4096}, {4096}}, ge::DT_INT64, ge::FORMAT_ND},
                                             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, valueSizeData},
                                             {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {
                                             {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    TilingInfo valueTilingInfo;
    ASSERT_TRUE(ExecuteTiling(valueContext, valueTilingInfo));
    EXPECT_EQ(valueTilingInfo.blockNum, 4U);
    ASSERT_GE(valueTilingInfo.tilingDataSize, sizeof(RaggedBinCountTilingData));
    const auto* valueTilingData = reinterpret_cast<const RaggedBinCountTilingData*>(valueTilingInfo.tilingData.get());
    EXPECT_EQ(valueTilingData->usedCoreNum, 4U);
    ASSERT_EQ(valueTilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(valueTilingInfo.workspaceSizes[0], PreciseValueWorkspace(4U, 1U));
}

TEST_F(RaggedBinCountTilingTest, test_binary_output_without_weights)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {4};
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo,
        DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 2U, "3 4 10 4 12 " + CoreAndPrivateHistogram(1U, 12U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_value_mapping_for_single_long_row)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {8};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{1, 8}, {1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 4U, "1 2 4096 8 8 " + CoreAndPrivateHistogram(4U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_empty_values)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 0U, "3 4 0 4 12 " + CoreAndPrivateHistogram(1U, 0U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_zero_size)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {0};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 0U, "2 3 6 0 0 " + CoreAndPrivateHistogram(1U, 0U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_values_and_size_dtype_must_match)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                        {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_weights_length_must_match)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_negative_size_is_rejected)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {-1};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_output_shape_must_match)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_row_binary_with_weights_selects_key_three)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo,
        DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 3U, "3 4 10 4 12 " + CoreAndPrivateHistogram(1U, 12U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_value_mapping_with_weights_selects_key_five)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {8};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{4096}, {4096}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{1, 8}, {1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 5U, "1 2 4096 8 8 " + CoreAndPrivateHistogram(4U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_precise_value_policy_and_workspace_boundaries)
{
    EXPECT_TRUE(NsRaggedBinCount::UsePreciseValuePath(262144, 1));
    EXPECT_FALSE(NsRaggedBinCount::UsePreciseValuePath(262145, 1));
    EXPECT_TRUE(NsRaggedBinCount::UsePreciseValuePath(131072, 2));
    EXPECT_FALSE(NsRaggedBinCount::UsePreciseValuePath(131073, 2));
    EXPECT_FALSE(NsRaggedBinCount::UsePreciseValuePath(4096, 3));

    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {2};
    gert::TilingContextPara preciseContext("RaggedBinCount",
                                           {
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{131072}, {131072}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                               {{{131072}, {131072}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           },
                                           {
                                               {{{1, 2}, {1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           },
                                           {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                           DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(preciseContext, ge::GRAPH_SUCCESS, 5U, "1 2 131072 2 2 " + CoreAndPrivateHistogram(64U, 0U),
                    {PreciseValueWorkspace(64U, 2U)});

    int32_t singleBinSizeData[1] = {1};
    gert::TilingContextPara singleBinPreciseContext(
        "RaggedBinCount",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{262144}, {262144}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, singleBinSizeData},
            {{{262144}, {262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(singleBinPreciseContext, ge::GRAPH_SUCCESS, 5U,
                    "1 2 262144 1 1 " + CoreAndPrivateHistogram(64U, 0U), {PreciseValueWorkspace(64U, 1U)});

    gert::TilingContextPara singleBinFallbackContext(
        "RaggedBinCount",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{262145}, {262145}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, singleBinSizeData},
            {{{262145}, {262145}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(singleBinFallbackContext, ge::GRAPH_SUCCESS, 5U,
                    "1 2 262145 1 1 " + CoreAndPrivateHistogram(64U, 1U), {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara fallbackContext("RaggedBinCount",
                                            {
                                                {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{131073}, {131073}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                {{{131073}, {131073}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{1, 2}, {1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                            DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(fallbackContext, ge::GRAPH_SUCCESS, 5U, "1 2 131073 2 2 " + CoreAndPrivateHistogram(64U, 2U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_value_binary_without_weights_selects_key_six)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {8};
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 8}, {1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo,
        DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 6U, "1 2 4096 8 8 " + CoreAndPrivateHistogram(4U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_value_binary_with_weights_selects_key_seven)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {8};
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
            {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 8}, {1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo,
        DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 7U, "1 2 4096 8 8 " + CoreAndPrivateHistogram(4U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_two_dimensional_values_and_exact_weights_shape_succeeds)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 1U, "2 3 6 4 8 " + CoreAndPrivateHistogram(1U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_splits_must_be_strictly_one_dimensional)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara scalarContext("RaggedBinCount",
                                          {
                                              {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                              {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                              {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {
                                              {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {}, &compileInfo);
    ExecuteTestCase(scalarContext, ge::GRAPH_FAILED);

    gert::TilingContextPara rankTwoContext("RaggedBinCount",
                                           {
                                               {{{1, 3}, {1, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                               {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           },
                                           {
                                               {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           },
                                           {}, &compileInfo);
    ExecuteTestCase(rankTwoContext, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_values_rank_must_be_one_or_two)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara scalarContext("RaggedBinCount",
                                          {
                                              {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                              {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {
                                              {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {}, &compileInfo);
    ExecuteTestCase(scalarContext, ge::GRAPH_FAILED);

    gert::TilingContextPara rankThreeContext("RaggedBinCount",
                                             {
                                                 {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                 {{{1, 2, 3}, {1, 2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                 {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                                 {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {}, &compileInfo);
    ExecuteTestCase(rankThreeContext, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_size_must_have_exact_shape_one)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[2] = {4, 5};
    gert::TilingContextPara scalarContext("RaggedBinCount",
                                          {
                                              {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                              {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                              {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {
                                              {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {}, &compileInfo);
    ExecuteTestCase(scalarContext, ge::GRAPH_FAILED);

    gert::TilingContextPara lengthTwoContext("RaggedBinCount",
                                             {
                                                 {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                 {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                 {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                 {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                                 {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {}, &compileInfo);
    ExecuteTestCase(lengthTwoContext, ge::GRAPH_FAILED);
}

// The GE verifier admits ranks 0 through 2.  Within that range, weights is validated by element
// count alone (see CheckWeightsShape): the kernel walks weights through the flattened values order,
// and canndev's runtime infershape compares GetShapeSize() only.  The cases below pin that rule from
// both sides -- a different supported rank/dimension layout is accepted, rank 3 is rejected, zero
// count in any supported shape means "no weights", and a genuine count mismatch is still rejected.
TEST_F(RaggedBinCountTilingTest, test_weights_matching_element_count_accepts_supported_shape_variants)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    // values is [2, 3]; a flat [6] and a permuted [3, 2] both carry those same six weights, so each
    // must reproduce byte-for-byte the key and TilingData that an exactly-shaped [2, 3] weights gives
    // in test_two_dimensional_values_and_exact_weights_shape_succeeds.
    const std::string expectedTilingData = "2 3 6 4 8 " + CoreAndPrivateHistogram(1U, 8U);

    gert::TilingContextPara flatContext("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(flatContext, ge::GRAPH_SUCCESS, 1U, expectedTilingData, {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara permutedContext("RaggedBinCount",
                                            {
                                                {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                            DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(permutedContext, ge::GRAPH_SUCCESS, 1U, expectedTilingData, {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara rankThreeContext("RaggedBinCount",
                                             {
                                                 {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                 {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                 {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                                 {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                             DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(rankThreeContext, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_empty_weights_of_any_supported_shape_means_no_weights)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    // [0], [0, 3] and [2, 0] all hold zero elements.  Each has to select the unweighted key rather
    // than be rejected for not being the 1-D [0] spelling -- canndev accepts all three.
    const std::string expectedTilingData = "2 3 6 4 8 " + CoreAndPrivateHistogram(1U, 8U);

    gert::TilingContextPara flatEmptyContext("RaggedBinCount",
                                             {
                                                 {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                 {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                 {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                 {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                                 {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                             DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(flatEmptyContext, ge::GRAPH_SUCCESS, 0U, expectedTilingData, {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara leadingZeroContext("RaggedBinCount",
                                               {
                                                   {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                   {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                   {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                   {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {
                                                   {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                               DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(leadingZeroContext, ge::GRAPH_SUCCESS, 0U, expectedTilingData, {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara trailingZeroContext("RaggedBinCount",
                                                {
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                    {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                    {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                                DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(trailingZeroContext, ge::GRAPH_SUCCESS, 0U, expectedTilingData, {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_weights_element_count_mismatch_is_rejected)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    // A scalar weights holds one element, not the six that values [2, 3] needs.  Relaxing the rank
    // check must not turn into accepting any non-empty weights whatsoever.
    gert::TilingContextPara scalarContext("RaggedBinCount",
                                          {
                                              {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                              {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                              {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {
                                              {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          },
                                          {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                                          DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(scalarContext, ge::GRAPH_FAILED);

    gert::TilingContextPara shortContext("RaggedBinCount",
                                         {
                                             {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                             {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                             {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {
                                             {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(shortContext, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_native_dtype_matrix_is_strict)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara splitsDtypeContext("RaggedBinCount",
                                               {
                                                   {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                   {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                   {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                   {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {
                                                   {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {}, &compileInfo);
    ExecuteTestCase(splitsDtypeContext, ge::GRAPH_FAILED);

    gert::TilingContextPara weightsDtypeContext("RaggedBinCount",
                                                {
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                    {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                    {{{0}, {0}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, &compileInfo);
    ExecuteTestCase(weightsDtypeContext, ge::GRAPH_FAILED);

    gert::TilingContextPara outputDtypeContext("RaggedBinCount",
                                               {
                                                   {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                   {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                   {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                   {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {
                                                   {{{2, 4}, {2, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                               },
                                               {}, &compileInfo);
    ExecuteTestCase(outputDtypeContext, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_output_must_be_rank_two)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_values_element_count_overflow_is_rejected)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {0};
    gert::TilingContextPara context(
        "RaggedBinCount",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{std::numeric_limits<int64_t>::max(), 2}, {std::numeric_limits<int64_t>::max(), 2}},
             ge::DT_INT64,
             ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 0}, {1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_zero_aiv_core_is_rejected)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, 0U, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_safe_ceil_div_handles_non_even_workload)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {1};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{2049}, {2049}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 4U, "1 2 2049 1 1 " + CoreAndPrivateHistogram(3U, 1U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_index_width_boundary_model_checks_all_three_domains)
{
    EXPECT_EQ(UINT32_INDEX_LIMIT, 4294836223ULL);
    EXPECT_TRUE(UseUint32IndexModel(UINT32_INDEX_LIMIT, UINT32_INDEX_LIMIT, UINT32_INDEX_LIMIT));
    EXPECT_FALSE(UseUint32IndexModel(UINT32_INDEX_LIMIT + 1U, 1U, 1U));
    EXPECT_FALSE(UseUint32IndexModel(1U, UINT32_INDEX_LIMIT + 1U, 1U));
    EXPECT_FALSE(UseUint32IndexModel(1U, 1U, UINT32_INDEX_LIMIT + 1U));
}

TEST_F(RaggedBinCountTilingTest, test_all_eight_keys_accept_both_native_dtypes)
{
    for (uint32_t key = 0U; key < 8U; ++key) {
        ExecuteNativeDtypeKeyCase<int32_t>(ge::DT_INT32, key);
        ExecuteNativeDtypeKeyCase<int64_t>(ge::DT_INT64, key);
    }
}

TEST_F(RaggedBinCountTilingTest, test_mapping_threshold_selects_row_at_1024_and_value_above_1024)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {1};
    gert::TilingContextPara rowContext("RaggedBinCount",
                                       {
                                           {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                           {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                           {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                           {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(rowContext, ge::GRAPH_SUCCESS, 0U, "1 2 1024 1 1 " + CoreAndPrivateHistogram(1U, 1U),
                    {EXPECTED_TOTAL_WORKSPACE});

    gert::TilingContextPara valueContext("RaggedBinCount",
                                         {
                                             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                             {{{1025}, {1025}}, ge::DT_INT32, ge::FORMAT_ND},
                                             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                             {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {
                                             {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         },
                                         {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(valueContext, ge::GRAPH_SUCCESS, 4U, "1 2 1025 1 1 " + CoreAndPrivateHistogram(2U, 1U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_private_histogram_is_refused_when_output_exceeds_ub_budget)
{
    // 40000 floats is 160000 bytes against the 131072 the SIMT DCache leaves behind, so the output
    // cannot be privatised however favourable the value count is.
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {40000};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{1, 40000}, {1, 40000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 4U, "1 2 1000000 40000 40000 " + CoreAndPrivateHistogram(64U, 0U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_private_histogram_is_refused_when_write_back_costs_more_than_it_saves)
{
    // 6400 floats fit the budget easily, but seven cores would each write the whole output back to
    // replace a mere 64 global atomics. The write-back guard has to reject that trade.
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {100};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{65}, {65}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{64, 100}, {64, 100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 0U, "64 65 64 100 6400 " + CoreAndPrivateHistogram(7U, 0U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_two_dimensional_int64_values_and_weights_use_exact_flat_numel)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {4};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 1U, "2 3 6 4 8 " + CoreAndPrivateHistogram(1U, 8U),
                    {EXPECTED_TOTAL_WORKSPACE});
}

TEST_F(RaggedBinCountTilingTest, test_values_dtype_outside_native_matrix_is_rejected)
{
    RaggedBinCountCompileInfo compileInfo;
    float sizeData[1] = {4.0F};
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_output_element_count_overflow_is_rejected_without_allocation)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {2};
    const int64_t numRows = std::numeric_limits<int64_t>::max() / 2 + 1;
    const int64_t numSplits = numRows + 1;
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{numSplits}, {numSplits}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{numRows, 2}, {numRows, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_output_byte_count_overflow_is_rejected_without_allocation)
{
    RaggedBinCountCompileInfo compileInfo;
    int64_t sizeData[1] = {1};
    const uint64_t maxFloatElements = std::numeric_limits<size_t>::max() / sizeof(float);
    const int64_t numRows = static_cast<int64_t>(maxFloatElements + 1U);
    const int64_t numSplits = numRows + 1;
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{numSplits}, {numSplits}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{numRows, 1}, {numRows, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_ub_must_exceed_dcache_reservation)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    constexpr uint64_t dcacheReservation = 128U * 1024U;
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, dcacheReservation, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountTilingTest, test_dynamic_ub_size_must_fit_host_api_range)
{
    RaggedBinCountCompileInfo compileInfo;
    int32_t sizeData[1] = {4};
    constexpr uint64_t dcacheReservation = 128U * 1024U;
    const uint64_t oversizedUb = dcacheReservation + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1U;
    gert::TilingContextPara context("RaggedBinCount",
                                    {
                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo, DEFAULT_CORE_NUM, oversizedUb, DEFAULT_TILING_DATA_SIZE);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
