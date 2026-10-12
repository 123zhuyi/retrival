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
#include <vector>

#include "gtest/gtest.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"
#include "../../../op_host/ragged_bin_count_check_support.h"

class RaggedBinCountInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RaggedBinCountInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RaggedBinCountInfershapeTest TearDown" << std::endl; }
};

TEST_F(RaggedBinCountInfershapeTest, check_support_accepts_int32_native_combination)
{
    EXPECT_TRUE(ops::IsRaggedBinCountNativeDtypeCombination(ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT,
                                                            ge::DT_FLOAT));
}

TEST_F(RaggedBinCountInfershapeTest, check_support_accepts_int64_native_combination)
{
    EXPECT_TRUE(ops::IsRaggedBinCountNativeDtypeCombination(ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT,
                                                            ge::DT_FLOAT));
}

TEST_F(RaggedBinCountInfershapeTest, check_support_rejects_mismatched_values_and_size)
{
    EXPECT_FALSE(ops::IsRaggedBinCountNativeDtypeCombination(ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT,
                                                             ge::DT_FLOAT));
}

TEST_F(RaggedBinCountInfershapeTest, check_support_rejects_non_fp32_weights)
{
    EXPECT_FALSE(ops::IsRaggedBinCountNativeDtypeCombination(ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16,
                                                             ge::DT_FLOAT));
}

TEST_F(RaggedBinCountInfershapeTest, test_int32_static_shape)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_int64_vector_size_2d_values_and_empty_weights)
{
    int64_t sizeData[1] = {7};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 7}});
}

TEST_F(RaggedBinCountInfershapeTest, test_scalar_size_is_rejected)
{
    int64_t sizeData[1] = {7};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_dynamic_splits)
{
    int32_t sizeData[1] = {8};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 8}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_size)
{
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, -1}});
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_too_short)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_empty_splits_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_element_count_mismatch_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_same_numel_but_different_shape_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_scalar_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_rank_two_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{1, 3}, {1, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_values_scalar_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_values_rank_three_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1, 2, 3}, {1, 2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_size_length_two_is_rejected)
{
    int32_t sizeData[2] = {5, 6};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_empty_size_is_rejected)
{
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_scalar_numel_mismatch_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_rank_three_is_rejected_by_public_contract)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{1, 1, 2}, {1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_zero_numel_of_any_supported_shape_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara leadingZeroContext("RaggedBinCount",
                                                   {
                                                       {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                       {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                       {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                   },
                                                   {
                                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                   });
    ExecuteTestCase(leadingZeroContext, ge::GRAPH_SUCCESS, {{2, 5}});

    gert::InfershapeContextPara trailingZeroContext("RaggedBinCount",
                                                    {
                                                        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                        {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    },
                                                    {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    });
    ExecuteTestCase(trailingZeroContext, ge::GRAPH_SUCCESS, {{2, 5}});

    gert::InfershapeContextPara allZeroContext("RaggedBinCount",
                                               {
                                                   {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                   {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                   {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                   {{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               },
                                               {
                                                   {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               });
    ExecuteTestCase(allZeroContext, ge::GRAPH_SUCCESS, {{2, 5}});

    gert::InfershapeContextPara zeroAndUnknownContext("RaggedBinCount",
                                                      {
                                                          {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                          {{{-1, 0}, {-1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(zeroAndUnknownContext, ge::GRAPH_SUCCESS, {{2, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_scalar_with_matching_numel_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_dynamic_weights_numel_validation_is_deferred_until_tiling)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara dynamicValuesContext("RaggedBinCount",
                                                     {
                                                         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                         {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                         {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                     },
                                                     {
                                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                     });
    ExecuteTestCase(dynamicValuesContext, ge::GRAPH_SUCCESS, {{2, 5}});

    gert::InfershapeContextPara dynamicWeightsContext("RaggedBinCount",
                                                      {
                                                          {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                                          {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(dynamicWeightsContext, ge::GRAPH_SUCCESS, {{2, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_zero_numel_values_rejects_nonempty_weights)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{0, -1}, {0, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_invalid_dimension_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{1, -2}, {1, -2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_element_count_overflow_is_rejected)
{
    int32_t sizeData[1] = {5};
    const int64_t maxDimension = std::numeric_limits<int64_t>::max();
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{maxDimension, 2}, {maxDimension, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_negative_size)
{
    int32_t sizeData[1] = {-1};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// -2 is the UNKNOWN_RANK marker. def.cpp declares DynamicRankSupportFlag(true), so every
// IsUnknownRank branch in the infershape must be exercised rather than assumed (red line R4).
TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_splits_yields_unknown_row_dim)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    // The row count cannot be resolved from an unknown-rank splits, but the bin count still can.
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_values_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_weights_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_size_yields_unknown_bin_dim)
{
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    // No const size tensor is available, so the bin dimension stays unknown.
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, -1}});
}
