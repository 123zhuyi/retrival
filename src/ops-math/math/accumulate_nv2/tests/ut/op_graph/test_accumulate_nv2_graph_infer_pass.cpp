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
#include <vector>

#include <gtest/gtest.h>
#include "../../../op_graph/accumulate_nv2_graph_infer.h"

TEST(AccumulateNV2GraphInfer, EmptyBroadcastDoesNotDivideByZero)
{
    std::vector<int64_t> output{1, 2, 1, 2};
    std::vector<int64_t> merged;
    ASSERT_EQ(ge::accumulate_nv2::MergeBroadcastShape(output, {0, 1, 1, 2}, merged), ge::GRAPH_SUCCESS);
    output = merged;
    ASSERT_EQ(ge::accumulate_nv2::MergeBroadcastShape(output, {0, 1, 3, 2}, merged), ge::GRAPH_SUCCESS);
    output = merged;
    ASSERT_EQ(ge::accumulate_nv2::MergeBroadcastShape(output, {1, 1, 2}, merged), ge::GRAPH_SUCCESS);
    EXPECT_EQ(merged, (std::vector<int64_t>{0, 2, 3, 2}));
}

TEST(AccumulateNV2GraphInfer, EqualZeroDimensionsAreValid)
{
    std::vector<int64_t> output;
    ASSERT_EQ(ge::accumulate_nv2::MergeBroadcastShape({0, 3}, {0, 1}, output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output, (std::vector<int64_t>{0, 3}));
}

TEST(AccumulateNV2GraphInfer, StandardBroadcast)
{
    std::vector<int64_t> output;
    ASSERT_EQ(ge::accumulate_nv2::MergeBroadcastShape({2, 1}, {1, 3}, output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output, (std::vector<int64_t>{2, 3}));
}

TEST(AccumulateNV2GraphInfer, IncompatibleShapesFail)
{
    std::vector<int64_t> output;
    EXPECT_EQ(ge::accumulate_nv2::MergeBroadcastShape({2, 3}, {4, 3}, output), ge::GRAPH_FAILED);
}
