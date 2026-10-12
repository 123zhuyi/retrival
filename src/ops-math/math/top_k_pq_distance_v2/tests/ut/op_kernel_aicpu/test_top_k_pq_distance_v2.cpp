/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

#include <vector>

using namespace std;
using namespace aicpu;

class TEST_TOP_K_PQ_DISTANCE_V2_UT : public testing::Test {};

namespace {
// order must be wrapped in std::string: a bare string literal would bind to NodeDefBuilder::Attr(name, bool)
// rather than Attr(name, const std::string&), silently registering "order" as a boolean attr.
#define CREATE_NODEDEF(node_def, shapes, data_types, datas, order, k, group_size) \
    NodeDefBuilder(node_def.get(), "TopKPQDistanceV2", "TopKPQDistanceV2")        \
        .Input({"pq_distance", data_types[0], shapes[0], datas[0]})               \
        .Input({"grouped_extreme_distance", data_types[1], shapes[1], datas[1]})  \
        .Output({"topk_distance", data_types[2], shapes[2], datas[2]})            \
        .Output({"topk_index", data_types[3], shapes[3], datas[3]})               \
        .Attr("order", std::string(order))                                        \
        .Attr("k", static_cast<int32_t>(k))                                       \
        .Attr("group_size", static_cast<int32_t>(group_size))

// Shared fixture: one batch row of 8 tie-free distances split into 4 groups of 2.
//   pq_distance = {5, 1, 9, 3, 7, 2, 8, 6}
//   groups       g0={5,1} g1={9,3} g2={7,2} g3={8,6}
// ASC keeps the 4 smallest {1,2,3,5} at indices {1,5,3,0}; DES keeps the 4 largest {9,8,7,6} at {2,6,4,7}.
} // namespace

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, DATA_TYPE_FLOAT_ASC_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    // ASC selects the smallest element of every group as its extreme.
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    constexpr int32_t kResultNum = 4;
    float expect_distance[kResultNum] = {1.0f, 2.0f, 3.0f, 5.0f};
    int32_t expect_index[kResultNum] = {1, 5, 3, 0};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_FLOAT_EQ(topk_distance[i], expect_distance[i]);
        EXPECT_EQ(topk_index[i], expect_index[i]);
    }
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, DATA_TYPE_FLOAT16_DES_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    // 1..9 are exactly representable in float16, so the comparison below stays exact.
    Eigen::half pq_distance[8] = {Eigen::half(5.0f), Eigen::half(1.0f), Eigen::half(9.0f), Eigen::half(3.0f),
                                  Eigen::half(7.0f), Eigen::half(2.0f), Eigen::half(8.0f), Eigen::half(6.0f)};
    // DES selects the largest element of every group as its extreme.
    Eigen::half grouped_extreme[4] = {Eigen::half(5.0f), Eigen::half(9.0f), Eigen::half(7.0f), Eigen::half(8.0f)};
    Eigen::half topk_distance[4] = {Eigen::half(0.0f)};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "DES", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    constexpr int32_t kResultNum = 4;
    float expect_distance[kResultNum] = {9.0f, 8.0f, 7.0f, 6.0f};
    int32_t expect_index[kResultNum] = {2, 6, 4, 7};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_FLOAT_EQ(static_cast<float>(topk_distance[i]), expect_distance[i]);
        EXPECT_EQ(topk_index[i], expect_index[i]);
    }
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, DATA_TYPE_INT32_ASC_MULTI_BATCH_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 8}, {2, 4}, {2, 4}, {2, 4}};

    int32_t pq_distance[16] = {5, 1, 9, 3, 7, 2, 8, 6, 5, 1, 9, 3, 7, 2, 8, 6};
    int32_t grouped_extreme[8] = {1, 3, 2, 6, 1, 3, 2, 6};
    int32_t topk_distance[8] = {0};
    int32_t topk_index[8] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // Both rows hold the same data, so both produce the same top-k. 2 batches * k(4) = 8 results.
    constexpr int32_t kResultNum = 8;
    int32_t expect_distance[kResultNum] = {1, 2, 3, 5, 1, 2, 3, 5};
    int32_t expect_index[kResultNum] = {1, 5, 3, 0, 1, 5, 3, 0};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_EQ(topk_distance[i], expect_distance[i]);
        EXPECT_EQ(topk_index[i], expect_index[i]);
    }
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, GROUP_SIZE_EQUALS_ELEMENT_NUM_SUCC)
{
    // extreme_size == 1 < k exercises the size = min(k, extreme_size) clamp.
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 1}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[1] = {1.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 8);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    constexpr int32_t kResultNum = 4;
    float expect_distance[kResultNum] = {1.0f, 2.0f, 3.0f, 5.0f};
    int32_t expect_index[kResultNum] = {1, 5, 3, 0};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_FLOAT_EQ(topk_distance[i], expect_distance[i]);
        EXPECT_EQ(topk_index[i], expect_index[i]);
    }
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, DATA_TYPE_UNSUPPORTED_FAIL)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    int64_t pq_distance[8] = {5, 1, 9, 3, 7, 2, 8, 6};
    int64_t grouped_extreme[4] = {1, 3, 2, 6};
    int64_t topk_distance[4] = {0};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, ATTR_ORDER_INVALID_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "asc", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, ATTR_K_NOT_POSITIVE_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 0, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, ATTR_K_GREATER_THAN_ELEMENT_NUM_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 16}, {1, 16}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[16] = {0.0f};
    int32_t topk_index[16] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 16, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, ATTR_GROUP_SIZE_ZERO_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 0);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, ELEMENT_NUM_NOT_MULTIPLE_OF_GROUP_SIZE_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 3);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, INPUT_NOT_2D_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{8}, {4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    float grouped_extreme[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOP_K_PQ_DISTANCE_V2_UT, DISTANCE_DTYPE_MISMATCH_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 8}, {1, 4}, {1, 4}, {1, 4}};

    float pq_distance[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
    Eigen::half grouped_extreme[4] = {Eigen::half(1.0f), Eigen::half(3.0f), Eigen::half(2.0f), Eigen::half(6.0f)};
    float topk_distance[4] = {0.0f};
    int32_t topk_index[4] = {0};

    vector<void*> datas = {(void*)pq_distance, (void*)grouped_extreme, (void*)topk_distance, (void*)topk_index};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, "ASC", 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
