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

class TEST_INPLACE_TOP_K_DISTANCE_UT : public testing::Test {};

namespace {
// InplaceTopKDistance has no output: it rewrites topk_pq_distance / topk_pq_index / topk_pq_ivf in place.
#define CREATE_NODEDEF(node_def, shapes, data_types, datas)                      \
    NodeDefBuilder(node_def.get(), "InplaceTopKDistance", "InplaceTopKDistance") \
        .Input({"topk_pq_distance", data_types[0], shapes[0], datas[0]})         \
        .Input({"topk_pq_index", data_types[1], shapes[1], datas[1]})            \
        .Input({"topk_pq_ivf", data_types[2], shapes[2], datas[2]})              \
        .Input({"pq_distance", data_types[3], shapes[3], datas[3]})              \
        .Input({"pq_index", data_types[4], shapes[4], datas[4]})                 \
        .Input({"pq_ivf", data_types[5], shapes[5], datas[5]})                   \
        .Attr("order", order)

// The distances below are deliberately tie-free: std::sort is not stable, so equal distances would leave
// the expected index/ivf ordering unspecified.
} // namespace

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, DATA_TYPE_FLOAT_ASC_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};
    string order = "asc";

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    float pq_distance[3] = {2.0f, 4.0f, 6.0f};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // merged ascending: 1(10,3) 2(20,9) 3(11,3) 4(21,9) 5(12,3) 6(22,9) 7(13,3) 9(14,3); keep the first 5.
    constexpr int32_t kResultNum = 5;
    float expect_distance[kResultNum] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int32_t expect_index[kResultNum] = {10, 20, 11, 21, 12};
    int32_t expect_ivf[kResultNum] = {3, 9, 3, 9, 3};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_FLOAT_EQ(topk_pq_distance[i], expect_distance[i]);
        EXPECT_EQ(topk_pq_index[i], expect_index[i]);
        EXPECT_EQ(topk_pq_ivf[i], expect_ivf[i]);
    }
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, DATA_TYPE_FLOAT16_DESC_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_INT32, DT_FLOAT16, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};
    string order = "desc";

    // 1..9 are exactly representable in float16, so the comparison below stays exact.
    Eigen::half topk_pq_distance[5] = {Eigen::half(1.0f), Eigen::half(3.0f), Eigen::half(5.0f), Eigen::half(7.0f),
                                       Eigen::half(9.0f)};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    Eigen::half pq_distance[3] = {Eigen::half(2.0f), Eigen::half(4.0f), Eigen::half(6.0f)};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // keep the largest 5 of the merged set, written back in descending order.
    constexpr int32_t kResultNum = 5;
    float expect_distance[kResultNum] = {9.0f, 7.0f, 6.0f, 5.0f, 4.0f};
    int32_t expect_index[kResultNum] = {14, 13, 22, 12, 21};
    int32_t expect_ivf[kResultNum] = {3, 3, 9, 3, 9};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_FLOAT_EQ(static_cast<float>(topk_pq_distance[i]), expect_distance[i]);
        EXPECT_EQ(topk_pq_index[i], expect_index[i]);
        EXPECT_EQ(topk_pq_ivf[i], expect_ivf[i]);
    }
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, DATA_TYPE_UNSUPPORTED_FAIL)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};
    string order = "asc";

    int32_t topk_pq_distance[5] = {1, 3, 5, 7, 9};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    int32_t pq_distance[3] = {2, 4, 6};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, DISTANCE_DTYPE_MISMATCH_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT16, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};
    string order = "asc";

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    Eigen::half pq_distance[3] = {Eigen::half(2.0f), Eigen::half(4.0f), Eigen::half(6.0f)};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, INDEX_DTYPE_INVALID_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};
    string order = "asc";

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int64_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    float pq_distance[3] = {2.0f, 4.0f, 6.0f};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, INDEX_ELEMENT_NUM_TOO_SMALL_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {3}, {5}, {3}, {3}, {}};
    string order = "asc";

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int32_t topk_pq_index[3] = {10, 11, 12};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    float pq_distance[3] = {2.0f, 4.0f, 6.0f};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, PQ_IVF_MULTI_ELEMENT_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    // pq_ivf is a shared bucket number; more than one element is meaningless and must be rejected.
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {2}};
    string order = "asc";

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    float pq_distance[3] = {2.0f, 4.0f, 6.0f};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf[2] = {9, 9};

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INPLACE_TOP_K_DISTANCE_UT, ATTR_ORDER_MISSING_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {3}, {3}, {}};

    float topk_pq_distance[5] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int32_t topk_pq_index[5] = {10, 11, 12, 13, 14};
    int32_t topk_pq_ivf[5] = {3, 3, 3, 3, 3};
    float pq_distance[3] = {2.0f, 4.0f, 6.0f};
    int32_t pq_index[3] = {20, 21, 22};
    int32_t pq_ivf = 9;

    vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                           (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "InplaceTopKDistance", "InplaceTopKDistance")
        .Input({"topk_pq_distance", data_types[0], shapes[0], datas[0]})
        .Input({"topk_pq_index", data_types[1], shapes[1], datas[1]})
        .Input({"topk_pq_ivf", data_types[2], shapes[2], datas[2]})
        .Input({"pq_distance", data_types[3], shapes[3], datas[3]})
        .Input({"pq_index", data_types[4], shapes[4], datas[4]})
        .Input({"pq_ivf", data_types[5], shapes[5], datas[5]});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
