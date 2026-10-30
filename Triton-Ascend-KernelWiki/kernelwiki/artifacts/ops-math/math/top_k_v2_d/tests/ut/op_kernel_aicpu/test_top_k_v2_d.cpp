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

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

#include <algorithm>
#include <cstdint>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {
template <typename T>
struct ValueIndex {
    T value;
    int32_t index;
};

template <typename T>
bool CompareDescending(const ValueIndex<T>& one, const ValueIndex<T>& another)
{
    if (one.value == another.value) {
        return one.index < another.index;
    }
    return one.value > another.value;
}

template <typename T>
bool CompareAscending(const ValueIndex<T>& one, const ValueIndex<T>& another)
{
    if (one.value == another.value) {
        return one.index < another.index;
    }
    return one.value < another.value;
}
} // namespace

class TEST_TOPKV2D_UT : public testing::Test {};

#define CREATE_NODEDEF_V2D(shapes, data_types, datas)              \
    auto node_def = CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(node_def.get(), "TopKV2D", "TopKV2D")           \
        .Input({"x", data_types[0], shapes[0], datas[0]})          \
        .Input({"k", data_types[1], shapes[1], datas[1]})          \
        .Input({"assist_seq", data_types[4], shapes[4], datas[4]}) \
        .Output({"values", data_types[2], shapes[2], datas[2]})    \
        .Output({"indices", data_types[3], shapes[3], datas[3]})   \
        .Attr("sorted", true)                                      \
        .Attr("largest", true)                                     \
        .Attr("dim", -1);

#define CREATE_NODEDEF_V2D_SMALLEST(shapes, data_types, datas)     \
    auto node_def = CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(node_def.get(), "TopKV2D", "TopKV2D")           \
        .Input({"x", data_types[0], shapes[0], datas[0]})          \
        .Input({"k", data_types[1], shapes[1], datas[1]})          \
        .Input({"assist_seq", data_types[4], shapes[4], datas[4]}) \
        .Output({"values", data_types[2], shapes[2], datas[2]})    \
        .Output({"indices", data_types[3], shapes[3], datas[3]})   \
        .Attr("sorted", true)                                      \
        .Attr("largest", false)                                    \
        .Attr("dim", -1);

#define CREATE_NODEDEF_V2D_DIM(shapes, data_types, datas, dim_val) \
    auto node_def = CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(node_def.get(), "TopKV2D", "TopKV2D")           \
        .Input({"x", data_types[0], shapes[0], datas[0]})          \
        .Input({"k", data_types[1], shapes[1], datas[1]})          \
        .Input({"assist_seq", data_types[4], shapes[4], datas[4]}) \
        .Output({"values", data_types[2], shapes[2], datas[2]})    \
        .Output({"indices", data_types[3], shapes[3], datas[3]})   \
        .Attr("sorted", true)                                      \
        .Attr("largest", true)                                     \
        .Attr("dim", dim_val);

#define ADD_CASE_V2D(base_type, aicpu_type)                                                                           \
    TEST_F(TEST_TOPKV2D_UT, TestTopKV2D_##aicpu_type##_LARGEST)                                                       \
    {                                                                                                                 \
        vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32, DT_FLOAT16};                       \
        vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}, {24}};                                                  \
        base_type input[24];                                                                                          \
        SetRandomValue<base_type>(input, 24);                                                                         \
        vector<ValueIndex<base_type>> output_expect(24);                                                              \
        for (int i = 0; i < 24; i++) {                                                                                \
            output_expect[i].index = i;                                                                               \
            output_expect[i].value = input[i];                                                                        \
        }                                                                                                             \
        sort(output_expect.begin(), output_expect.end(), CompareDescending<base_type>);                               \
        int32_t k = 7;                                                                                                \
        base_type output_value[7] = {(base_type)0};                                                                   \
        int32_t output_index[7] = {0};                                                                                \
        Eigen::half assist_seq[24];                                                                                   \
        for (int i = 0; i < 24; i++) {                                                                                \
            assist_seq[i] = Eigen::half(0.0f);                                                                        \
        }                                                                                                             \
        vector<void*> datas = {(void*)input, (void*)&k, (void*)output_value, (void*)output_index, (void*)assist_seq}; \
        CREATE_NODEDEF_V2D(shapes, data_types, datas);                                                                \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                 \
        for (int i = 0; i < 7; i++) {                                                                                 \
            EXPECT_EQ(output_value[i], output_expect[i].value);                                                       \
            EXPECT_EQ(output_index[i], output_expect[i].index);                                                       \
        }                                                                                                             \
    }                                                                                                                 \
    TEST_F(TEST_TOPKV2D_UT, TestTopKV2D_##aicpu_type##_SMALLEST)                                                      \
    {                                                                                                                 \
        vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32, DT_FLOAT16};                       \
        vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}, {24}};                                                  \
        base_type input[24];                                                                                          \
        SetRandomValue<base_type>(input, 24);                                                                         \
        vector<ValueIndex<base_type>> output_expect(24);                                                              \
        for (int i = 0; i < 24; i++) {                                                                                \
            output_expect[i].index = i;                                                                               \
            output_expect[i].value = input[i];                                                                        \
        }                                                                                                             \
        sort(output_expect.begin(), output_expect.end(), CompareAscending<base_type>);                                \
        int32_t k = 7;                                                                                                \
        base_type output_value[7] = {(base_type)0};                                                                   \
        int32_t output_index[7] = {0};                                                                                \
        Eigen::half assist_seq[24];                                                                                   \
        for (int i = 0; i < 24; i++) {                                                                                \
            assist_seq[i] = Eigen::half(0.0f);                                                                        \
        }                                                                                                             \
        vector<void*> datas = {(void*)input, (void*)&k, (void*)output_value, (void*)output_index, (void*)assist_seq}; \
        CREATE_NODEDEF_V2D_SMALLEST(shapes, data_types, datas);                                                       \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                 \
        for (int i = 0; i < 7; i++) {                                                                                 \
            EXPECT_EQ(output_value[i], output_expect[i].value);                                                       \
            EXPECT_EQ(output_index[i], output_expect[i].index);                                                       \
        }                                                                                                             \
    }                                                                                                                 \
    TEST_F(TEST_TOPKV2D_UT, TestTopKV2D_##aicpu_type##_SECOND_LAST_DIM)                                               \
    {                                                                                                                 \
        vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32, DT_FLOAT16};                       \
        vector<vector<int64_t>> shapes = {{2, 3, 4}, {}, {2, 2, 4}, {2, 2, 4}, {2, 3, 4}};                            \
        base_type input[24];                                                                                          \
        for (int i = 0; i < 24; i++) {                                                                                \
            input[i] = base_type(i + 1);                                                                              \
        }                                                                                                             \
        base_type output_value_expect[16] = {base_type(9),  base_type(10), base_type(11), base_type(12),              \
                                             base_type(5),  base_type(6),  base_type(7),  base_type(8),               \
                                             base_type(21), base_type(22), base_type(23), base_type(24),              \
                                             base_type(17), base_type(18), base_type(19), base_type(20)};             \
        int32_t output_index_expect[16] = {2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1};                           \
        int32_t k = 2;                                                                                                \
        base_type output_value[16] = {(base_type)0};                                                                  \
        int32_t output_index[16] = {0};                                                                               \
        Eigen::half assist_seq[24];                                                                                   \
        for (int i = 0; i < 24; i++) {                                                                                \
            assist_seq[i] = Eigen::half(0.0f);                                                                        \
        }                                                                                                             \
        vector<void*> datas = {(void*)input, (void*)&k, (void*)output_value, (void*)output_index, (void*)assist_seq}; \
        CREATE_NODEDEF_V2D_DIM(shapes, data_types, datas, -2);                                                        \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                 \
        for (int i = 0; i < 16; i++) {                                                                                \
            EXPECT_EQ(output_value[i], output_value_expect[i]);                                                       \
            EXPECT_EQ(output_index[i], output_index_expect[i]);                                                       \
        }                                                                                                             \
    }

TEST_F(TEST_TOPKV2D_UT, TestTopKV2D_KVALUE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}, {24}};
    int64_t input[24];
    SetRandomValue<int64_t>(input, 24);
    int32_t k = -1;
    int64_t output_value[7] = {(int64_t)0};
    int32_t output_index[7] = {0};
    Eigen::half assist_seq[24];
    for (int i = 0; i < 24; i++) {
        assist_seq[i] = Eigen::half(0.0f);
    }
    vector<void*> datas = {(void*)input, (void*)&k, (void*)output_value, (void*)output_index, (void*)assist_seq};
    CREATE_NODEDEF_V2D(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TOPKV2D_UT, TestTopKV2D_INPUT_DATADYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}, {24}};
    bool input[24];
    SetRandomValue<bool>(input, 24);
    int32_t k = 7;
    bool output_value[7] = {(bool)0};
    int32_t output_index[7] = {0};
    Eigen::half assist_seq[24];
    for (int i = 0; i < 24; i++) {
        assist_seq[i] = Eigen::half(0.0f);
    }
    vector<void*> datas = {(void*)input, (void*)&k, (void*)output_value, (void*)output_index, (void*)assist_seq};
    CREATE_NODEDEF_V2D(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE_V2D(Eigen::half, DT_FLOAT16)

ADD_CASE_V2D(float, DT_FLOAT)

ADD_CASE_V2D(double, DT_DOUBLE)

ADD_CASE_V2D(int8_t, DT_INT8)

ADD_CASE_V2D(int16_t, DT_INT16)

ADD_CASE_V2D(int32_t, DT_INT32)

ADD_CASE_V2D(int64_t, DT_INT64)

ADD_CASE_V2D(uint8_t, DT_UINT8)

ADD_CASE_V2D(uint16_t, DT_UINT16)

ADD_CASE_V2D(uint32_t, DT_UINT32)

ADD_CASE_V2D(uint64_t, DT_UINT64)
