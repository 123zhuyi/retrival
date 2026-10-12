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

using namespace std;
using namespace aicpu;

class TEST_CALC_BUCKETS_LIMIT_AND_OFFSET_UT : public testing::Test {};

template <typename T>
void CalcExpect(const NodeDef& node_def, int32_t expect_out1[], int32_t expect_out2[], bool print_data = false)
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    auto input2 = node_def.MutableInputs(2);
    T* input2_data = (T*)input2->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    int64_t input2_num = input2->NumElements();
    auto attrs = node_def.Attrs();
    int64_t total_limit = attrs["total_limit"]->GetInt();
    int32_t limit = 0;
    int64_t sum = 0;
    for (int64_t i = 0; i < input0_num; ++i) {
        if (input0_data[i] >= input1_num || input0_data[i] >= input2_num) {
            cout << "input0_data[" << i << "] out of range input1 [0," << input1_num << ")" << endl;
            cout << "input0_data[" << i << "] out of range input2 [0," << input2_num << ")" << endl;
            return;
        }
        expect_out1[i] = input1_data[input0_data[i]];
        expect_out2[i] = input2_data[input0_data[i]];
        sum += expect_out1[i];
        limit = max(limit, expect_out1[i]);
    }

    while (sum > total_limit) {
        sum = 0;
        limit--;
        for (int64_t i = 0; i < input0_num; ++i) {
            sum += min(limit, expect_out1[i]);
        }
    }

    for (int64_t i = 0; i < input0_num; ++i) {
        if (expect_out1[i] > limit) {
            expect_out1[i] = limit;
        }
    }

    if (print_data) {
        cout << "expect_buckets_limit:[";
        for (int64_t i = 0; i < input0_num; ++i) {
            cout << expect_out1[i] << ", ";
        }
        cout << "]" << endl;
        cout << "expect_buckets_offset:[";
        for (int64_t i = 0; i < input0_num; ++i) {
            cout << expect_out2[i] << ", ";
        }
        cout << "]" << endl;
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas, total_limit)                               \
    auto node_def = CpuKernelUtils::CreateNodeDef();                                         \
    NodeDefBuilder(node_def.get(), "CalcBucketsLimitAndOffset", "CalcBucketsLimitAndOffset") \
        .Input({"bucket_list", data_types[0], shapes[0], datas[0]})                          \
        .Input({"ivf_count", data_types[1], shapes[1], datas[1]})                            \
        .Input({"ivf_offset", data_types[2], shapes[2], datas[2]})                           \
        .Output({"buckets_limit", data_types[3], shapes[3], datas[3]})                       \
        .Output({"buckets_offset", data_types[4], shapes[4], datas[4]})                      \
        .Attr("total_limit", total_limit)

template <typename T>
void RunLimitKernel(vector<DataType> data_types, vector<vector<int64_t>>& shapes, int32_t& max_count,
                    int32_t& total_limit, bool print_data = false)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    uint64_t input2_size = CalTotalElements(shapes, 1);
    uint64_t input3_size = CalTotalElements(shapes, 2);
    T* input1 = new T[input1_size];
    SetRandomValue<T>(input1, input1_size, 0.0, static_cast<float>(input2_size));
    if (print_data) {
        cout << "bucket_list:[";
        for (uint64_t i = 0; i < input1_size; ++i) {
            cout << input1[i] << ", ";
        }
        cout << "]" << endl;
    }

    T* input2 = new T[input2_size];
    SetRandomValue<T>(input2, input2_size, 0.0, static_cast<float>(max_count + 1));
    if (print_data) {
        cout << "ivf_counts:[";
        for (uint64_t i = 0; i < input2_size; ++i) {
            cout << input2[i] << ", ";
        }
        cout << "]" << endl;
    }

    T* input3 = new T[input3_size];
    SetRandomValue<T>(input3, input3_size, 0.0, static_cast<float>(max_count + 1));
    if (print_data) {
        cout << "ivf_offset:[";
        for (uint64_t i = 0; i < input3_size; ++i) {
            cout << input3[i] << ", ";
        }
        cout << "]" << endl;
    }

    T* output1 = new T[input1_size];
    T* output2 = new T[input1_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)input3, (void*)output1, (void*)output2};

    CREATE_NODEDEF(shapes, data_types, datas, total_limit);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    if (print_data) {
        cout << "buckets_limit:[";
        for (uint64_t i = 0; i < input1_size; ++i) {
            cout << output1[i] << ", ";
        }
        cout << "]" << endl;
        cout << "buckets_offset:[";
        for (uint64_t i = 0; i < input1_size; ++i) {
            cout << output2[i] << ", ";
        }
        cout << "]" << endl;
    }

    T* output_exp1 = new T[input1_size];
    T* output_exp2 = new T[input1_size];

    CalcExpect<T>(*node_def.get(), output_exp1, output_exp2, print_data);

    bool compare = CompareResult(output1, output_exp1, input1_size);
    EXPECT_EQ(compare, true);
    compare = CompareResult(output2, output_exp2, input1_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] input3;
    delete[] output1;
    delete[] output2;
    delete[] output_exp1;
    delete[] output_exp2;
}

TEST_F(TEST_CALC_BUCKETS_LIMIT_AND_OFFSET_UT, Normal_Succ)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {20}, {20}, {5}, {5}};
    int32_t max_count = 10;
    int32_t total_limit = 20;

    RunLimitKernel<int32_t>(data_types, shapes, max_count, total_limit, true);
}

TEST_F(TEST_CALC_BUCKETS_LIMIT_AND_OFFSET_UT, Max_succ)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64}, {1000000}, {1000000}, {64}, {64}};
    int32_t max_count = 400000;
    int32_t total_limit = 20000000;
    RunLimitKernel<int32_t>(data_types, shapes, max_count, total_limit);
}
