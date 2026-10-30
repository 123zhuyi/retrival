/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

using namespace std;
using namespace aicpu;

class TEST_LESS_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num == input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            expect_out[j] = input0_data[j] < input1_data[j] ? true : false;
        }
    }
}

template <typename T>
void CalcExpectWithDiffShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num > input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            int64_t i = j % input1_num;
            expect_out[j] = input0_data[j] < input1_data[i] ? true : false;
        }
    } else {
        for (int64_t j = 0; j < input1_num; ++j) {
            int64_t i = j % input0_num;
            expect_out[j] = input0_data[i] < input1_data[j] ? true : false;
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Less", "Less")                   \
        .Input({"x1", data_types[0], shapes[0], datas[0]})           \
        .Input({"x2", data_types[1], shapes[1], datas[1]})           \
        .Output({"y", data_types[2], shapes[2], datas[2]})

template <typename T1, typename T2, typename T3>
void RunLessKernel(
    vector<DataType> data_types, vector<vector<int64_t>>& shapes, const T1* input1_data, const T2* input2_data,
    const T3* output_exp_data)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];

    uint64_t input2_size = CalTotalElements(shapes, 1);
    T2* input2 = new T2[input2_size];

    for (uint64_t i = 0; i < input1_size; ++i) {
        input1[i] = input1_data[i];
    }
    for (uint64_t i = 0; i < input2_size; ++i) {
        input2[i] = input2_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T3* output_exp = new T3[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1, typename T2, typename T3>
void RunLessKernel2(vector<DataType> data_types, vector<vector<int64_t>>& shapes)
{
    // gen data use SetRandomValue for input1
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];
    SetRandomValue<T1>(input1, input1_size);

    // gen data use SetRandomValue for input2
    uint64_t input2_size = CalTotalElements(shapes, 1);
    T2* input2 = new T2[input2_size];
    SetRandomValue<T2>(input2, input2_size);

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // calculate output_exp
    T3* output_exp = new T3[output_size];
    if (input1_size == input2_size) {
        CalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
    } else {
        CalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT32_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};

    const int32_t input1_data[] = {2, 7, 0, 4, 2, 1, 3, 3, 3, 9, 1, 2, 0, 7, 0, 5, 3, 9, 4, 7, 8, 5, 6, 6,
                                   7, 7, 6, 4, 0, 1, 5, 7, 5, 3, 7, 0, 8, 9, 8, 9, 6, 3, 7, 6, 9, 5, 9, 4,
                                   4, 2, 7, 2, 1, 9, 6, 8, 8, 9, 6, 2, 7, 9, 1, 2, 6, 3, 1, 2, 5, 6, 4, 8};

    const int32_t input2_data[] = {6, 6, 1, 8, 5, 7, 2, 3, 6, 2, 8, 2};

    const bool output_exp_data[] = {true,  false, true,  true,  true,  true,  false, false, true,  false, true,  false,
                                    true,  false, true,  true,  true,  false, false, false, false, false, true,  false,
                                    false, false, false, true,  true,  true,  false, false, true,  false, true,  true,
                                    false, false, false, false, false, true,  false, false, false, false, false, false,
                                    true,  true,  false, true,  true,  false, false, false, false, false, true,  false,
                                    false, false, false, true,  false, true,  true,  true,  true,  false, true,  false};

    RunLessKernel<int32_t, int32_t, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT16_DIFF_SUCC)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
    RunLessKernel2<uint16_t, uint16_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT16_SUCC)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_BOOL};
    vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
    RunLessKernel2<uint16_t, uint16_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT32_SUCC)
{
    vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
    RunLessKernel2<uint32_t, uint32_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
    RunLessKernel2<uint64_t, uint64_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    uint64_t input1[6] = {100, 3, 9, 4, 6, 8};
    uint64_t input2[6] = {3, 5, 9, 6, 0, 4};
    bool output[6] = {false};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool output_exp[6] = {false};
    CalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

// exception instance
TEST_F(TEST_LESS_UT, INPUT_SHAPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    int32_t input1[12] = {(int32_t)1};
    int32_t input2[16] = {(int32_t)0};
    bool output[16] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int32_t input1[22] = {(int32_t)1};
    int64_t input2[22] = {(int64_t)0};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_BOOL_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool input1[22] = {(bool)1};
    bool input2[22] = {(bool)0};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT32_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{64, 32}, {32}, {64, 32}};

    int32_t* input1_data = new int32_t[2048];
    int32_t* input2_data = new int32_t[32];
    bool* output_exp_data = new bool[2048];

    for (int64_t i = 0; i < 2048; ++i) {
        input1_data[i] = static_cast<int32_t>(i % 100);
    }
    for (int64_t i = 0; i < 32; ++i) {
        input2_data[i] = static_cast<int32_t>(50 + i % 50);
    }
    for (int64_t i = 0; i < 2048; ++i) {
        int64_t idx = i % 32;
        output_exp_data[i] = input1_data[i] < input2_data[idx] ? true : false;
    }

    RunLessKernel<int32_t, int32_t, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
    delete[] input1_data;
    delete[] input2_data;
    delete[] output_exp_data;
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT32_SAME_SHAPE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{64, 128}, {64, 128}, {64, 128}};

    int32_t* input1_data = new int32_t[8192];
    int32_t* input2_data = new int32_t[8192];
    bool* output_exp_data = new bool[8192];

    for (int64_t i = 0; i < 8192; ++i) {
        input1_data[i] = static_cast<int32_t>(i % 100);
        input2_data[i] = static_cast<int32_t>((i + 50) % 100);
        output_exp_data[i] = input1_data[i] < input2_data[i] ? true : false;
    }

    RunLessKernel<int32_t, int32_t, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
    delete[] input1_data;
    delete[] input2_data;
    delete[] output_exp_data;
}