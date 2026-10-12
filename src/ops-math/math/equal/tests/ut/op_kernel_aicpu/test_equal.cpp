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
#include <string>
#include <iostream>
using namespace std;
using namespace aicpu;

class TEST_EQUAL_UT : public testing::Test {};

namespace {
template <typename T>
void CalcEqualExpectWithSameShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num == input1_num) {
        for (int64_t i = 0; i < input0_num; ++i) {
            expect_out[i] = (input0_data[i] == input1_data[i]);
        }
    }
}

template <typename T>
void CalcEqualExpectWithDiffShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num > input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            expect_out[j] = (input0_data[j] == input1_data[0]);
        }
    }
}
} // namespace

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Equal", "Equal")                 \
        .Input({"x1", data_types[0], shapes[0], datas[0]})           \
        .Input({"x2", data_types[1], shapes[1], datas[1]})           \
        .Output({"y", data_types[2], shapes[2], datas[2]})

#define ADD_CASE(base_type, aicpu_type)                                       \
    TEST_F(TEST_EQUAL_UT, TestEqualBroad_##aicpu_type)                        \
    {                                                                         \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_BOOL};      \
        vector<vector<int64_t>> shapes = {{3, 11}, {3, 11}, {3, 11}};         \
        base_type input[33];                                                  \
        SetRandomValue<base_type>(input, 33);                                 \
        bool output[33] = {(bool)0};                                          \
        vector<void*> datas = {(void*)input, (void*)input, (void*)output};    \
        CREATE_NODEDEF(shapes, data_types, datas);                            \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                         \
        bool expect_out[33] = {(bool)0};                                      \
        CalcEqualExpectWithSameShape<base_type>(*node_def.get(), expect_out); \
        bool ret01 = CompareResult<bool>(output, expect_out, 33);             \
        EXPECT_EQ(ret01, true);                                               \
    }

#define ADD_CASE_WITH_BROADCAST(base_type, aicpu_type)                        \
    TEST_F(TEST_EQUAL_UT, TestEqual_##aicpu_type)                             \
    {                                                                         \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_BOOL};      \
        vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};             \
        base_type input[22];                                                  \
        SetRandomValue<base_type>(input, 22);                                 \
        bool output[22] = {(bool)0};                                          \
        vector<void*> datas = {(void*)input, (void*)input, (void*)output};    \
        CREATE_NODEDEF(shapes, data_types, datas);                            \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                         \
        bool expect_out[22] = {(bool)0};                                      \
        CalcEqualExpectWithDiffShape<base_type>(*node_def.get(), expect_out); \
        bool ret02 = CompareResult<bool>(output, expect_out, 22);             \
        EXPECT_EQ(ret02, true);                                               \
    }

TEST_F(TEST_EQUAL_UT, ExpInput)
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

TEST_F(TEST_EQUAL_UT, ExpInputDiffDtype)
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

TEST_F(TEST_EQUAL_UT, ExpInputNull)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_EQUAL_UT, ExpInputBool)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool input1[22] = {(bool)1};
    bool input2[22] = {(bool)0};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_EQUAL_UT, ExpInputInf)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    double input1[1] = {(double)std::numeric_limits<double>::infinity()};
    double input2[1] = {(double)std::numeric_limits<double>::infinity()};
    bool output[1] = {(bool)1};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_EQUAL_UT, ExpInputNan)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    double input1[1] = {(double)std::numeric_limits<double>::quiet_NaN()};
    double input2[1] = {(double)std::numeric_limits<double>::quiet_NaN()};
    bool output[1] = {(bool)1};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_EQUAL_UT, NotSupportType)
{
    vector<DataType> data_types = {DT_STRING, DT_STRING, DT_STRING};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    std::string input1[1] = {"test"};
    std::string input2[22] = {"train"};
    bool output[1] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(std::complex<float>, DT_COMPLEX64)

ADD_CASE(std::complex<double>, DT_COMPLEX128)

ADD_CASE_WITH_BROADCAST(int32_t, DT_INT32)

template <typename T>
void EqualBroadcast(
    DataType dtype0, DataType dtype1, DataType output_dtype, vector<vector<int64_t>>& shapes, const T* input0_data,
    const T* input1_data, const bool* output_exp_data)
{
    uint64_t input_size = CalTotalElements(shapes, 0);
    T* input0 = new T[input_size];
    for (uint64_t i = 0; i < input_size; ++i) {
        input0[i] = input0_data[i];
    }

    input_size = CalTotalElements(shapes, 1);
    T* input1 = new T[input_size];
    for (uint64_t i = 0; i < input_size; ++i) {
        input1[i] = input1_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 2);
    bool* output = new bool[output_size];

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "Equal", "Equal")
        .Input({"x1", dtype0, shapes[0], (void*)input0})
        .Input({"x2", dtype1, shapes[1], (void*)input1})
        .Output({"y", output_dtype, shapes[2], (void*)output});

    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool* output_exp = new bool[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input0;
    delete[] input1;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_EQUAL_UT, TestEqual_int32_int32_broadcast1)
{
    vector<vector<int64_t>> shapes{{2, 4}, {2, 2, 4}, {2, 2, 4}};
    const int32_t input0_data[] = {8, -3, 6, 5, -6, 8, 3, 7};
    const int32_t input1_data[] = {-7, -7, 9, 7, -8, 6, 4, -10, -5, -7, -1, -6, -3, -2, 2, -5};
    const bool output_exp_data[] = {false, false, false, false, false, false, false, false,
                                    false, false, false, false, false, false, false, false};
    EqualBroadcast<int32_t>(DT_INT32, DT_INT32, DT_BOOL, shapes, input0_data, input1_data, output_exp_data);
}
