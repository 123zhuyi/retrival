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

class TEST_RIGHTSHIFT_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "right_shift_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "right_shift_test TearDown\n" << endl;
    }
};

template <typename T>
void RightCalcExpectWithSameShape(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();

    T* in1_clamped = new T[input1_num];
    for (int64_t i = 0; i < input1_num; i++) {
        in1_clamped[i] = input1_data[i];
        if (in1_clamped[i] < 0) {
            in1_clamped[i] = 0;
        } else if (in1_clamped[i] > sizeof(T) * CHAR_BIT - 1) {
            in1_clamped[i] = sizeof(T) * CHAR_BIT - 1;
        }
    }

    if (input0_num == input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            expect_out[j] = input0_data[j] >> in1_clamped[j];
        }
    }
    delete[] in1_clamped;
}

template <typename T>
void RightCalcExpectWithDiffShape(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();

    T* in1_clamped = new T[input1_num];
    for (int64_t i = 0; i < input1_num; i++) {
        in1_clamped[i] = input1_data[i];
        if (in1_clamped[i] < 0) {
            in1_clamped[i] = 0;
        } else if (in1_clamped[i] > sizeof(T) * CHAR_BIT - 1) {
            in1_clamped[i] = sizeof(T) * CHAR_BIT - 1;
        }
    }

    if (input0_num > input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            int64_t i = j % input1_num;
            expect_out[j] = input0_data[j] >> in1_clamped[i];
        }
    } else {
        for (int64_t j = 0; j < input1_num; ++j) {
            int64_t i = j % input0_num;
            expect_out[j] = input0_data[i] >> in1_clamped[j];
        }
    }
    delete[] in1_clamped;
}

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "RightShift", "RightShift")       \
        .Input({"x", data_types[0], shapes[0], datas[0]})            \
        .Input({"y", data_types[1], shapes[1], datas[1]})            \
        .Output({"z", data_types[2], shapes[2], datas[2]})

// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1, typename T2, typename T3>
void RunRightShiftKernel2(vector<DataType> data_types, vector<vector<int64_t>>& shapes)
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
        RightCalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
    } else {
        RightCalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT8_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
    const int8_t input1[72] = {59, 18, 15, 36, 50, 94, 45, 81, 67, 3,  19, 25, 49, 2,  23, 14, 96, 30,
                               5,  99, 73, 7,  8,  70, 39, 87, 58, 84, 48, 46, 14, 21, 87, 83, 80, 79,
                               73, 74, 73, 46, 70, 12, 83, 55, 54, 89, 37, 63, 42, 73, 84, 20, 50, 23,
                               62, 34, 15, 33, 29, 41, 13, 6,  8,  62, 72, 25, 46, 38, 66, 87, 57, 17};
    const int8_t input2[12] = {2, 6, 3, 1, 2, 5, 6, 4, 8, 6, 6, 1};
    int8_t output_exp[72] = {14, 0, 1,  18, 12, 2, 0, 5, 0, 0, 0, 12, 12, 0, 2, 7,  24, 0, 0, 6, 0, 0, 0, 35,
                             9,  1, 7,  42, 12, 1, 0, 1, 0, 1, 1, 39, 18, 1, 9, 23, 17, 0, 1, 3, 0, 1, 0, 31,
                             10, 1, 10, 10, 12, 0, 0, 2, 0, 0, 0, 20, 3,  0, 1, 31, 18, 0, 0, 2, 0, 1, 0, 8};

    int8_t output[72] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 72);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT32_SUCC)
{
    vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};
    const uint32_t input1[36] = {59, 18, 15, 36, 50, 94, 45, 81, 67, 3,  19, 25, 49, 2,  23, 14, 96, 30,
                                 5,  99, 73, 7,  8,  70, 39, 87, 58, 84, 48, 46, 14, 21, 87, 83, 80, 79};
    const uint32_t input2[12] = {8, 9, 8, 9, 6, 3, 7, 6, 9, 5, 9, 4};
    uint32_t output_exp[36] = {0, 0, 0, 0, 0, 11, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3,
                               0, 1, 0, 0, 0, 4,  0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 4};

    uint32_t output[36] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 36);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT16_SUCC2)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    RunRightShiftKernel2<uint16_t, uint16_t, uint16_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT32_SUCC2)
{
    vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
    RunRightShiftKernel2<uint32_t, uint32_t, uint32_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_SUCC2)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
    RunRightShiftKernel2<uint64_t, uint64_t, uint64_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_DIFFTRENT_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
    uint64_t input1[6] = {100, 3, 20, 14, 50, 100};
    uint64_t input2[6] = {2, 3, 4};
    uint64_t output[6] = {0, 0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[6] = {25, 0, 1, 3, 7, 6};
    RightCalcExpectWithDiffShape<uint64_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    uint64_t input1[6] = {100, 3, 9, 14, 6, 8};
    uint64_t input2[6] = {3, 5, 2, 3, 1, 2};
    uint64_t output[6] = {0, 0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[6] = {12, 0, 2, 1, 3, 2};
    RightCalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

// exception instance
TEST_F(TEST_RIGHTSHIFT_UT, INPUT_SHAPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    int32_t input1[16] = {(int32_t)1};
    int32_t input2[12] = {(int32_t)0};
    int32_t output[16] = {(int32_t)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int32_t input1[22] = {(int32_t)1};
    int64_t input2[22] = {(int64_t)0};
    int64_t output[22] = {(int64_t)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int64_t output[22] = {(int64_t)0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_BOOL_UNSUPPORT)
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

TEST_F(TEST_RIGHTSHIFT_UT, INPUT2_NEGATIVE_OR_GREATER)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    int32_t input1[6] = {10, 10, 10, 10, 10, 10};
    int32_t input2[6] = {2, -2, 32};
    int32_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[6] = {0};
    RightCalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT16_SUCC)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int16_t input1[6] = {100, 50, 25, 12, 8, 4};
    const int16_t input2[6] = {2, 3, 1, 2, 1, 2};
    int16_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[6] = {25, 6, 12, 3, 4, 1};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT64_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int64_t input1[6] = {1000, 500, 250, 125, 62, 31};
    const int64_t input2[6] = {3, 4, 2, 3, 1, 2};
    int64_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[6] = {125, 31, 62, 15, 31, 7};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT8_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const uint8_t input1[6] = {200, 100, 50, 25, 12, 6};
    const uint8_t input2[6] = {2, 3, 1, 2, 1, 1};
    uint8_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[6] = {50, 12, 25, 6, 6, 3};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, LARGE_DATA_NOBCAST_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64, 32}, {64, 32}, {64, 32}};

    constexpr uint64_t input1_size = 64 * 32;
    int32_t input1[input1_size];
    for (int64_t i = 0; i < input1_size; ++i) {
        input1[i] = static_cast<int32_t>((i + 100) % 1000);
    }

    constexpr uint64_t input2_size = 64 * 32;
    int32_t input2[input2_size];
    for (int64_t i = 0; i < input2_size; ++i) {
        input2[i] = static_cast<int32_t>((i + 1) % 4);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    int32_t* in1_clamped = new int32_t[input2_size];
    for (int64_t i = 0; i < input2_size; i++) {
        in1_clamped[i] = input2[i];
        if (in1_clamped[i] < 0) {
            in1_clamped[i] = 0;
        } else if (in1_clamped[i] > static_cast<int32_t>(sizeof(int32_t) * CHAR_BIT) - 1) {
            in1_clamped[i] = static_cast<int32_t>(sizeof(int32_t) * CHAR_BIT) - 1;
        }
    }
    for (int64_t i = 0; i < output_size; ++i) {
        output_exp[i] = input1[i] >> in1_clamped[i];
    }
    delete[] in1_clamped;

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, LARGE_DATA_BCAST_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64, 32}, {32}, {64, 32}};

    constexpr uint64_t input1_size = 64 * 32;
    int32_t input1[input1_size];
    for (int64_t i = 0; i < input1_size; ++i) {
        input1[i] = static_cast<int32_t>((i + 100) % 1000);
    }

    constexpr uint64_t input2_size = 32;
    int32_t input2[input2_size];
    for (int64_t i = 0; i < input2_size; ++i) {
        input2[i] = static_cast<int32_t>((i + 1) % 4);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    int32_t* in1_clamped = new int32_t[input2_size];
    for (int64_t i = 0; i < input2_size; i++) {
        in1_clamped[i] = input2[i];
        if (in1_clamped[i] < 0) {
            in1_clamped[i] = 0;
        } else if (in1_clamped[i] > static_cast<int32_t>(sizeof(int32_t) * CHAR_BIT) - 1) {
            in1_clamped[i] = static_cast<int32_t>(sizeof(int32_t) * CHAR_BIT) - 1;
        }
    }
    for (int64_t i = 0; i < output_size; ++i) {
        int64_t idx = i % input2_size;
        output_exp[i] = input1[i] >> in1_clamped[idx];
    }
    delete[] in1_clamped;

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}