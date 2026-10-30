/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in root of the software repository for the full text of the License.
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

class TEST_SUB_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sub_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sub_test TearDown\n" << endl;
    }
};

template <typename T>
void SubCalcExpectWithSameShape(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();

    if (input0_num == input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            expect_out[j] = input0_data[j] - input1_data[j];
        }
    }
}

template <typename T>
void SubCalcExpectWithDiffShape(const NodeDef& node_def, T expect_out[])
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
            expect_out[j] = input0_data[j] - input1_data[i];
        }
    } else {
        for (int64_t j = 0; j < input1_num; ++j) {
            int64_t i = j % input0_num;
            expect_out[j] = input0_data[i] - input1_data[j];
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Sub", "Sub")       \
        .Input({"x1", data_types[0], shapes[0], datas[0]})            \
        .Input({"x2", data_types[1], shapes[1], datas[1]})            \
        .Output({"y", data_types[2], shapes[2], datas[2]})

template <typename T1, typename T2, typename T3>
void RunSubKernel(vector<DataType> data_types, vector<vector<int64_t>>& shapes)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];
    SetRandomValue<T1>(input1, input1_size);

    uint64_t input2_size = CalTotalElements(shapes, 1);
    T2* input2 = new T2[input2_size];
    SetRandomValue<T2>(input2, input2_size);

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T3* output_exp = new T3[output_size];
    if (input1_size == input2_size) {
        SubCalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
    } else {
        SubCalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_SUB_UT, DATA_TYPE_FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const float input1[4] = {10.5f, 20.3f, 30.7f, 40.1f};
    const float input2[4] = {1.5f, 2.3f, 3.7f, 4.1f};
    float output[4] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float output_exp[4] = {9.0f, 18.0f, 27.0f, 36.0f};
    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const double input1[4] = {10.5, 20.3, 30.7, 40.1};
    const double input2[4] = {1.5, 2.3, 3.7, 4.1};
    double output[4] = {0};
    double output_exp[4] = {9.0, 18.0, 27.0, 36.0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const Eigen::half input1[4] = {Eigen::half(10.5f), Eigen::half(20.3f), Eigen::half(30.7f), Eigen::half(40.1f)};
    const Eigen::half input2[4] = {Eigen::half(1.5f), Eigen::half(2.3f), Eigen::half(3.7f), Eigen::half(4.1f)};
    Eigen::half output[4] = {Eigen::half(0)};
    Eigen::half output_exp[4] = {Eigen::half(9.0f), Eigen::half(18.0f), Eigen::half(27.0f), Eigen::half(36.0f)};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT8_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const int8_t input1[4] = {10, 20, 30, 40};
    const int8_t input2[4] = {1, 2, 3, 4};
    int8_t output[4] = {0};
    int8_t output_exp[4] = {9, 18, 27, 36};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_UINT8_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const uint8_t input1[4] = {10, 20, 30, 40};
    const uint8_t input2[4] = {1, 2, 3, 4};
    uint8_t output[4] = {0};
    uint8_t output_exp[4] = {9, 18, 27, 36};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT16_SUCC)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const int16_t input1[4] = {100, 200, 300, 400};
    const int16_t input2[4] = {10, 20, 30, 40};
    int16_t output[4] = {0};
    int16_t output_exp[4] = {90, 180, 270, 360};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_UINT16_SUCC)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const uint16_t input1[4] = {100, 200, 300, 400};
    const uint16_t input2[4] = {10, 20, 30, 40};
    uint16_t output[4] = {0};
    uint16_t output_exp[4] = {90, 180, 270, 360};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT32_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const int32_t input1[4] = {1000, 2000, 3000, 4000};
    const int32_t input2[4] = {100, 200, 300, 400};
    int32_t output[4] = {0};
    int32_t output_exp[4] = {900, 1800, 2700, 3600};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT64_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const int64_t input1[4] = {10000, 20000, 30000, 40000};
    const int64_t input2[4] = {1000, 2000, 3000, 4000};
    int64_t output[4] = {0};
    int64_t output_exp[4] = {9000, 18000, 27000, 36000};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, BROADCAST_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 4}, {4}, {2, 4}};
    const float input1[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    const float input2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[8] = {0};
    float output_exp[8] = {9.0f, 18.0f, 27.0f, 36.0f, 49.0f, 58.0f, 67.0f, 76.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 8);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, X_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {4}, {4}};
    const float input1[1] = {10.0f};
    const float input2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    float output_exp[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, Y_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4}, {}, {4}};
    const float input1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float input2[1] = {1.0f};
    float output[4] = {0};
    float output_exp[4] = {9.0f, 19.0f, 29.0f, 39.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, TWO_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    const float input1[1] = {10.0f};
    const float input2[1] = {1.0f};
    float output[1] = {0};
    float output_exp[1] = {9.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 1);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT32_SUCC2)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    RunSubKernel<int32_t, int32_t, int32_t>(data_types, shapes);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT64_SUCC2)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
    RunSubKernel<int64_t, int64_t, int64_t>(data_types, shapes);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT64_DIFFTRENT_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
    uint64_t input1[6] = {100, 3, 20, 14, 50, 100};
    uint64_t input2[6] = {2, 3, 4};
    uint64_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[6] = {98, 0, 16, 12, 48, 96};
    SubCalcExpectWithDiffShape<uint64_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, DATA_TYPE_INT64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    uint64_t input1[6] = {100, 3, 9, 14, 6, 8};
    uint64_t input2[6] = {3, 5, 2, 3, 1, 2};
    uint64_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[6] = {97, 0, 7, 11, 5, 6};
    SubCalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, INPUT_SHAPE_EXCEPTION)
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

TEST_F(TEST_SUB_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int64_t output[22] = {(int64_t)0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SUB_UT, LARGE_DATA_NOBCAST_PARALLEL_SUCC)
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
        input2[i] = static_cast<int32_t>((i + 1) % 10);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    for (int64_t i = 0; i < output_size; ++i) {
        output_exp[i] = input1[i] - input2[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SUB_UT, LARGE_DATA_BCAST_PARALLEL_SUCC)
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
        input2[i] = static_cast<int32_t>((i + 1) % 10);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    for (int64_t i = 0; i < output_size; ++i) {
        int64_t idx = i % input2_size;
        output_exp[i] = input1[i] - input2[idx];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}
