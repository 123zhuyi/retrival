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

class TEST_SQUAREDDIFFERENCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
    NodeDefBuilder(node_def.get(), "SquaredDifference", "SquaredDifference") \
        .Input({"x1", data_types[0], shapes[0], datas[0]})                   \
        .Input({"x2", data_types[1], shapes[1], datas[1]})                   \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_X_NUM_ONE_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {1, 3}, {1, 3}};

    constexpr uint64_t input1_size = 1;
    int32_t input1[input1_size] = {2};

    constexpr uint64_t input2_size = 3;
    int32_t input2[input2_size] = {7, 0, 4};

    constexpr uint64_t output_size = 3;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {25, 4, 4};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_Y_NUM_ONESUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 3}, {1}, {1, 3}};

    constexpr uint64_t input1_size = 3;
    int32_t input1[input1_size] = {2, 7, 0};

    constexpr uint64_t input2_size = 1;
    int32_t input2[input2_size] = {4};

    constexpr uint64_t output_size = 3;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {4, 9, 16};
    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4, 16}, {1, 16}, {4, 16}};

    constexpr uint64_t input1_size = 4 * 16;
    int32_t input1[input1_size] = {2, 7, 0, 4, 2, 1, 3, 3, 3, 9, 1, 2, 0, 7, 0, 5, 3, 9, 4, 7, 8, 5,
                                   6, 6, 7, 7, 6, 4, 0, 1, 5, 7, 5, 3, 7, 0, 8, 9, 8, 9, 6, 3, 7, 6,
                                   9, 5, 9, 4, 4, 2, 7, 2, 1, 9, 6, 8, 8, 9, 6, 2, 7, 9, 1, 2};

    constexpr uint64_t input2_size = 16;
    int32_t input2[input2_size] = {6, 3, 1, 2, 5, 6, 4, 8, 6, 6, 1, 8, 5, 7, 2, 3};

    constexpr uint64_t output_size = 4 * 16;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {16, 16, 1,  4,  9,  25, 1,  25, 9, 9, 0,  36, 25, 0,  4,  4,
                                       9,  36, 9,  25, 9,  1,  4,  4,  1, 1, 25, 16, 25, 36, 9,  16,
                                       1,  0,  36, 4,  9,  9,  16, 1,  0, 9, 36, 4,  16, 4,  49, 1,
                                        4,  1,  36, 0,  16, 9,  4,  0,  4, 9, 25, 36, 4,  4,  1,  1};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    constexpr uint64_t input1_size = 6;
    Eigen::half input1[input1_size] = {Eigen::half(1.0f), Eigen::half(2.0f), Eigen::half(3.0f),
                                      Eigen::half(4.0f), Eigen::half(5.0f), Eigen::half(6.0f)};

    constexpr uint64_t input2_size = 6;
    Eigen::half input2[input2_size] = {Eigen::half(0.5f), Eigen::half(1.5f), Eigen::half(2.5f),
                                      Eigen::half(3.5f), Eigen::half(4.5f), Eigen::half(5.5f)};

    constexpr uint64_t output_size = 6;
    Eigen::half output[output_size] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half output_exp[output_size] = {Eigen::half(0.25f), Eigen::half(0.25f), Eigen::half(0.25f),
                                         Eigen::half(0.25f), Eigen::half(0.25f), Eigen::half(0.25f)};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    constexpr uint64_t input1_size = 6;
    float input1[input1_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    constexpr uint64_t input2_size = 6;
    float input2[input2_size] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

    constexpr uint64_t output_size = 6;
    float output[output_size] = {0.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float output_exp[output_size] = {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    constexpr uint64_t input1_size = 6;
    double input1[input1_size] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    constexpr uint64_t input2_size = 6;
    double input2[input2_size] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};

    constexpr uint64_t output_size = 6;
    double output[output_size] = {0.0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    double output_exp[output_size] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INT64_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    constexpr uint64_t input1_size = 6;
    int64_t input1[input1_size] = {10, 20, 30, 40, 50, 60};

    constexpr uint64_t input2_size = 6;
    int64_t input2[input2_size] = {5, 15, 25, 35, 45, 55};

    constexpr uint64_t output_size = 6;
    int64_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[output_size] = {25, 25, 25, 25, 25, 25};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, COMPLEX64_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};

    constexpr uint64_t input1_size = 4;
    std::complex<float> input1[input1_size] = {
        std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f),
        std::complex<float>(5.0f, 6.0f), std::complex<float>(7.0f, 8.0f)
    };

    constexpr uint64_t input2_size = 4;
    std::complex<float> input2[input2_size] = {
        std::complex<float>(0.5f, 1.0f), std::complex<float>(1.5f, 2.0f),
        std::complex<float>(2.5f, 3.0f), std::complex<float>(3.5f, 4.0f)
    };

    constexpr uint64_t output_size = 4;
    std::complex<float> output[output_size] = {std::complex<float>(0.0f, 0.0f)};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<float> output_exp[output_size] = {
        std::complex<float>(1.25f, 0.0f), std::complex<float>(6.25f, 0.0f),
        std::complex<float>(15.25f, 0.0f), std::complex<float>(28.25f, 0.0f)
    };

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, COMPLEX128_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};

    constexpr uint64_t input1_size = 4;
    std::complex<double> input1[input1_size] = {
        std::complex<double>(1.0, 2.0), std::complex<double>(3.0, 4.0),
        std::complex<double>(5.0, 6.0), std::complex<double>(7.0, 8.0)
    };

    constexpr uint64_t input2_size = 4;
    std::complex<double> input2[input2_size] = {
        std::complex<double>(0.5, 1.0), std::complex<double>(1.5, 2.0),
        std::complex<double>(2.5, 3.0), std::complex<double>(3.5, 4.0)
    };

    constexpr uint64_t output_size = 4;
    std::complex<double> output[output_size] = {std::complex<double>(0.0, 0.0)};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<double> output_exp[output_size] = {
        std::complex<double>(1.25, 0.0), std::complex<double>(6.25, 0.0),
        std::complex<double>(15.25, 0.0), std::complex<double>(28.25, 0.0)
    };

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, LARGE_DATA_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64, 32}, {64, 32}, {64, 32}};

    constexpr uint64_t input1_size = 64 * 32;
    int32_t input1[input1_size];
    for (int64_t i = 0; i < input1_size; ++i) {
        input1[i] = static_cast<int32_t>(i % 10);
    }

    constexpr uint64_t input2_size = 64 * 32;
    int32_t input2[input2_size];
    for (int64_t i = 0; i < input2_size; ++i) {
        input2[i] = static_cast<int32_t>((i + 5) % 10);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    for (int64_t i = 0; i < output_size; ++i) {
        int32_t expected = (input1[i] - input2[i]) * (input1[i] - input2[i]);
        EXPECT_EQ(output[i], expected);
    }
}
