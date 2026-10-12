/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <complex>

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

using namespace std;
using namespace aicpu;

class TEST_RSQRT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
    auto node_def = CpuKernelUtils::CreateNodeDef();      \
    NodeDefBuilder(node_def.get(), "Rsqrt", "Rsqrt")                      \
        .Input({"x", data_types[0], shapes[0], datas[0]})                 \
        .Output({"y", data_types[1], shapes[1], datas[1]})

// -------------------------------------------------------------------------
// Success cases: DT_FLOAT
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
    // input: 1.0 ~ 25.0, expected: 1/sqrt(i)
    float input[25];
    for (int i = 0; i < 25; i++) {
        input[i] = static_cast<float>(i + 1);
    }
    float output[25] = {0.0f};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float output_exp[25];
    for (int i = 0; i < 25; i++) {
        output_exp[i] = 1.0f / std::sqrt(static_cast<float>(i + 1));
    }
    EXPECT_EQ(CompareResult(output, output_exp, 25), true);
}

// -------------------------------------------------------------------------
// Success cases: DT_DOUBLE
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
    double input[25];
    for (int i = 0; i < 25; i++) {
        input[i] = static_cast<double>(i + 1);
    }
    double output[25] = {0.0};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    double output_exp[25];
    for (int i = 0; i < 25; i++) {
        output_exp[i] = 1.0 / std::sqrt(static_cast<double>(i + 1));
    }
    EXPECT_EQ(CompareResult(output, output_exp, 25), true);
}

// -------------------------------------------------------------------------
// Success cases: DT_FLOAT16
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
    Eigen::half input[25];
    for (int i = 0; i < 25; i++) {
        input[i] = static_cast<Eigen::half>(i + 1);
    }
    Eigen::half output[25];
    for (int i = 0; i < 25; i++) {
        output[i] = static_cast<Eigen::half>(0);
    }
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half output_exp[25];
    for (int i = 0; i < 25; i++) {
        output_exp[i] = static_cast<Eigen::half>(1.0f / std::sqrt(static_cast<float>(i + 1)));
    }
    EXPECT_EQ(CompareResult(output, output_exp, 25), true);
}

// -------------------------------------------------------------------------
// Success cases: DT_FLOAT multicore (>8192 elements triggers parallel path)
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT_MULTICORE_SUCC)
{
    constexpr int N = 9 * 1024;  // 9216 > 8192, triggers parallel path
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{9, 1024}, {9, 1024}};
    float *input = new float[N];
    float *output = new float[N];
    float *output_exp = new float[N];
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 100 + 1);
        output[i] = 0.0f;
        output_exp[i] = 1.0f / std::sqrt(static_cast<float>(i % 100 + 1));
    }
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult(output, output_exp, N), true);
    delete[] input;
    delete[] output;
    delete[] output_exp;
}

// -------------------------------------------------------------------------
// Success cases: DT_COMPLEX64
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX64_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
    // Use real-only complex values: rsqrt(a+0i) = 1/sqrt(a) + 0i
    std::complex<float> input[25];
    for (int i = 0; i < 25; i++) {
        input[i] = std::complex<float>(static_cast<float>(i + 1), 0.0f);
    }
    std::complex<float> output[25] = {};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<float> output_exp[25];
    for (int i = 0; i < 25; i++) {
        float a = static_cast<float>(i + 1);
        // rsqrt(a+0i) = sqrt(conj(a+0i)) / |a+0i| = sqrt(a) / a = 1/sqrt(a)
        output_exp[i] = std::complex<float>(1.0f / std::sqrt(a), 0.0f);
    }
    EXPECT_EQ(CompareResult(output, output_exp, 25), true);
}

// -------------------------------------------------------------------------
// Success cases: DT_COMPLEX128
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX128_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
    std::complex<double> input[25];
    for (int i = 0; i < 25; i++) {
        input[i] = std::complex<double>(static_cast<double>(i + 1), 0.0);
    }
    std::complex<double> output[25] = {};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<double> output_exp[25];
    for (int i = 0; i < 25; i++) {
        double a = static_cast<double>(i + 1);
        output_exp[i] = std::complex<double>(1.0 / std::sqrt(a), 0.0);
    }
    EXPECT_EQ(CompareResult(output, output_exp, 25), true);
}

// -------------------------------------------------------------------------
// Success cases: Complex multicore (>4096 elements triggers parallel path)
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX64_MULTICORE_SUCC)
{
    constexpr int N = 5 * 1024;  // 5120 > 4096, triggers complex parallel path
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{5, 1024}, {5, 1024}};
    std::complex<float> *input = new std::complex<float>[N];
    std::complex<float> *output = new std::complex<float>[N];
    std::complex<float> *output_exp = new std::complex<float>[N];
    for (int i = 0; i < N; i++) {
        float a = static_cast<float>(i % 100 + 1);
        input[i] = std::complex<float>(a, 0.0f);
        output[i] = std::complex<float>(0.0f, 0.0f);
        output_exp[i] = std::complex<float>(1.0f / std::sqrt(a), 0.0f);
    }
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult(output, output_exp, N), true);
    delete[] input;
    delete[] output;
    delete[] output_exp;
}

// -------------------------------------------------------------------------
// Success cases: zero inputs (rsqrt(0) = inf, kernel should not crash)
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, INPUT_ZERO_FLOAT_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
    float input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {0.0f};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_RSQRT_UT, INPUT_ZERO_DOUBLE_SUCCESS)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
    double input[4] = {0.0, 0.0, 0.0, 0.0};
    double output[4] = {0.0};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_RSQRT_UT, INPUT_ZERO_FLOAT16_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
    Eigen::half input[4] = {static_cast<Eigen::half>(0)};
    Eigen::half output[4] = {static_cast<Eigen::half>(0)};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// -------------------------------------------------------------------------
// Exception cases
// -------------------------------------------------------------------------
TEST_F(TEST_RSQRT_UT, INPUT_SHAPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 10}, {2, 11}};
    double input[20] = {1.0};
    double output[22] = {0.0};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
    double input[22] = {1.0};
    float output[22] = {0.0f};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
    double output[22] = {0.0};
    vector<void*> datas = {(void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, OUTPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
    double input[22] = {1.0};
    vector<void*> datas = {(void*)input, (void*)nullptr};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_BOOL_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
    bool input[22] = {true};
    bool output[22] = {false};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
