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

class TEST_ADD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Add", "Add")                     \
        .Input({"x1", data_types[0], shapes[0], datas[0]})           \
        .Input({"x2", data_types[1], shapes[1], datas[1]})           \
        .Output({"y", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ADD_UT, INT8_SCALAR_ADD_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{}, {}, {}};

    int8_t input1[1] = {1};
    int8_t input2[1] = {2};
    int8_t output[1] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int8_t output_exp[1] = {3};
    bool compare = CompareResult(output, output_exp, 1);
}

TEST_F(TEST_ADD_UT, INT16_SCALAR_ADD_VECTOR_SUCC)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{}, {2}, {2}};

    int16_t input1[1] = {1};
    int16_t input2[2] = {2, 3};
    int16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[2] = {3, 4};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, FLOAT_SCALAR_ADD_COMPLEX_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{1}, {2}, {2}};

    std::complex<float> input1[1] = {{1.2, 3.2}};
    std::complex<float> input2[2] = {{2.6, 3}, {3.5, 5.2}};
    std::complex<float> output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<float> output_exp[2] = {{3.8, 6.2}, {4.7, 8.4}};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, DOUBLE_SCALAR_ADD_COMPLEX_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{1}, {2}, {2}};

    std::complex<double> input1[1] = {{3.2, 1.2}};
    std::complex<double> input2[2] = {{4.6, 5.8}, {2.5, 4.2}};
    std::complex<double> output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<double> output_exp[2] = {{7.8, 7.0}, {5.7, 5.4}};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT32_VECTOR_ADD_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    int32_t input1[2] = {2, 5};
    int32_t input2[1] = {3};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[2] = {5, 8};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT64_VECTOR_ADD_VECTOR_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};

    int64_t input1[2] = {2, 5};
    int64_t input2[2] = {3, 5};
    int64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[2] = {5, 10};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT64_VECTOR_ADD_VECTOR_BROADCAST_0)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1}, {2, 1, 1, 1}, {2, 1, 1, 1}};

    int64_t input1[1] = {2};
    int64_t input2[2] = {3, 5};
    int64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[2] = {5, 7};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_BROADCAST_1)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 1, 1, 1}, {1}, {2, 1, 1, 1}};

    int16_t input1[2] = {2, 3};
    int16_t input2[1] = {3};
    int16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[2] = {5, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_DIM_7)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1}};

    int16_t input1[2] = {2, 3};
    int16_t input2[1] = {3};
    int16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[2] = {5, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_DIM_8)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1, 1}};

    int16_t input1[2] = {2, 3};
    int16_t input2[1] = {3};
    int16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[2] = {5, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_INVALID_DIM)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}};

    int16_t input1[2] = {2, 3};
    int16_t input2[1] = {3};
    int16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, INT32_VECTOR_ADD_VECTOR_BROADCAST_BOTH)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 2}};

    int32_t input1[2] = {2, 3};
    int32_t input2[2] = {3, 10};
    int32_t output[4] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[4] = {5, 6, 12, 13};
    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, UNDEFINED_DTYPE_EXCEPTION)
{
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_UNDEFINED, DT_UNDEFINED, DT_UNDEFINED};
    vector<int32_t> input1 = {1};
    vector<int32_t> input2 = {1};
    vector<int32_t> output = {0};
    vector<void*> datas = {(void*)input1.data(), (void*)input2.data(), (void*)output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DTYPE_BOOL_UNSORPORT)
{
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    bool input1[1] = {false};
    bool input2[1] = {true};
    bool output[1] = {false};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, INPUT_DTYPE_DISMATCH)
{
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
    float input1[1] = {1.0F};
    double input2[1] = {2.0};
    double output[1] = {0.0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_MAX_SIZE_DISMATCH)
{
    vector<vector<int64_t>> shapes = {{1}, {1}, {1, 1}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[1] = {1.0F};
    float input2[1] = {2.0F};
    float output[1] = {0.0F};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_SHAPE_DISMATCH)
{
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 1}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[2] = {1.0F};
    float input2[3] = {2.0F};
    float output[1] = {0.0F};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_NOT_SUPPORT)
{
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 3}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[2] = {1.0F};
    float input2[3] = {2.0F};
    float output[3] = {0.0F};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_TYPE)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    int32_t input1[2] = {2, 5};
    int64_t input2[1] = {3};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_OUTPUT_TYPE)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    int32_t input1[2] = {2, 5};
    int32_t input2[1] = {3};
    int64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_BOOL_TYPE)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    bool input1[2] = {1, 1};
    bool input2[1] = {1};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_BIG_OUTPUT_SMALL_INT32)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    int64_t input1[2] = {2, 5};
    int64_t input2[1] = {3};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_BIG_OUTPUT_SMALL_FLOAT)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    float input1[2] = {2, 5};
    float input2[1] = {3};
    Eigen::half output[2];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DIFFERENT_INPUT_SMALL_OUTPUT_BIG_FLOAT)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    Eigen::half input1[2];
    for (int i = 0; i < 2; ++i) {
        input1[i] = (Eigen::half)i;
    }
    Eigen::half input2[1];
    input2[0] = (Eigen::half)2;
    float output[2];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ------------------------------------------------------------------
// Additional UTs for the optimized implementation.
// - Cover same-shape fast path at parallel scale (>= 192KB output).
// - Cover scalar-bcast fast path at parallel scale.
// - Cover multi-dim broadcast fallback (generic Eigen path) for
//   numerical equivalence with the original implementation.
// ------------------------------------------------------------------

TEST_F(TEST_ADD_UT, FLOAT_SAME_SHAPE_LARGE_PARALLEL)
{
    // 96K elements * 4B = 384KB > 192KB threshold -> parallel path
    const int64_t n = 96 * 1024;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{n}, {n}, {n}};

    vector<float> in0(n);
    vector<float> in1(n);
    vector<float> out(n, 0.0f);
    vector<float> exp(n);
    for (int64_t i = 0; i < n; ++i) {
        in0[i] = static_cast<float>(i) * 0.5f;
        in1[i] = static_cast<float>(i) * 0.25f + 1.0f;
        exp[i] = in0[i] + in1[i];
    }
    vector<void*> datas = {(void*)in0.data(), (void*)in1.data(), (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(out.data(), exp.data(), n);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT32_SAME_SHAPE_SMALL_SERIAL)
{
    // 16 elements -> well below threshold, serial path
    const int64_t n = 16;
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{n}, {n}, {n}};

    vector<int32_t> in0(n), in1(n), out(n, 0), exp(n);
    for (int64_t i = 0; i < n; ++i) {
        in0[i] = static_cast<int32_t>(i);
        in1[i] = static_cast<int32_t>(i * 2);
        exp[i] = in0[i] + in1[i];
    }
    vector<void*> datas = {(void*)in0.data(), (void*)in1.data(), (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out.data(), exp.data(), n), true);
}

TEST_F(TEST_ADD_UT, DOUBLE_SCALAR_BCAST_LARGE_PARALLEL)
{
    // input0 scalar, input1 large vector. Output bytes = 32K * 8 = 256KB > thresh.
    const int64_t n = 32 * 1024;
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{1}, {n}, {n}};

    double in0[1] = {3.5};
    vector<double> in1(n), out(n, 0.0), exp(n);
    for (int64_t i = 0; i < n; ++i) {
        in1[i] = static_cast<double>(i) * 0.125;
        exp[i] = in0[0] + in1[i];
    }
    vector<void*> datas = {(void*)in0, (void*)in1.data(), (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out.data(), exp.data(), n), true);
}

TEST_F(TEST_ADD_UT, INT64_SCALAR_BCAST_RHS_LARGE)
{
    // input1 scalar, input0 large vector.
    const int64_t n = 30 * 1024;
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{n}, {1}, {n}};

    vector<int64_t> in0(n), out(n, 0), exp(n);
    int64_t in1[1] = {7};
    for (int64_t i = 0; i < n; ++i) {
        in0[i] = i;
        exp[i] = in0[i] + in1[0];
    }
    vector<void*> datas = {(void*)in0.data(), (void*)in1, (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out.data(), exp.data(), n), true);
}

TEST_F(TEST_ADD_UT, FLOAT_GENERIC_BCAST_2D)
{
    // shape {3,1} + {1,4} -> {3,4}, goes through generic Eigen path.
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 1}, {1, 4}, {3, 4}};

    float in0[3] = {1.0f, 2.0f, 3.0f};
    float in1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[12] = {0};
    float exp[12] = {11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43};
    vector<void*> datas = {(void*)in0, (void*)in1, (void*)out};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out, exp, 12), true);
}

TEST_F(TEST_ADD_UT, COMPLEX64_SAME_SHAPE_PARALLEL)
{
    // complex64 = 8B, 24K elems -> 192KB exactly at threshold boundary.
    const int64_t n = 32 * 1024;
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{n}, {n}, {n}};

    vector<std::complex<float>> in0(n), in1(n), out(n), exp(n);
    for (int64_t i = 0; i < n; ++i) {
        in0[i] = {static_cast<float>(i), static_cast<float>(-i)};
        in1[i] = {1.0f, 2.0f};
        exp[i] = in0[i] + in1[i];
    }
    vector<void*> datas = {(void*)in0.data(), (void*)in1.data(), (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out.data(), exp.data(), n), true);
}

TEST_F(TEST_ADD_UT, FLOAT16_SCALAR_BCAST_SMALL_SERIAL)
{
    // Eigen::half + small -> serial scalar-bcast.
    const int64_t n = 64;
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1}, {n}, {n}};

    Eigen::half in0[1] = {Eigen::half(1.5f)};
    vector<Eigen::half> in1(n), out(n), exp(n);
    for (int64_t i = 0; i < n; ++i) {
        in1[i] = Eigen::half(static_cast<float>(i) * 0.25f);
        exp[i] = in0[0] + in1[i];
    }
    vector<void*> datas = {(void*)in0, (void*)in1.data(), (void*)out.data()};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(CompareResult(out.data(), exp.data(), n), true);
}
