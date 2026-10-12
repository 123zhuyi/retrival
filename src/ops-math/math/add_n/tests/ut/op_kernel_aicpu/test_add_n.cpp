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

using namespace std;
using namespace aicpu;

class TEST_ADD_N_UT : public testing::Test {};

#define CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n)        \
    auto node_def = CpuKernelUtils::CreateNodeDef();             \
    NodeDefBuilder builder(node_def.get(), "AddN", "AddN");       \
    for (int i = 0; i < n; i++) {                                \
        builder.Input({"x" + std::to_string(i), data_types[i], shapes[i], datas[i]}); \
    }                                                              \
    builder.Output({"y", data_types[n], shapes[n], datas[n]});   \
    builder.Attr("N", n);

TEST_F(TEST_ADD_N_UT, INT8_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};

    int8_t input1[2] = {1, 2};
    int8_t input2[2] = {3, 4};
    int8_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int8_t output_exp[2] = {4, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, INT16_THREE_INPUTS_ADD_N_SUCC)
{
    int n = 3;
    vector<vector<int64_t>> shapes = {{3}, {3}, {3}, {3}};
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16, DT_INT16};

    int16_t input1[3] = {1, 2, 3};
    int16_t input2[3] = {4, 5, 6};
    int16_t input3[3] = {7, 8, 9};
    int16_t output[3] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)input3, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[3] = {12, 15, 18};
    bool compare = CompareResult(output, output_exp, 3);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, INT32_FOUR_INPUTS_ADD_N_SUCC)
{
    int n = 4;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}, {2}};
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};

    int32_t input1[2] = {1, 2};
    int32_t input2[2] = {3, 4};
    int32_t input3[2] = {5, 6};
    int32_t input4[2] = {7, 8};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)input3, (void*)input4, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[2] = {16, 20};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, INT64_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};

    int64_t input1[4] = {1, 2, 3, 4};
    int64_t input2[4] = {5, 6, 7, 8};
    int64_t output[4] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[4] = {6, 8, 10, 12};
    bool compare = CompareResult(output, output_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, UINT8_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};

    uint8_t input1[2] = {1, 2};
    uint8_t input2[2] = {3, 4};
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {4, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, UINT16_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};

    uint16_t input1[2] = {1, 2};
    uint16_t input2[2] = {3, 4};
    uint16_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint16_t output_exp[2] = {4, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, UINT32_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};

    uint32_t input1[2] = {1, 2};
    uint32_t input2[2] = {3, 4};
    uint32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint32_t output_exp[2] = {4, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, UINT64_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};

    uint64_t input1[2] = {1, 2};
    uint64_t input2[2] = {3, 4};
    uint64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[2] = {4, 6};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, FLOAT16_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};

    Eigen::half input1[2];
    Eigen::half input2[2];
    Eigen::half output[2];
    for (int i = 0; i < 2; ++i) {
        input1[i] = (Eigen::half)(i + 1);
        input2[i] = (Eigen::half)(i + 3);
    }
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half output_exp[2];
    output_exp[0] = (Eigen::half)4;
    output_exp[1] = (Eigen::half)6;
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, FLOAT_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float input1[2] = {1.5f, 2.5f};
    float input2[2] = {3.5f, 4.5f};
    float output[2] = {0.0f};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float output_exp[2] = {5.0f, 7.0f};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, DOUBLE_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};

    double input1[2] = {1.5, 2.5};
    double input2[2] = {3.5, 4.5};
    double output[2] = {0.0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    double output_exp[2] = {5.0, 7.0};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, COMPLEX64_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};

    std::complex<float> input1[2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::complex<float> input2[2] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    std::complex<float> output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<float> output_exp[2] = {{6.0f, 8.0f}, {10.0f, 12.0f}};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, COMPLEX128_TWO_INPUTS_ADD_N_SUCC)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};

    std::complex<double> input1[2] = {{1.0, 2.0}, {3.0, 4.0}};
    std::complex<double> input2[2] = {{5.0, 6.0}, {7.0, 8.0}};
    std::complex<double> output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<double> output_exp[2] = {{6.0, 8.0}, {10.0, 12.0}};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_N_UT, INPUT_DTYPE_DISMATCH)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT64};

    int32_t input1[2] = {1, 2};
    int32_t input2[2] = {3, 4};
    int64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_N_UT, INPUT_OUTPUT_DTYPE_DISMATCH)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT64};

    int32_t input1[2] = {1, 2};
    int32_t input2[2] = {3, 4};
    int64_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_N_UT, INPUT_SHAPE_DISMATCH)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {3}};
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};

    int32_t input1[2] = {1, 2};
    int32_t input2[2] = {3, 4};
    int32_t output[3] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_N_UT, UNSUPPORTED_DTYPE_BOOL)
{
    int n = 2;
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};

    bool input1[2] = {true, false};
    bool input2[2] = {false, true};
    bool output[2] = {false};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF_ADD_N(shapes, data_types, datas, n);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
