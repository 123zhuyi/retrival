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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_COMPARE_AND_BIT_PACK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
    auto node_def = CpuKernelUtils::CreateNodeDef();                         \
    NodeDefBuilder(node_def.get(), "CompareAndBitpack", "CompareAndBitpack") \
        .Input({"x", data_types[0], shapes[0], datas[0]})                    \
        .Input({"threshold", data_types[1], shapes[1], datas[1]})            \
        .Output({"output", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_COMPLEX64_UNSUPPORTED)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    std::complex<float> input0[16] = {2.6, 4.9, 3.5, 2.1, 1.9, 1.8, 4.5, 2.9, 2.6, 1.2, 1.6, 4.1, 1.5, 2.2, 4.3, 1.5};
    std::complex<float> input1 = 3.2;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_FLOAT16_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    Eigen::half input0[16] = {(Eigen::half)2.6, (Eigen::half)4.9, (Eigen::half)3.5, (Eigen::half)2.1,
                              (Eigen::half)1.9, (Eigen::half)1.8, (Eigen::half)4.5, (Eigen::half)2.9,
                              (Eigen::half)2.6, (Eigen::half)1.2, (Eigen::half)1.6, (Eigen::half)4.1,
                              (Eigen::half)1.5, (Eigen::half)2.2, (Eigen::half)4.3, (Eigen::half)1.5};
    Eigen::half input1 = (Eigen::half)3.2;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_BOOL_SUCCESS)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    bool input0[16] = {false, true,  true,  false, false, false, true, false,
                       false, false, false, true,  false, false, true, false};
    bool input1 = false;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_FLOAT_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    float input0[16] = {2.6, 4.9, 3.5, 2.1, 1.9, 1.8, 4.5, 2.9, 2.6, 1.2, 1.6, 4.1, 1.5, 2.2, 4.3, 1.5};
    float input1 = 3.2;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_INT8_SUCCESS)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    int8_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4, 1};
    int8_t input1 = 3;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_INT16_SUCCESS)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    int16_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4, 1};
    int16_t input1 = 3;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_INT32_SUCCESS)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    int32_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4, 1};
    int32_t input1 = 3;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_INT64_SUCCESS)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    int64_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4, 1};
    int64_t input1 = 3;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, DOUBLE_SUCCESS)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211};
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[2] = {98, 18};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_0_EMPTY)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211};
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)nullptr, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_1_EMPTY)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211};
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, OUTPUT_EMPTY)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
    double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211};
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)nullptr, (void*)nullptr};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_1_NOT_SCALAR)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 16}, {1, 2}, {1, 2}};
    double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211};
    double input1[2] = {3.20928929, 20928929};
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_0_SCALAR)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{}, {}, {1, 2}};
    double input0 = 2.69160455;
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)&input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BIT_PACK_UT, INPUT_0_DIM_ERROR)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 17}, {}, {1, 2}};
    double input0[17] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                         4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                         1.53567731, 2.29360969, 4.3461758,  1.57004211, 2.57004211};
    double input1 = 3.20928929;
    uint8_t output[2] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
