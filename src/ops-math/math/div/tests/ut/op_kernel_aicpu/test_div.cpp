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

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_DIV_UT : public testing::Test {};

#define CREATE_DIV_NODEDEF(shapes, data_types, datas)      \
    auto node_def = CpuKernelUtils::CreateNodeDef();       \
    NodeDefBuilder(node_def.get(), "Div", "Div")           \
        .Input({"x1", data_types[0], shapes[0], datas[0]}) \
        .Input({"x2", data_types[1], shapes[1], datas[1]}) \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_DIV_UT, IntFloorDivisionSameShape)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    int32_t input1[4] = {-7, -5, 7, 5};
    int32_t input2[4] = {2, -2, -2, 2};
    int32_t output[4] = {0};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expected[4] = {-4, 2, -4, 2};
    EXPECT_EQ(CompareResult(output, expected, 4), true);
}

TEST_F(TEST_DIV_UT, IntBroadcastDivScalar)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{4}, {1}, {4}};
    int16_t input1[4] = {12, 5, -12, -5};
    int16_t input2[1] = {3};
    int16_t output[4] = {0};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t expected[4] = {4, 1, -4, -2};
    EXPECT_EQ(CompareResult(output, expected, 4), true);
}

TEST_F(TEST_DIV_UT, FloatBroadcastScalarDivVector)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {4}, {4}};
    float input1[1] = {8.0f};
    float input2[4] = {16.0f, 8.0f, 4.0f, 2.0f};
    float output[4] = {0.0f};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expected[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    EXPECT_EQ(CompareResult(output, expected, 4), true);
}

TEST_F(TEST_DIV_UT, ComplexSameShape)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    std::complex<float> input1[2] = {{4.0f, 2.0f}, {3.0f, -3.0f}};
    std::complex<float> input2[2] = {{2.0f, 0.0f}, {1.0f, -1.0f}};
    std::complex<float> output[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::complex<float> expected[2] = {{2.0f, 1.0f}, {3.0f, 0.0f}};
    EXPECT_EQ(CompareResult(output, expected, 2), true);
}

TEST_F(TEST_DIV_UT, InvalidBroadcastShape)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    int32_t input1[16] = {0};
    int32_t input2[16] = {0};
    int32_t output[16] = {0};
    std::fill_n(input1, 16, 1);
    std::fill_n(input2, 16, 1);
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIV_UT, IntegerZeroDivisorReturnsError)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    uint16_t input1[4] = {1, 2, 3, 4};
    uint16_t input2[4] = {1, 0, 2, 3};
    uint16_t output[4] = {0};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, IntegerMinDivMinusOneReturnsErrorSameShape)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    int32_t input1[2] = {std::numeric_limits<int32_t>::min(), 8};
    int32_t input2[2] = {-1, 2};
    int32_t output[2] = {0};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, IntegerMinDivMinusOneReturnsErrorBroadcastScalar)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {1}, {4}};
    int32_t input1[4] = {1, std::numeric_limits<int32_t>::min(), 3, 4};
    int32_t input2[1] = {-1};
    int32_t output[4] = {0};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, FloatZeroDivisorKeepsOkStatus)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    float input1[1] = {1.0f};
    float input2[1] = {0.0f};
    float output[1] = {0.0f};
    vector<void*> datas = {input1, input2, output};

    CREATE_DIV_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_TRUE(std::isinf(output[0]));
    EXPECT_GT(output[0], 0.0f);
}