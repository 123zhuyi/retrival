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

class TEST_INVERT_PERMUTATION_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
    NodeDefBuilder(node_def.get(), "InvertPermutation", "InvertPermutation") \
        .Input({"x", data_types[0], shapes[0], datas[0]})                    \
        .Output({"y", data_types[1], shapes[1], datas[1]})

template <typename T1, typename T2>
void RunInvertPermutationKernel(vector<DataType> data_types, vector<vector<int64_t>>& shapes, const T1* input_data, 
    const T2* output_exp_data)
{
    uint64_t input_size = CalTotalElements(shapes, 0);
    T1 input[input_size];
    for (uint64_t i = 0; i < input_size; ++i) {
        input[i] = input_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 1);
    T2 output[output_size];
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T2 output_exp[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, DATA_TYPE_INT32_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{32}, {32}};
    const int32_t input_data[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                                  15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
    const int32_t output_exp_data[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                                       15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
    RunInvertPermutationKernel<int32_t, int32_t>(data_types, shapes, input_data, output_exp_data);
}

// exception test instance
TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT16, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}};
    int32_t input[4] = {(int16_t)1};
    int32_t output[4] = {(int32_t)1};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_DIM_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 4}, {4}};
    int32_t input[8] = {(int32_t)1};
    int32_t output[4] = {(int32_t)1};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_VALUE_RANGE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}};
    int32_t input[4];
    input[0] = 4;
    input[1] = 0;
    input[2] = 2;
    input[3] = 3;
    int32_t output[4] = {(int32_t)1};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_ELEMENT_DUPLICATED_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}};
    int32_t input[4];
    input[0] = 1;
    input[1] = 0;
    input[2] = 1;
    input[3] = 3;
    int32_t output[4] = {(int32_t)1};
    vector<void*> datas = {(void*)input, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}