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
#include <complex>
#include <vector>

using namespace std;
using namespace aicpu;

class TEST_CONJ_UT : public testing::Test {};

#define CREATE_NODEDEF(node_def, shapes, data_types, datas)   \
    NodeDefBuilder(node_def.get(), "Conj", "Conj")            \
        .Input({"input", data_types[0], shapes[0], datas[0]}) \
        .Output({"output", data_types[1], shapes[1], datas[1]})

template <typename T>
void RunConjKernel(vector<DataType> data_types, vector<vector<int64_t>>& shapes, vector<T>& input,
                   vector<T>& output_exp)
{
    uint64_t output_size = CalTotalElements(shapes, 1);
    vector<T> output(output_size);
    vector<void*> datas = {(void*)input.data(), (void*)output.data()};

    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output.data(), output_exp.data(), output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_CONJ_UT, DATA_TYPE_COMPLEX64_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<complex<float>> input = {{1.0f, 2.0f},  {3.0f, -4.0f}, {5.0f, 6.0f},
                                    {-7.0f, 8.0f}, {0.0f, 1.0f},  {2.0f, 0.0f}};
    vector<complex<float>> output_exp = {{1.0f, -2.0f},  {3.0f, 4.0f},  {5.0f, -6.0f},
                                         {-7.0f, -8.0f}, {0.0f, -1.0f}, {2.0f, 0.0f}};
    RunConjKernel<complex<float>>(data_types, shapes, input, output_exp);
}

TEST_F(TEST_CONJ_UT, DATA_TYPE_COMPLEX128_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<complex<double>> input = {{1.5, 2.5}, {-3.5, 4.5}, {0.0, -1.0}, {2.0, 0.0}};
    vector<complex<double>> output_exp = {{1.5, -2.5}, {-3.5, -4.5}, {0.0, 1.0}, {2.0, 0.0}};
    RunConjKernel<complex<double>>(data_types, shapes, input, output_exp);
}

// large input triggers the ParallelFor branch (data_size > kParallelDataNums).
TEST_F(TEST_CONJ_UT, DATA_TYPE_COMPLEX128_BIGDATA_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{64, 1024}, {64, 1024}};
    uint64_t num = 64 * 1024;
    vector<complex<double>> input(num), output_exp(num);
    for (uint64_t i = 0; i < num; i++) {
        input[i] = complex<double>(static_cast<double>(i), -static_cast<double>(i));
        output_exp[i] = complex<double>(static_cast<double>(i), static_cast<double>(i));
    }
    RunConjKernel<complex<double>>(data_types, shapes, input, output_exp);
}

TEST_F(TEST_CONJ_UT, INPUT_DTYPE_UNSUPPORT)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    float input[6] = {0.0f};
    float output[6] = {0.0f};
    vector<void*> datas = {(void*)input, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJ_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    complex<float> output[6];
    vector<void*> datas = {(void*)nullptr, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJ_UT, OUTPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    complex<float> input[6];
    vector<void*> datas = {(void*)input, (void*)nullptr};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
