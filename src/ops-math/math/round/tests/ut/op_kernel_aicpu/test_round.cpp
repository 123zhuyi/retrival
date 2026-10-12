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
#include <cmath>
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_ROUND_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    int64_t input0_num = input0->NumElements();
    for (int64_t i = 0; i < input0_num; ++i) {
        expect_out[i] = std::round(input0_data[i]);
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas, decimals)          \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Round", "Round")                 \
        .Input({"input", data_types[0], shapes[0], datas[0]})        \
        .Output({"output", data_types[1], shapes[1], datas[1]})      \
        .Attr("decimals", decimals);

#define ADD_CASE(base_type, aicpu_type)                         \
    TEST_F(TEST_ROUND_UT, TestRound_##aicpu_type)               \
    {                                                           \
        vector<DataType> data_types = {aicpu_type, aicpu_type}; \
        vector<vector<int64_t>> shapes = {{4}, {4}};            \
        base_type input[4];                                     \
        SetRandomValue<base_type>(input, 4);                    \
        base_type output[4] = {(base_type)0};                   \
        vector<void*> datas = {(void*)input, (void*)output};    \
        int64_t decimals = 0;                                   \
        CREATE_NODEDEF(shapes, data_types, datas, decimals);    \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);           \
        base_type expect_out[4] = {(base_type)0};               \
        CalcExpectFunc(*node_def.get(), expect_out);            \
        CompareResult<base_type>(output, expect_out, 4);        \
    }

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

TEST_F(TEST_ROUND_UT, Input_DT_UINT64_Error)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{4}, {4}};
    uint64_t input[4];
    SetRandomValue<uint64_t>(input, 4);
    uint64_t output[4] = {(uint64_t)0};
    vector<void*> datas = {(void*)input, (void*)output};
    int64_t decimals = 0;
    CREATE_NODEDEF(shapes, data_types, datas, decimals);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

template <typename T1, typename T2>
void RunRoundKernel(
    vector<DataType> data_types, vector<vector<int64_t>>& shapes, int64_t decimals, const T1* input1_data,
    const T2* output_exp_data)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];

    for (uint64_t i = 0; i < input1_size; ++i) {
        input1[i] = input1_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 1);
    T2* output = new T2[output_size];
    vector<void*> datas = {(void*)input1, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, decimals);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T2* output_exp = new T2[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_1)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
    const float input1_data[] = {6.2675395f,  -1.4176291f, 3.4629595f, -5.394341f,   -2.8289938f, -5.5214853f,
                                 -7.9450874f, 2.382679f,   6.640656f,  -0.08286998f, 3.7438624f,  3.4867163f,
                                 -0.7270659f, 0.87583184f, 9.734853f,  4.1450114f};
    const float output_exp_data[] = {6.3f, -1.4f, 3.5f, -5.4f, -2.8f, -5.5f, -7.9f, 2.4f,
                                     6.6f, -0.1f, 3.7f, 3.5f,  -0.7f, 0.9f,  9.7f,  4.1f};

    RunRoundKernel<float, float>(data_types, shapes, 1, input1_data, output_exp_data);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_3)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
    const float input1_data[] = {6.2675395f,  -1.4176291f, 3.4629595f, -5.394341f,   -2.8289938f, -5.5214853f,
                                 -7.9450874f, 2.382679f,   6.640656f,  -0.08286998f, 3.7438624f,  3.4867163f,
                                 -0.7270659f, 0.87583184f, 9.734853f,  4.1450114f};
    const float output_exp_data[] = {10.0f, -0.0f, 0.0f, -10.0f, -0.0f, -10.0f, -10.0f, 0.0f,
                                     10.0f, -0.0f, 0.0f, 0.0f,   -0.0f, 0.0f,   10.0f,  0.0f};

    RunRoundKernel<float, float>(data_types, shapes, -1, input1_data, output_exp_data);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT64_WITH_DECIMALS_1)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
    const double input1_data[] = {6.267539549051225,   -1.417629153086562,   3.462959626471891,  -5.394341160937996,
                                  -2.8289938691491923, -5.521485489869223,   -7.945087511045936, 2.382679030305246,
                                  6.64065611902118,    -0.08286997928567885, 3.7438623618393585, 3.4867161584126123,
                                  -0.7270658933236245, 0.8758318294360041,   9.734852974591487,  4.145011300751571};
    const double output_exp_data[] = {6.3, -1.4, 3.5, -5.4, -2.8, -5.5, -7.9, 2.4,
                                      6.6, -0.1, 3.7, 3.5,  -0.7, 0.9,  9.7,  4.1};

    RunRoundKernel<double, double>(data_types, shapes, 1, input1_data, output_exp_data);
}

TEST_F(TEST_ROUND_UT, DATA_IS_POINT_FIVE)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {1}};
    float_t input[1] = {-79.2500};
    float_t output1[1] = {0.0f};
    vector<void*> datas = {(void*)input, (void*)output1};
    CREATE_NODEDEF(shapes, data_types, datas, 1);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}
