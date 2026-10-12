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

using namespace std;
using namespace aicpu;

class TEST_CONJUGATETRANSPOSE_UT : public testing::Test {};

#define CREATE_NODEDEF(node_def, shapes, data_types, datas)                    \
    NodeDefBuilder(node_def.get(), "ConjugateTranspose", "ConjugateTranspose") \
        .Input({"x", data_types[0], shapes[0], datas[0]})                      \
        .Input({"perm", data_types[1], shapes[1], datas[1]})                   \
        .Output({"y", data_types[2], shapes[2], datas[2]})

// x {2,3} complex128, perm {1,0} -> y {3,2}: transpose then conjugate.
TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2, 3}, {2}, {3, 2}};
    complex<double> input0[6] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};
    int32_t input1[2] = {1, 0};
    complex<double> output_exp[6] = {{1, -1}, {4, -4}, {2, -2}, {5, -5}, {3, -3}, {6, -6}};
    complex<double> output[6];
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

// x {2,3} complex64, perm {1,0} -> y {3,2}.
TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2}, {3, 2}};
    complex<float> input0[6] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};
    int64_t input1[2] = {1, 0};
    complex<float> output_exp[6] = {{1, -1}, {4, -4}, {2, -2}, {5, -5}, {3, -3}, {6, -6}};
    complex<float> output[6];
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

// x {2,3} bool, perm {1,0} -> y {3,2}: conjugate is identity for real types.
TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_BOOL_SUCC)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {2}, {3, 2}};
    bool input0[6] = {false, false, false, true, true, true};
    int32_t input1[2] = {1, 0};
    bool output_exp[6] = {false, true, false, true, false, true};
    bool output[6];
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

// x {2,2,2} int32, perm {0,2,1} -> swap last two axes.
TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT32_3D_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 2, 2}, {3}, {2, 2, 2}};
    int32_t input0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t input1[3] = {0, 2, 1};
    int32_t output_exp[8] = {0, 2, 1, 3, 4, 6, 5, 7};
    int32_t output[8];
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, 8);
    EXPECT_EQ(compare, true);
}

// perm is not 1-D -> param invalid.
TEST_F(TEST_CONJUGATETRANSPOSE_UT, PERM_NOT_1D_EXCEPTION)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 1}, {3, 2}};
    complex<double> input0[6] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};
    int32_t input1[2] = {1, 0};
    complex<double> output[6];
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
