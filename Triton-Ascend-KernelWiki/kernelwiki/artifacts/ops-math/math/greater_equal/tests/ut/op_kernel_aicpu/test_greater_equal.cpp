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

using namespace std;
using namespace aicpu;

class TEST_GREATER_EQUAL_UT : public testing::Test {};

#define CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas)    \
    auto node_def = CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(node_def.get(), "GreaterEqual", "GreaterEqual") \
        .Input({"x1", data_types[0], shapes[0], datas[0]})         \
        .Input({"x2", data_types[1], shapes[1], datas[1]})         \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_GREATER_EQUAL_UT, SameShapeFloatSuccess)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    float input0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float input1[4] = {1.0f, 1.5f, 5.0f, 4.0f};
    bool output[4] = {false};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

    CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[4] = {true, true, false, true};
    EXPECT_EQ(CompareResult<bool>(output, expected, 4), true);
}

TEST_F(TEST_GREATER_EQUAL_UT, BroadcastVectorSuccess)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 4}, {4}, {2, 4}};
    int32_t input0[8] = {5, 4, 6, 7, 5, 4, 6, 7};
    int32_t input1[4] = {5, 2, 5, 10};
    bool output[8] = {false};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

    CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[8] = {true, true, true, false, true, true, true, false};
    EXPECT_EQ(CompareResult<bool>(output, expected, 8), true);
}

TEST_F(TEST_GREATER_EQUAL_UT, ScalarBroadcastSuccess)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_BOOL};
    vector<vector<int64_t>> shapes = {{4}, {1}, {4}};
    uint16_t input0[4] = {1, 2, 3, 4};
    uint16_t input1[1] = {3};
    bool output[4] = {false};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

    CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[4] = {false, false, true, true};
    EXPECT_EQ(CompareResult<bool>(output, expected, 4), true);
}

TEST_F(TEST_GREATER_EQUAL_UT, InvalidBroadcastShape)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    int32_t input0[16] = {0};
    int32_t input1[16] = {0};
    bool output[16] = {false};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

    CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_EQUAL_UT, DifferentInputTypeReturnsError)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    int32_t input0[2] = {1, 2};
    int64_t input1[2] = {1, 1};
    bool output[2] = {false};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

    CREATE_GREATER_EQUAL_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}