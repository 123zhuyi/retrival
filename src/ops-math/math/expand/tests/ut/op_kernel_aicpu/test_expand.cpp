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

class TEST_EXPAND_UT : public testing::Test {};

#define CREATE_EXPAND_NODEDEF(shapes, data_types, datas)      \
    auto node_def = CpuKernelUtils::CreateNodeDef();          \
    NodeDefBuilder(node_def.get(), "Expand", "Expand")        \
        .Input({"x", data_types[0], shapes[0], datas[0]})     \
        .Input({"shape", data_types[1], shapes[1], datas[1]}) \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_EXPAND_UT, TestExpandInt32ShapeInt32)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 1, 6}, {3}, {2, 3, 6}};
    int32_t input[12] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
    int32_t shape[3] = {2, 3, 6};
    int32_t output[36] = {0};
    int32_t expected[36] = {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3,
                            4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<int32_t>(output, expected, 36));
}

TEST_F(TEST_EXPAND_UT, TestExpandInt64ShapeInt64)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 4}, {3}, {2, 2, 4}};
    int64_t input[8] = {1, 1, 2, 2, 3, 3, 4, 4};
    int64_t shape[3] = {2, 2, 4};
    int64_t output[16] = {0};
    int64_t expected[16] = {1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<int64_t>(output, expected, 16));
}

TEST_F(TEST_EXPAND_UT, TestExpandBool)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1, 2}, {2}, {3, 2}};
    bool input[2] = {true, false};
    int32_t shape[2] = {3, 2};
    bool output[6] = {false};
    bool expected[6] = {true, false, true, false, true, false};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<bool>(output, expected, 6));
}

TEST_F(TEST_EXPAND_UT, TestExpandBFloat16)
{
    vector<DataType> data_types = {DT_BFLOAT16, DT_INT64, DT_BFLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2}, {2}, {2, 2}};
    Eigen::bfloat16 input[2] = {static_cast<Eigen::bfloat16>(1.5F), static_cast<Eigen::bfloat16>(2.5F)};
    int64_t shape[2] = {2, 2};
    Eigen::bfloat16 output[4] = {static_cast<Eigen::bfloat16>(0.0F)};
    Eigen::bfloat16 expected[4] = {
        static_cast<Eigen::bfloat16>(1.5F), static_cast<Eigen::bfloat16>(2.5F), static_cast<Eigen::bfloat16>(1.5F),
        static_cast<Eigen::bfloat16>(2.5F)};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<Eigen::bfloat16>(output, expected, 4));
}

TEST_F(TEST_EXPAND_UT, TestExpandEmptyTensorBool)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{}, {0}, {}};
    bool input[1] = {true};
    int32_t shape[1] = {0};
    bool output[1] = {false};
    bool expected[1] = {true};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<bool>(output, expected, 1));
}

TEST_F(TEST_EXPAND_UT, TestExpandInvalidBroadcast)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2, 3}};
    int32_t input[4] = {1, 2, 3, 4};
    int32_t shape[2] = {2, 3};
    int32_t output[6] = {0};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_EXPAND_UT, TestExpandInvalidShortShapeRank)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 1, 3}, {2}, {2, 3}};
    int32_t input[6] = {1, 2, 3, 4, 5, 6};
    int32_t shape[2] = {2, 3};
    int32_t output[6] = {0};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_EXPAND_UT, TestExpandNegativeOneUsesInputDim)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 1, 3}, {3}, {2, 4, 3}};
    int32_t input[6] = {1, 2, 3, 4, 5, 6};
    int32_t shape[3] = {-1, 4, 3};
    int32_t output[24] = {0};
    int32_t expected[24] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
    vector<void*> datas = {input, shape, output};

    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult<int32_t>(output, expected, 24));
}