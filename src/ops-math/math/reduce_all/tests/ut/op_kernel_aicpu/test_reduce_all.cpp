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

class TEST_REDUCE_ALL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, keep_dims) \
    auto node_def = CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "ReduceAll", "ReduceAll") \
        .Input({"x", data_types[0], shapes[0], datas[0]}) \
        .Input({"axes", data_types[1], shapes[1], datas[1]}) \
        .Output({"y", data_types[2], shapes[2], datas[2]}) \
        .Attr("keep_dims", keep_dims)

TEST_F(TEST_REDUCE_ALL_UT, ReduceSingleAxisInt32)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {1}, {2}};
    bool input[6] = {true, true, false, true, true, true};
    int32_t axes[1] = {1};
    bool output[2] = {false, false};
    vector<void *> datas = {(void *)input, (void *)axes, (void *)output};

    CREATE_NODEDEF(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[2] = {false, true};
    EXPECT_EQ(CompareResult<bool>(output, expected, 2), true);
}

TEST_F(TEST_REDUCE_ALL_UT, ReduceMultiAxisInt64KeepDims)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT64, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {1, 2, 1}};
    bool input[8] = {true, true, true, false, true, true, true, true};
    int64_t axes[2] = {0, -1};
    bool output[2] = {false, false};
    vector<void *> datas = {(void *)input, (void *)axes, (void *)output};

    CREATE_NODEDEF(shapes, data_types, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[2] = {true, false};
    EXPECT_EQ(CompareResult<bool>(output, expected, 2), true);
}

TEST_F(TEST_REDUCE_ALL_UT, EmptyAxesReduceAll)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2}, {}, {}};
    bool input[4] = {true, true, true, false};
    bool output[1] = {true};
    vector<void *> datas = {(void *)input, nullptr, (void *)output};

    CREATE_NODEDEF(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[1] = {false};
    EXPECT_EQ(CompareResult<bool>(output, expected, 1), true);
}

TEST_F(TEST_REDUCE_ALL_UT, EmptyInputKeepDims)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 0, 3}, {1}, {2, 1, 3}};
    bool input[1] = {false};
    int32_t axes[1] = {1};
    bool output[6] = {false, false, false, false, false, false};
    vector<void *> datas = {(void *)input, (void *)axes, (void *)output};

    CREATE_NODEDEF(shapes, data_types, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool expected[6] = {true, true, true, true, true, true};
    EXPECT_EQ(CompareResult<bool>(output, expected, 6), true);
}

TEST_F(TEST_REDUCE_ALL_UT, AxisOutOfRange)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2}, {1}, {2}};
    bool input[4] = {true, true, false, true};
    int32_t axes[1] = {2};
    bool output[2] = {false, false};
    vector<void *> datas = {(void *)input, (void *)axes, (void *)output};

    CREATE_NODEDEF(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}