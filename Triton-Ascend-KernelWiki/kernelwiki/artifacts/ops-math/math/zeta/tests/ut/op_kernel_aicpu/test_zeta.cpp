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

#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>

using namespace std;
using namespace aicpu;

class TEST_ZETA_UT : public testing::Test {};

#define CREATE_NODEDEF_ZETA(shapes, data_types, datas)    \
    auto node_def = CpuKernelUtils::CreateNodeDef();      \
    NodeDefBuilder node(node_def.get(), "Zeta", "Zeta");  \
    node.Input({"x", data_types[0], shapes[0], datas[0]}) \
        .Input({"q", data_types[1], shapes[1], datas[1]}) \
        .Output({"z", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ZETA_UT, TestZeta_DT_FLOAT)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    float input_x[6] = {2.0f, 3.0f, 4.0f, 2.0f, 3.0f, 4.0f};
    float input_q[6] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    float output_z[6] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    // zeta(2,1) = pi^2/6 ≈ 1.6449, zeta(3,1) ≈ 1.2021, zeta(4,1) ≈ 1.0823
    // zeta(2,2) = pi^2/6 - 1 ≈ 0.6449, zeta(3,2) ≈ 0.2021, zeta(4,2) ≈ 0.0823
    EXPECT_NEAR(output_z[0], Eigen::numext::zeta(2.0f, 1.0f), 1e-4f);
    EXPECT_NEAR(output_z[1], Eigen::numext::zeta(3.0f, 1.0f), 1e-4f);
    EXPECT_NEAR(output_z[2], Eigen::numext::zeta(4.0f, 1.0f), 1e-4f);
    EXPECT_NEAR(output_z[3], Eigen::numext::zeta(2.0f, 2.0f), 1e-4f);
    EXPECT_NEAR(output_z[4], Eigen::numext::zeta(3.0f, 2.0f), 1e-4f);
    EXPECT_NEAR(output_z[5], Eigen::numext::zeta(4.0f, 2.0f), 1e-4f);
}

TEST_F(TEST_ZETA_UT, TestZeta_DT_DOUBLE)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    double input_x[6] = {2.0, 3.0, 4.0, 2.0, 3.0, 4.0};
    double input_q[6] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    double output_z[6] = {0.0};
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_NEAR(output_z[0], Eigen::numext::zeta(2.0, 1.0), 1e-10);
    EXPECT_NEAR(output_z[1], Eigen::numext::zeta(3.0, 1.0), 1e-10);
    EXPECT_NEAR(output_z[2], Eigen::numext::zeta(4.0, 1.0), 1e-10);
    EXPECT_NEAR(output_z[3], Eigen::numext::zeta(2.0, 2.0), 1e-10);
    EXPECT_NEAR(output_z[4], Eigen::numext::zeta(3.0, 2.0), 1e-10);
    EXPECT_NEAR(output_z[5], Eigen::numext::zeta(4.0, 2.0), 1e-10);
}

TEST_F(TEST_ZETA_UT, TestZeta_INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    float* null_ptr = nullptr;
    float input_q[6] = {1.0f};
    float output_z[6] = {0.0f};
    vector<void*> datas = {(void*)null_ptr, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZETA_UT, TestZeta_OUTPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
    float input_x[1] = {2.0f};
    float input_q[1] = {1.0f};
    float* null_ptr = nullptr;
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)null_ptr};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZETA_UT, TestZeta_SHAPE_MISMATCH_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 6}, {2, 8}, {2, 6}};
    float input_x[12] = {2.0f};
    float input_q[16] = {1.0f};
    float output_z[12] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZETA_UT, TestZeta_DTYPE_MISMATCH_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    float input_x[6] = {2.0f};
    double input_q[6] = {1.0};
    float output_z[6] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZETA_UT, TestZeta_UNSUPPORTED_TYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    int32_t input_x[6] = {2};
    int32_t input_q[6] = {1};
    int32_t output_z[6] = {0};
    vector<void*> datas = {(void*)input_x, (void*)input_q, (void*)output_z};
    CREATE_NODEDEF_ZETA(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZETA_UT, TestZeta_NO_OUTPUT_EXCEPTION)
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    float input_x[6] = {2.0f};
    NodeDefBuilder node(node_def.get(), "Zeta", "Zeta");
    node.Input({"x", DT_FLOAT, {2, 3}, input_x});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
