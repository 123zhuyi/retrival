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
#include <cmath>

using namespace std;
using namespace aicpu;

class TEST_ABS_GRAD_UT : public testing::Test {};

#define CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas)        \
    auto node_def = CpuKernelUtils::CreateNodeDef();              \
    NodeDefBuilder builder(node_def.get(), "AbsGrad", "AbsGrad"); \
    builder.Input({"y", data_types[0], shapes[0], datas[0]});     \
    builder.Input({"dy", data_types[1], shapes[1], datas[1]});    \
    builder.Output({"z", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_POSITIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // y > 0
    float dy[4] = {0.5f, 1.0f, 1.5f, 2.0f}; // 梯度
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[4] = {0.5f, 1.0f, 1.5f, 2.0f}; // z = dy * 1
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_NEGATIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[4] = {-1.0f, -2.0f, -3.0f, -4.0f}; // y < 0
    float dy[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[4] = {-0.5f, -1.0f, -1.5f, -2.0f}; // z = dy * -1
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_ZERO_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // y = 0
    float dy[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // z = dy * 0
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_MIXED_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{6}, {6}, {6}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[6] = {1.0f, -2.0f, 0.0f, 3.0f, -4.0f, 0.0f}; // 正、负、零混合
    float dy[6] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    float z[6] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[6] = {0.5f, -1.0f, 0.0f, 2.0f, -2.5f, 0.0f};
    bool compare = CompareResult(z, z_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_NAN_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float nan_val = std::nanf("");
    float y[4] = {nan_val, nan_val, nan_val, nan_val}; // y = NaN
    float dy[4] = {0.5f, 1.0f, 1.5f, 2.0f};            // dy 正常值
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // z = 0 * dy = 0
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_NAN_GRADIENT_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float nan_val = std::nanf("");
    float y[4] = {1.0f, -2.0f, 0.0f, nan_val};
    float dy[4] = {nan_val, nan_val, nan_val, nan_val}; // dy = NaN
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[4] = {nan_val, nan_val, nan_val, nan_val}; // NaN * any = NaN
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT16_POSITIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};

    Eigen::half y[4];
    Eigen::half dy[4];
    Eigen::half z[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = (Eigen::half)(i + 1); // y > 0
        dy[i] = (Eigen::half)(0.5f * (i + 1));
    }
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half z_exp[4];
    for (int i = 0; i < 4; ++i) {
        z_exp[i] = (Eigen::half)(0.5f * (i + 1)); // z = dy * 1
    }
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT16_NEGATIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};

    Eigen::half y[4];
    Eigen::half dy[4];
    Eigen::half z[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = (Eigen::half)(-(i + 1)); // y < 0
        dy[i] = (Eigen::half)(0.5f * (i + 1));
    }
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half z_exp[4];
    for (int i = 0; i < 4; ++i) {
        z_exp[i] = (Eigen::half)(-0.5f * (i + 1)); // z = dy * -1
    }
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT16_MIXED_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{6}, {6}, {6}};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};

    Eigen::half y[6] = {(Eigen::half)1.0f, (Eigen::half)-2.0f, (Eigen::half)0.0f,
                        (Eigen::half)3.0f, (Eigen::half)-4.0f, (Eigen::half)0.0f};
    Eigen::half dy[6] = {(Eigen::half)0.5f, (Eigen::half)1.0f, (Eigen::half)1.5f,
                         (Eigen::half)2.0f, (Eigen::half)2.5f, (Eigen::half)3.0f};
    Eigen::half z[6];
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half z_exp[6] = {(Eigen::half)0.5f, (Eigen::half)-1.0f, (Eigen::half)0.0f,
                            (Eigen::half)2.0f, (Eigen::half)-2.5f, (Eigen::half)0.0f};
    bool compare = CompareResult(z, z_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, BF16_POSITIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_BF16, DT_BF16, DT_BF16};

    Eigen::bfloat16 y[4];
    Eigen::bfloat16 dy[4];
    Eigen::bfloat16 z[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = (Eigen::bfloat16)(i + 1); // y > 0
        dy[i] = (Eigen::bfloat16)(0.5f * (i + 1));
    }
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::bfloat16 z_exp[4];
    for (int i = 0; i < 4; ++i) {
        z_exp[i] = (Eigen::bfloat16)(0.5f * (i + 1)); // z = dy * 1
    }
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, BF16_NEGATIVE_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_BF16, DT_BF16, DT_BF16};

    Eigen::bfloat16 y[4];
    Eigen::bfloat16 dy[4];
    Eigen::bfloat16 z[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = (Eigen::bfloat16)(-(i + 1)); // y < 0
        dy[i] = (Eigen::bfloat16)(0.5f * (i + 1));
    }
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::bfloat16 z_exp[4];
    for (int i = 0; i < 4; ++i) {
        z_exp[i] = (Eigen::bfloat16)(-0.5f * (i + 1)); // z = dy * -1
    }
    bool compare = CompareResult(z, z_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_2D_SHAPE_SUCC)
{
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[6] = {1.0f, -2.0f, 0.0f, 3.0f, -4.0f, 5.0f};
    float dy[6] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    float z[6] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[6] = {0.5f, -1.0f, 0.0f, 2.0f, -2.5f, 3.0f};
    bool compare = CompareResult(z, z_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, FLOAT32_3D_SHAPE_SUCC)
{
    vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[8] = {1.0f, -1.0f, 2.0f, -2.0f, 0.0f, 0.0f, 3.0f, -3.0f};
    float dy[8] = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f};
    float z[8] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float z_exp[8] = {1.0f, -1.0f, 2.0f, -2.0f, 0.0f, 0.0f, 4.0f, -4.0f};
    bool compare = CompareResult(z, z_exp, 8);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ABS_GRAD_UT, INPUT_DTYPE_DISMATCH_Y_DY)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT}; // y 和 dy 类型不匹配

    float y[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Eigen::half dy[4];
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ABS_GRAD_UT, INPUT_OUTPUT_DTYPE_DISMATCH)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT16}; // 输入输出类型不匹配

    float y[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dy[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    Eigen::half z[4];
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ABS_GRAD_UT, INPUT_SHAPE_DISMATCH)
{
    vector<vector<int64_t>> shapes = {{4}, {5}, {4}}; // y 和 dy shape 不匹配
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    float y[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dy[5] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    float z[4] = {0.0f};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ABS_GRAD_UT, UNSUPPORTED_DTYPE_INT32)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32}; // INT32 不支持

    int32_t y[4] = {1, 2, 3, 4};
    int32_t dy[4] = {5, 6, 7, 8};
    int32_t z[4] = {0};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ABS_GRAD_UT, UNSUPPORTED_DTYPE_BOOL)
{
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL}; // BOOL 不支持

    bool y[4] = {true, false, true, false};
    bool dy[4] = {true, true, false, false};
    bool z[4] = {false};
    vector<void*> datas = {(void*)y, (void*)dy, (void*)z};

    CREATE_NODEDEF_ABS_GRAD(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
