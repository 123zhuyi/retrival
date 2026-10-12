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

class TEST_COMPLEX_ABS_UT : public testing::Test {};

#define CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas)           \
    auto node_def = CpuKernelUtils::CreateNodeDef();                    \
    NodeDefBuilder builder(node_def.get(), "ComplexAbs", "ComplexAbs"); \
    builder.Input({"x", data_types[0], shapes[0], datas[0]});           \
    builder.Output({"y", data_types[1], shapes[1], datas[1]});

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX32_BASIC_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX32, DT_FLOAT16};

    std::complex<Eigen::half> x[4] = {{Eigen::half(1.0f), Eigen::half(0.0f)},
                                      {Eigen::half(0.0f), Eigen::half(1.0f)},
                                      {Eigen::half(3.0f), Eigen::half(4.0f)},
                                      {Eigen::half(-3.0f), Eigen::half(-4.0f)}};
    Eigen::half y[4] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half y_exp[4] = {
        Eigen::half(1.0f), // sqrt(1^2 + 0^2) = 1
        Eigen::half(1.0f), // sqrt(0^2 + 1^2) = 1
        Eigen::half(5.0f), // sqrt(3^2 + 4^2) = 5
        Eigen::half(5.0f)  // sqrt((-3)^2 + (-4)^2) = 5
    };
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_BASIC_SUCC)
{
    vector<vector<int64_t>> shapes = {{6}, {6}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[6] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {3.0f, 4.0f}, {-3.0f, -4.0f}, {5.0f, 12.0f}};
    float y[6] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[6] = {
        1.0f,      // sqrt(1^2 + 0^2) = 1
        1.0f,      // sqrt(0^2 + 1^2) = 1
        1.414213f, // sqrt(1^2 + 1^2) = sqrt(2)
        5.0f,      // sqrt(3^2 + 4^2) = 5
        5.0f,      // sqrt((-3)^2 + (-4)^2) = 5
        13.0f      // sqrt(5^2 + 12^2) = 13
    };
    bool compare = CompareResult(y, y_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_2D_SHAPE_SUCC)
{
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[6] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}, {5.0f, 5.0f}, {6.0f, 6.0f}};
    float y[6] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[6] = {1.414213f, 2.828427f, 4.242640f, 5.656854f, 7.071067f, 8.485281f};
    bool compare = CompareResult(y, y_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_3D_SHAPE_SUCC)
{
    vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 2, 2}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[8] = {{1.0f, 0.0f}, {0.0f, 1.0f},  {1.0f, 1.0f},  {2.0f, 2.0f},
                                {3.0f, 4.0f}, {-3.0f, 4.0f}, {5.0f, 12.0f}, {0.0f, 0.0f}};
    float y[8] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[8] = {1.0f, 1.0f, 1.414213f, 2.828427f, 5.0f, 5.0f, 13.0f, 0.0f};
    bool compare = CompareResult(y, y_exp, 8);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX32_ZERO_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX32, DT_FLOAT16};

    std::complex<Eigen::half> x[4] = {{Eigen::half(0.0f), Eigen::half(0.0f)},
                                      {Eigen::half(0.0f), Eigen::half(0.0f)},
                                      {Eigen::half(0.0f), Eigen::half(0.0f)},
                                      {Eigen::half(0.0f), Eigen::half(0.0f)}};
    Eigen::half y[4] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    Eigen::half y_exp[4] = {Eigen::half(0.0f), Eigen::half(0.0f), Eigen::half(0.0f), Eigen::half(0.0f)};
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_ZERO_VALUE_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_NEGATIVE_VALUES_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[4] = {{-1.0f, -1.0f}, {-3.0f, -4.0f}, {-5.0f, -12.0f}, {-7.0f, -24.0f}};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[4] = {
        1.414213f, // sqrt((-1)^2 + (-1)^2) = sqrt(2)
        5.0f,      // sqrt((-3)^2 + (-4)^2) = 5
        13.0f,     // sqrt((-5)^2 + (-12)^2) = 13
        25.0f      // sqrt((-7)^2 + (-24)^2) = 25
    };
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_PURE_REAL_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[4] = {{1.0f, 0.0f}, {-2.0f, 0.0f}, {3.0f, 0.0f}, {-4.0f, 0.0f}};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX64_PURE_IMAGINARY_SUCC)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT};

    std::complex<float> x[4] = {{0.0f, 1.0f}, {0.0f, -2.0f}, {0.0f, 3.0f}, {0.0f, -4.0f}};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float y_exp[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bool compare = CompareResult(y, y_exp, 4);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPLEX_ABS_UT, COMPLEX128_UNSUPPORTED)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX128, DT_DOUBLE};

    std::complex<double> x[4] = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
    double y[4] = {0.0};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPLEX_ABS_UT, INPUT_FLOAT_UNSUPPORTED)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPLEX_ABS_UT, DTYPE_MISMATCH_COMPLEX32_FLOAT32)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX32, DT_FLOAT}; // COMPLEX32 应输出 FLOAT16

    std::complex<Eigen::half> x[4] = {{Eigen::half(1.0f), Eigen::half(1.0f)},
                                      {Eigen::half(2.0f), Eigen::half(2.0f)},
                                      {Eigen::half(3.0f), Eigen::half(3.0f)},
                                      {Eigen::half(4.0f), Eigen::half(4.0f)}};
    float y[4] = {0.0f};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPLEX_ABS_UT, DTYPE_MISMATCH_COMPLEX64_FLOAT16)
{
    vector<vector<int64_t>> shapes = {{4}, {4}};
    vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT16}; // COMPLEX64 应输出 FLOAT32

    std::complex<float> x[4] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}};
    Eigen::half y[4] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)x, (void*)y};

    CREATE_NODEDEF_COMPLEX_ABS(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
