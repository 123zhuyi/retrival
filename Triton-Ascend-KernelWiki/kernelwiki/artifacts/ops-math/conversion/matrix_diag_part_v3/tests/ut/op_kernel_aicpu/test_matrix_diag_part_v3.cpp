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
#include <string>
#include <vector>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_MATRIX_DIAG_PART_V3_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, align)                     \
    do {                                                                     \
        NodeDefBuilder(node_def.get(), "MatrixDiagPartV3", "MatrixDiagPartV3") \
            .Input({"input", data_types[0], shapes[0], datas[0], FORMAT_ND})   \
            .Input({"k", data_types[1], shapes[1], datas[1], FORMAT_ND})       \
            .Input({"padding_value", data_types[2], shapes[2], datas[2], FORMAT_ND}) \
            .Output({"diagonal", data_types[3], shapes[3], datas[3], FORMAT_ND}) \
            .Attr("align", align);                                            \
    } while (0)

#define MATRIX_DIAG_PART_V3_BASIC_CASE(base_type, aicpu_type)                    \
    TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, aicpu_type##_SUCCESS) {                  \
        string align = "RIGHT_LEFT";                                               \
        vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type,          \
                                       aicpu_type};                                \
        vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {3}};                  \
        base_type x[9];                                                             \
        x[0] = (base_type)1;                                                        \
        x[1] = (base_type)2;                                                        \
        x[2] = (base_type)3;                                                        \
        x[3] = (base_type)4;                                                        \
        x[4] = (base_type)5;                                                        \
        x[5] = (base_type)6;                                                        \
        x[6] = (base_type)7;                                                        \
        x[7] = (base_type)8;                                                        \
        x[8] = (base_type)9;                                                        \
        int32_t k[1] = {0};                                                         \
        base_type padding_value[1];                                                 \
        padding_value[0] = (base_type)0;                                            \
        base_type output[3];                                                        \
        for (int i = 0; i < 3; ++i) {                                               \
            output[i] = (base_type)0;                                               \
        }                                                                           \
        base_type expected[3];                                                      \
        expected[0] = (base_type)1;                                                 \
        expected[1] = (base_type)5;                                                 \
        expected[2] = (base_type)9;                                                 \
        vector<void *> datas = {x, k, padding_value, output};                      \
        auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();          \
        CREATE_NODEDEF(shapes, data_types, datas, align);                         \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
        EXPECT_TRUE(CompareResult(output, expected, 3));                          \
    }

MATRIX_DIAG_PART_V3_BASIC_CASE(int32_t, DT_INT32)
MATRIX_DIAG_PART_V3_BASIC_CASE(int64_t, DT_INT64)
MATRIX_DIAG_PART_V3_BASIC_CASE(float, DT_FLOAT)
MATRIX_DIAG_PART_V3_BASIC_CASE(double, DT_DOUBLE)
MATRIX_DIAG_PART_V3_BASIC_CASE(Eigen::half, DT_FLOAT16)
MATRIX_DIAG_PART_V3_BASIC_CASE(int8_t, DT_INT8)
MATRIX_DIAG_PART_V3_BASIC_CASE(int16_t, DT_INT16)
MATRIX_DIAG_PART_V3_BASIC_CASE(uint8_t, DT_UINT8)
MATRIX_DIAG_PART_V3_BASIC_CASE(uint16_t, DT_UINT16)
MATRIX_DIAG_PART_V3_BASIC_CASE(uint32_t, DT_UINT32)
MATRIX_DIAG_PART_V3_BASIC_CASE(uint64_t, DT_UINT64)

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, COMPLEX64_SUCCESS)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {3}};
    std::complex<float> x[9] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f},
                                {4.0f, 1.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                {7.0f, 0.0f}, {8.0f, -1.0f}, {9.0f, 0.0f}};
    int32_t k[1] = {0};
    std::complex<float> padding_value[1] = {{0.0f, 0.0f}};
    std::complex<float> output[3] = {};
    std::complex<float> expected[3] = {{1.0f, 0.0f}, {5.0f, 0.0f}, {9.0f, 0.0f}};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expected, 3));
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, COMPLEX128_SUCCESS)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {3}};
    std::complex<double> x[9] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
                                 {4.0, 1.0}, {5.0, 0.0}, {6.0, 0.0},
                                 {7.0, 0.0}, {8.0, -1.0}, {9.0, 0.0}};
    int32_t k[1] = {0};
    std::complex<double> padding_value[1] = {{0.0, 0.0}};
    std::complex<double> output[3] = {};
    std::complex<double> expected[3] = {{1.0, 0.0}, {5.0, 0.0}, {9.0, 0.0}};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expected, 3));
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, COMPLEX128_BAND_SUCCESS)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{3, 3}, {2}, {1}, {2, 3}};
    std::complex<double> x[9] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
                                 {4.0, 1.0}, {5.0, 0.0}, {6.0, 0.0},
                                 {7.0, 0.0}, {8.0, -1.0}, {9.0, 0.0}};
    int32_t k[2] = {-1, 0};
    std::complex<double> padding_value[1] = {{-1.0, -1.0}};
    std::complex<double> output[6] = {};
    std::complex<double> expected[6] = {{1.0, 0.0}, {5.0, 0.0}, {9.0, 0.0},
                                        {4.0, 1.0}, {8.0, -1.0}, {-1.0, -1.0}};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expected, 6));
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, BATCH_SUCCESS)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3, 3}, {1}, {1}, {2, 3}};
    int32_t x[18] = {1, 0, 0, 0, 2, 0, 0, 0, 3,
                     4, 0, 0, 0, 5, 0, 0, 0, 6};
    int32_t k[1] = {0};
    int32_t padding_value[1] = {0};
    int32_t output[6] = {0};
    int32_t expected[6] = {1, 2, 3, 4, 5, 6};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expected, 6));
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, ALIGN_INVALID)
{
    string align = "INVALID_ALIGN";
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {3}};
    double x[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    int32_t k[1] = {0};
    double padding_value[1] = {0.0};
    double output[3] = {0.0};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, K_RANGE_INVALID)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 3}, {2}, {1}, {3}};
    int32_t x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t k[2] = {1, 0};
    int32_t padding_value[1] = {0};
    int32_t output[3] = {0};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, BOOL_INPUT_UNSUPPORT)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{3, 3}, {1}, {1}, {3}};
    bool x[9] = {true, false, true, false, true, false, true, false, true};
    int32_t k[1] = {0};
    bool padding_value[1] = {false};
    bool output[3] = {false};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_PART_V3_UT, INPUT_NOT_MATRIX)
{
    string align = "RIGHT_LEFT";
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {1}, {1}, {3}};
    int32_t x[3] = {1, 2, 3};
    int32_t k[1] = {0};
    int32_t padding_value[1] = {0};
    int32_t output[3] = {0};
    vector<void *> datas = {x, k, padding_value, output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
