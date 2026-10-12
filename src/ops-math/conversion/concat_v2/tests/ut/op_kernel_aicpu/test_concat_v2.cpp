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

#undef private
#undef protected

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_CONCATV2_UT : public testing::Test {};

// ---------- helper: single-input variant that duplicates input0 twice ----------
template <typename T>
void CalcExpectFunc(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    int32_t inputs_size = node_def.InputsSize() - 1;
    int64_t input0_num = input0->NumElements();
    for (int32_t i = 0; i < inputs_size; ++i) {
        for (int64_t j = 0; j < input0_num; ++j) {
            int64_t index = input0_num * i + j;
            expect_out[index] = input0_data[j];
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas, builder)             \
    do {                                                               \
        (builder)                                                      \
            .Input({"x0", data_types[0], shapes[0], datas[0]})         \
            .Input({"x1", data_types[1], shapes[1], datas[1]})         \
            .Input({"concat_dim", data_types[2], shapes[2], datas[2]}) \
            .Output({"y", data_types[3], shapes[3], datas[3]})         \
            .Attr("N", 2);                                             \
    } while (0)

#define ADD_CASE(base_type, aicpu_type)                                                        \
    TEST_F(TEST_CONCATV2_UT, TestConcatV2_##aicpu_type)                                        \
    {                                                                                          \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32, aicpu_type};          \
        vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {}, {4, 11}};                      \
        base_type input[22];                                                                   \
        SetRandomValue<base_type>(input, 22);                                                  \
        base_type output[44] = {(base_type)0};                                                 \
        int32_t concat_dim = 0;                                                                \
        vector<void*> datas = {(void*)input, (void*)input, (void*)&concat_dim, (void*)output}; \
        auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                       \
        NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");                        \
        CREATE_NODEDEF(shapes, data_types, datas, builder);                                    \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                          \
        base_type expect_out[44] = {(base_type)0};                                             \
        CalcExpectFunc(*node_def.get(), expect_out);                                           \
        CompareResult<base_type>(output, expect_out, 44);                                      \
    }

#define ADD_CASE_WITH_SHAPE(case_name, base_type, aicpu_type, shapes, data_num)                \
    TEST_F(TEST_CONCATV2_UT, TestConcatV2_##aicpu_type##_##case_name)                          \
    {                                                                                          \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32, aicpu_type};          \
        base_type input[data_num];                                                             \
        SetRandomValue<base_type>(input, data_num);                                            \
        base_type output[data_num * 2] = {(base_type)0};                                       \
        int32_t concat_dim = 0;                                                                \
        vector<void*> datas = {(void*)input, (void*)input, (void*)&concat_dim, (void*)output}; \
        auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                       \
        NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");                        \
        CREATE_NODEDEF(shapes, data_types, datas, builder);                                    \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                          \
        base_type expect_out[data_num * 2] = {(base_type)0};                                   \
        CalcExpectFunc(*node_def.get(), expect_out);                                           \
        CompareResult<base_type>(output, expect_out, data_num * 2);                            \
    }

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

vector<vector<int64_t>> shapes = {{2, 4}, {2, 4}, {}, {4, 4}};
ADD_CASE_WITH_SHAPE(2_4__2_4__0__4_4, float, DT_FLOAT, shapes, 8)

ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(uint16_t, DT_UINT16)
ADD_CASE(uint32_t, DT_UINT32)
ADD_CASE(uint64_t, DT_UINT64)
ADD_CASE(bool, DT_BOOL)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(std::complex<float>, DT_COMPLEX64)
ADD_CASE(std::complex<double>, DT_COMPLEX128)
ADD_CASE(Eigen::bfloat16, DT_BFLOAT16)

// =========================================================================
//   Extended UT: covers critical paths added by the optimization
//   (axis selection, ByRow/ByInput code paths, >2 inputs, int64 concat_dim,
//    negative axis, row-level correctness, error paths).
// =========================================================================

// Concat along axis=1 (inner) -> exercises RunParallelByRow with row_size>0
TEST_F(TEST_CONCATV2_UT, TestConcatV2_Axis1_Float)
{
    // Shapes: x0[2,3] x1[2,5] -> y[2,8]
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 5}, {}, {2, 8}};
    float x0[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float x1[10] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
    float y[16] = {0.0f};
    int32_t concat_dim = 1;
    vector<void*> datas = {(void*)x0, (void*)x1, (void*)&concat_dim, (void*)y};

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], datas[0]})
           .Input({"x1", DT_FLOAT, shapes[1], datas[1]})
           .Input({"concat_dim", DT_INT32, shapes[2], datas[2]})
           .Output({"y", DT_FLOAT, shapes[3], datas[3]})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expect[16] = {
        1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
        4.0f, 5.0f, 6.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(y[i], expect[i]) << "mismatch at i=" << i;
    }
}

// Concat along axis=0 with flat_dim0=1 after reshape path: shapes [1,N]
// -> exercises RunParallelByInput when flat_dim0==1.
TEST_F(TEST_CONCATV2_UT, TestConcatV2_ByInputPath_Float)
{
    // After axis=0 concat: x0[3] x1[4] -> y[7], inputs_flat_dim0_=1 (axis=0 on 1-D)
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3}, {4}, {}, {7}};
    float x0[3] = {1.0f, 2.0f, 3.0f};
    float x1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float y[7] = {0.0f};
    int32_t concat_dim = 0;
    vector<void*> datas = {(void*)x0, (void*)x1, (void*)&concat_dim, (void*)y};

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], datas[0]})
           .Input({"x1", DT_FLOAT, shapes[1], datas[1]})
           .Input({"concat_dim", DT_INT32, shapes[2], datas[2]})
           .Output({"y", DT_FLOAT, shapes[3], datas[3]})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expect[7] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    for (int i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(y[i], expect[i]) << "mismatch at i=" << i;
    }
}

// Three inputs with different axis=1 sizes (exercises multi-input ByRow hot loop)
TEST_F(TEST_CONCATV2_UT, TestConcatV2_ThreeInputs_Axis1)
{
    vector<vector<int64_t>> shapes = {{2, 1}, {2, 2}, {2, 3}, {}, {2, 6}};
    int32_t x0[2] = {1, 2};
    int32_t x1[4] = {10, 11, 20, 21};
    int32_t x2[6] = {100, 101, 102, 200, 201, 202};
    int32_t y[12] = {0};
    int32_t concat_dim = 1;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_INT32, shapes[0], (void*)x0})
           .Input({"x1", DT_INT32, shapes[1], (void*)x1})
           .Input({"x2", DT_INT32, shapes[2], (void*)x2})
           .Input({"concat_dim", DT_INT32, shapes[3], (void*)&concat_dim})
           .Output({"y", DT_INT32, shapes[4], (void*)y})
           .Attr("N", 3);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect[12] = {1, 10, 11, 100, 101, 102, 2, 20, 21, 200, 201, 202};
    for (int i = 0; i < 12; ++i) {
        EXPECT_EQ(y[i], expect[i]) << "mismatch at i=" << i;
    }
}

// concat_dim supplied as DT_INT64
TEST_F(TEST_CONCATV2_UT, TestConcatV2_ConcatDimInt64)
{
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {1}, {2, 4}};
    float x0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float y[8] = {0.0f};
    int64_t concat_dim = 1;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1})
           .Input({"concat_dim", DT_INT64, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expect[8] = {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(y[i], expect[i]);
    }
}

// negative concat_dim -> should normalize to positive
TEST_F(TEST_CONCATV2_UT, TestConcatV2_NegativeConcatDim)
{
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {}, {2, 4}};
    float x0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float y[8] = {0.0f};
    int32_t concat_dim = -1;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1})
           .Input({"concat_dim", DT_INT32, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expect[8] = {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(y[i], expect[i]);
    }
}

// Large input, multiple rows, axis=1 -> stresses ParallelFor + row loop
TEST_F(TEST_CONCATV2_UT, TestConcatV2_Large_Axis1)
{
    constexpr int64_t kRows = 64;
    constexpr int64_t kCols0 = 31;
    constexpr int64_t kCols1 = 17;
    vector<vector<int64_t>> shapes = {{kRows, kCols0}, {kRows, kCols1}, {}, {kRows, kCols0 + kCols1}};
    vector<float> x0(kRows * kCols0);
    vector<float> x1(kRows * kCols1);
    for (int i = 0; i < kRows * kCols0; ++i) x0[i] = static_cast<float>(i);
    for (int i = 0; i < kRows * kCols1; ++i) x1[i] = static_cast<float>(i + 100000);
    vector<float> y(kRows * (kCols0 + kCols1), 0.0f);
    int32_t concat_dim = 1;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0.data()})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1.data()})
           .Input({"concat_dim", DT_INT32, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y.data()})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    for (int64_t r = 0; r < kRows; ++r) {
        for (int64_t c = 0; c < kCols0; ++c) {
            EXPECT_FLOAT_EQ(y[r * (kCols0 + kCols1) + c], x0[r * kCols0 + c]);
        }
        for (int64_t c = 0; c < kCols1; ++c) {
            EXPECT_FLOAT_EQ(y[r * (kCols0 + kCols1) + kCols0 + c], x1[r * kCols1 + c]);
        }
    }
}

// Error path: bad concat_dim out of range -> returns KERNEL_STATUS_PARAM_INVALID
TEST_F(TEST_CONCATV2_UT, TestConcatV2_BadConcatDim)
{
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {}, {2, 4}};
    float x0[4] = {0.0f};
    float x1[4] = {0.0f};
    float y[8] = {0.0f};
    int32_t concat_dim = 5; // out of range for rank-2 tensor

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1})
           .Input({"concat_dim", DT_INT32, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Error path: shape rank mismatch between inputs
TEST_F(TEST_CONCATV2_UT, TestConcatV2_ShapeRankMismatch)
{
    vector<vector<int64_t>> shapes = {{2, 2}, {4}, {}, {4, 2}};
    float x0[4] = {0.0f};
    float x1[4] = {0.0f};
    float y[8] = {0.0f};
    int32_t concat_dim = 0;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1})
           .Input({"concat_dim", DT_INT32, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Empty input tensor (NumElements == 0 in one input) -> should be skipped
TEST_F(TEST_CONCATV2_UT, TestConcatV2_EmptyInputSkipped)
{
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 0}, {}, {2, 3}};
    float x0[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float x1[1] = {0.0f}; // unused
    float y[6] = {0.0f};
    int32_t concat_dim = 1;

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");
    builder.Input({"x0", DT_FLOAT, shapes[0], (void*)x0})
           .Input({"x1", DT_FLOAT, shapes[1], (void*)x1})
           .Input({"concat_dim", DT_INT32, shapes[2], (void*)&concat_dim})
           .Output({"y", DT_FLOAT, shapes[3], (void*)y})
           .Attr("N", 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(y[i], x0[i]);
    }
}
