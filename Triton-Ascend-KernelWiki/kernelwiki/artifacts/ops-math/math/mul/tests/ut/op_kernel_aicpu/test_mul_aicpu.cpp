/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <memory>
#include <numeric>
#include <vector>

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

class TEST_MUL_AICPU_UT : public testing::Test {};

auto CreateMulNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                      const vector<void*>& datas) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "Mul", "Mul")
        .Input({"x1", data_types[0], shapes[0], datas[0]})
        .Input({"x2", data_types[1], shapes[1], datas[1]})
        .Output({"y", data_types[2], shapes[2], datas[2]});
    return node_def;
}

template <typename T>
void RunMulKernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types, const vector<T>& input1,
                  const vector<T>& input2, const vector<T>& expect_output, uint32_t expect_status = KERNEL_STATUS_OK)
{
    auto calc_size = [](const vector<int64_t>& shape) -> uint64_t {
        return shape.empty() ? 1 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
    };

    const uint64_t in0_size = calc_size(shapes[0]);
    const uint64_t in1_size = calc_size(shapes[1]);
    const uint64_t out_size = calc_size(shapes[2]);

    auto x1_data = make_unique<T[]>(in0_size);
    auto x2_data = make_unique<T[]>(in1_size);
    auto output_data = make_unique<T[]>(out_size);

    for (uint64_t i = 0; i < in0_size; ++i) {
        x1_data[i] = input1[i];
    }
    for (uint64_t i = 0; i < in1_size; ++i) {
        x2_data[i] = input2[i];
    }
    for (uint64_t i = 0; i < out_size; ++i) {
        output_data[i] = T();
    }

    vector<void*> datas = {static_cast<void*>(x1_data.get()), static_cast<void*>(x2_data.get()),
                           static_cast<void*>(output_data.get())};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, expect_status);

    if (expect_status == KERNEL_STATUS_OK) {
        auto expect = make_unique<T[]>(out_size);
        for (uint64_t i = 0; i < out_size; ++i) {
            expect[i] = expect_output[i];
        }
        EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), out_size));
    }
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> x2 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    vector<float> expect = {6.0f, 10.0f, 12.0f, 12.0f, 10.0f, 6.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT32_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<int32_t> x1 = {1, 2, 3, 4, 5, 6};
    vector<int32_t> x2 = {6, 5, 4, 3, 2, 1};
    vector<int32_t> expect = {6, 10, 12, 12, 10, 6};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, DOUBLE_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<double> x1 = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
    vector<double> x2 = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    vector<double> expect = {3.0, 5.0, 7.0, 9.0, 11.0, 13.0};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT16_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<Eigen::half> x1 = {Eigen::half(1.0), Eigen::half(2.0), Eigen::half(3.0),
                              Eigen::half(4.0), Eigen::half(5.0), Eigen::half(6.0)};
    vector<Eigen::half> x2 = {Eigen::half(6.0), Eigen::half(5.0), Eigen::half(4.0),
                              Eigen::half(3.0), Eigen::half(2.0), Eigen::half(1.0)};
    vector<Eigen::half> expect = {Eigen::half(6.0),  Eigen::half(10.0), Eigen::half(12.0),
                                  Eigen::half(12.0), Eigen::half(10.0), Eigen::half(6.0)};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, COMPLEX64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<complex<float>> x1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    vector<complex<float>> x2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    vector<complex<float>> expect = {{1.0f * 5.0f - 2.0f * 6.0f, 1.0f * 6.0f + 2.0f * 5.0f},
                                     {3.0f * 7.0f - 4.0f * 8.0f, 3.0f * 8.0f + 4.0f * 7.0f}};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<float> x1 = {3.0f};
    vector<float> x2 = {4.0f};
    vector<float> expect = {12.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_X_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {2, 3}, {2, 3}};
    vector<float> x1 = {3.0f};
    vector<float> x2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> expect = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_Y_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {}, {2, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> x2 = {3.0f};
    vector<float> expect = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_BOTH_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 1}, {2, 2}};
    vector<float> x1 = {1.0f, 2.0f};
    vector<float> x2 = {3.0f, 4.0f};
    vector<float> expect = {3.0f, 6.0f, 4.0f, 8.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT64_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 2048}, {4, 2048}, {4, 2048}};
    vector<int64_t> x1(4 * 2048);
    vector<int64_t> x2(4 * 2048);
    vector<int64_t> expect(4 * 2048);
    for (int i = 0; i < 4 * 2048; ++i) {
        x1[i] = static_cast<int64_t>(i % 100);
        x2[i] = static_cast<int64_t>(2);
        expect[i] = x1[i] * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_INT8_UINT8_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_UINT8, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    int8_t x1[6] = {1, 2, 3, -1, -2, -3};
    uint8_t x2[6] = {1, 2, 3, 1, 2, 3};
    int16_t output[6] = {0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int16_t expect[6] = {1, 4, 9, static_cast<int16_t>(-1), static_cast<int16_t>(-4), static_cast<int16_t>(-9)};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(6)));
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_FLOAT_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    float x1[2] = {1.5f, 2.5f};
    double x2[2] = {2.0, 4.0};
    double output[2] = {0.0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    double expect[2] = {3.0, 10.0};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(2)));
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_UINT8_INT32_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    uint8_t x1[4] = {0, 100, 200, 255};
    int32_t x2[4] = {3, 3, 3, 3};
    int32_t output[4] = {0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect[4] = {0, 300, 600, 765};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(4)));
}

TEST_F(TEST_MUL_AICPU_UT, INPUT_DTYPE_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    bool x1[6] = {true};
    bool x2[6] = {true};
    bool output[6] = {false};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, BCAST_SHAPE_MISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 3}, {1, 2}, {1, 3}};
    float x1[3] = {1.0f, 2.0f, 3.0f};
    float x2[2] = {4.0f, 5.0f};
    float output[3] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(nullptr), static_cast<void*>(nullptr), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
