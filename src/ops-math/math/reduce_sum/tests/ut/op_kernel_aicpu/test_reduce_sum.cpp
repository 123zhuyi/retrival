/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"

namespace {
template <typename T>
inline aicpu::DataType ToDataType()
{
    return aicpu::DataType::DT_UNDEFINED;
}

template <>
inline aicpu::DataType ToDataType<std::int32_t>()
{
    return aicpu::DataType::DT_INT32;
}

template <>
inline aicpu::DataType ToDataType<std::int64_t>()
{
    return aicpu::DataType::DT_INT64;
}

template <>
inline aicpu::DataType ToDataType<double>()
{
    return aicpu::DataType::DT_DOUBLE;
}

template <>
inline aicpu::DataType ToDataType<std::complex<float>>()
{
    return aicpu::DataType::DT_COMPLEX64;
}

inline void RunKernelReduceSum(std::shared_ptr<aicpu::NodeDef> node_def, std::uint32_t expect_status)
{
    std::string node_def_str;
    node_def->SerializeToString(node_def_str);
    aicpu::CpuKernelContext ctx(aicpu::DeviceType::HOST);
    EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KERNEL_STATUS_OK);
    const std::uint32_t ret = aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, expect_status);
}

template <typename T, typename AxisT>
void CreateAndRunReduceSum(const std::vector<std::int64_t> &input_shape, const std::vector<std::int64_t> &axes_shape,
    const std::vector<std::int64_t> &output_shape, bool keep_dims, std::vector<T> &input, std::vector<AxisT> &axes,
    std::vector<T> &output, aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK,
    bool noop_with_empty_axes = true)
{
    const auto input_dtype = ToDataType<T>();
    const auto axes_dtype = ToDataType<AxisT>();
    ASSERT_NE(input_dtype, aicpu::DataType::DT_UNDEFINED);
    ASSERT_NE(axes_dtype, aicpu::DataType::DT_UNDEFINED);

    auto node_def = aicpu::CpuKernelUtils::CreateNodeDef();
    aicpu::NodeDefBuilder(node_def.get(), "ReduceSum", "ReduceSum")
        .Input({"x", input_dtype, input_shape, input.data()})
        .Input({"axes", axes_dtype, axes_shape, axes.empty() ? nullptr : axes.data()})
        .Output({"y", input_dtype, output_shape, output.data()})
        .Attr("keep_dims", keep_dims)
        .Attr("noop_with_empty_axes", noop_with_empty_axes);
    RunKernelReduceSum(node_def, status);
}
}  // namespace

class TEST_REDUCE_SUM_UT : public testing::Test {};

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT32_AXIS_INT32)
{
    const std::vector<std::int64_t> input_shape{2, 3};
    const std::vector<std::int64_t> axes_shape{1};
    const std::vector<std::int64_t> output_shape{2};
    std::vector<std::int32_t> input{1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> axes{1};
    std::vector<std::int32_t> output(output_shape[0], 0);
    const std::vector<std::int32_t> expect{6, 15};

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, false, input, axes, output);
    EXPECT_EQ(CompareResult(output.data(), const_cast<std::int32_t *>(expect.data()), expect.size()), true);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT64_AXIS_INT64_KEEP_DIMS)
{
    const std::vector<std::int64_t> input_shape{2, 3};
    const std::vector<std::int64_t> axes_shape{1};
    const std::vector<std::int64_t> output_shape{2, 1};
    std::vector<std::int64_t> input{1, 2, 3, 4, 5, 6};
    std::vector<std::int64_t> axes{-1};
    std::vector<std::int64_t> output(output_shape[0] * output_shape[1], 0);
    const std::vector<std::int64_t> expect{6, 15};

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, true, input, axes, output);
    EXPECT_EQ(CompareResult(output.data(), const_cast<std::int64_t *>(expect.data()), expect.size()), true);
}

TEST_F(TEST_REDUCE_SUM_UT, EMPTY_AXES_IS_NOOP)
{
    const std::vector<std::int64_t> input_shape{2, 3};
    const std::vector<std::int64_t> axes_shape{0};
    const std::vector<std::int64_t> output_shape{2, 3};
    std::vector<double> input{1.5, -2.0, 3.0, 4.5, 0.5, -1.0};
    std::vector<std::int32_t> axes;
    std::vector<double> output(input.size(), 0.0);
    const std::vector<double> expect = input;

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, false, input, axes, output);
    EXPECT_EQ(CompareResult(output.data(), const_cast<double *>(expect.data()), expect.size()), true);
}

TEST_F(TEST_REDUCE_SUM_UT, EMPTY_AXES_REDUCE_ALL_WHEN_NOOP_FALSE)
{
    const std::vector<std::int64_t> input_shape{2, 3};
    const std::vector<std::int64_t> axes_shape{0};
    const std::vector<std::int64_t> output_shape{1, 1};
    std::vector<std::int32_t> input{1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> axes;
    std::vector<std::int32_t> output(1, 0);
    const std::vector<std::int32_t> expect{21};

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, true, input, axes, output,
        aicpu::KERNEL_STATUS_OK, false);
    EXPECT_EQ(CompareResult(output.data(), const_cast<std::int32_t *>(expect.data()), expect.size()), true);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_COMPLEX64)
{
    const std::vector<std::int64_t> input_shape{2, 2};
    const std::vector<std::int64_t> axes_shape{1};
    const std::vector<std::int64_t> output_shape{2};
    std::vector<std::complex<float>> input{{1.0f, 2.0f}, {3.0f, -1.0f}, {-2.0f, 0.5f}, {4.0f, 1.0f}};
    std::vector<std::int32_t> axes{1};
    std::vector<std::complex<float>> output(output_shape[0]);
    const std::vector<std::complex<float>> expect{{4.0f, 1.0f}, {2.0f, 1.5f}};

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, false, input, axes, output);
    EXPECT_EQ(CompareResult(output.data(), const_cast<std::complex<float> *>(expect.data()), expect.size()), true);
}

TEST_F(TEST_REDUCE_SUM_UT, EMPTY_COMPLEX_INPUT_IS_INVALID)
{
    const std::vector<std::int64_t> input_shape{0};
    const std::vector<std::int64_t> axes_shape{1};
    const std::vector<std::int64_t> output_shape{1};
    std::vector<std::complex<float>> input{{0.0f, 0.0f}};
    std::vector<std::int32_t> axes{0};
    std::vector<std::complex<float>> output(output_shape[0], {1.0f, 1.0f});

    CreateAndRunReduceSum(input_shape, axes_shape, output_shape, false, input, axes, output,
        aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_REDUCE_SUM_UT, INVALID_AXIS)
{
    const std::vector<std::int64_t> input_shape{2, 3};
    const std::vector<std::int64_t> axes_shape{1};
    const std::vector<std::int64_t> output_shape{2};
    std::vector<std::int32_t> input{1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> axes{2};
    std::vector<std::int32_t> output(output_shape[0], 0);

    CreateAndRunReduceSum(
        input_shape, axes_shape, output_shape, false, input, axes, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}