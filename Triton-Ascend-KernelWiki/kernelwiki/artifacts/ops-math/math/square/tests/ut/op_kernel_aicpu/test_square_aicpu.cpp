/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <malloc.h>
#include <Eigen/Core>
#include <complex>
#include <iostream>
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_SQUARE_UT : public testing::Test {
protected:
    std::float_t* float_null_{nullptr};
    std::float_t float_0_[0];
    std::float_t float_12_[12]{1.0f};
    std::float_t float_16_[16]{0.0f};
    std::int32_t int32_22_[22]{1};
    std::int64_t int64_22_[22]{0L};
    bool bool_22_[22]{true};
};

template <typename T>
inline aicpu::DataType ToDataType()
{
    return aicpu::DataType::DT_UNDEFINED;
}

template <>
inline aicpu::DataType ToDataType<bool>()
{
    return aicpu::DataType::DT_BOOL;
}

template <>
inline aicpu::DataType ToDataType<Eigen::half>()
{
    return aicpu::DataType::DT_FLOAT16;
}

template <>
inline aicpu::DataType ToDataType<std::float_t>()
{
    return aicpu::DataType::DT_FLOAT;
}

template <>
inline aicpu::DataType ToDataType<std::double_t>()
{
    return aicpu::DataType::DT_DOUBLE;
}

template <>
inline aicpu::DataType ToDataType<std::int8_t>()
{
    return aicpu::DataType::DT_INT8;
}

template <>
inline aicpu::DataType ToDataType<std::int16_t>()
{
    return aicpu::DataType::DT_INT16;
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
inline aicpu::DataType ToDataType<std::uint8_t>()
{
    return aicpu::DataType::DT_UINT8;
}

template <>
inline aicpu::DataType ToDataType<std::uint16_t>()
{
    return aicpu::DataType::DT_UINT16;
}

template <>
inline aicpu::DataType ToDataType<std::uint32_t>()
{
    return aicpu::DataType::DT_UINT32;
}

template <>
inline aicpu::DataType ToDataType<std::uint64_t>()
{
    return aicpu::DataType::DT_UINT64;
}

template <>
inline aicpu::DataType ToDataType<std::complex<std::float_t>>()
{
    return aicpu::DataType::DT_COMPLEX64;
}
template <>
inline aicpu::DataType ToDataType<std::complex<std::double_t>>()
{
    return aicpu::DataType::DT_COMPLEX128;
}

inline std::uint64_t SizeOf(std::vector<std::int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <std::shared_ptr<aicpu::Device> aicpu::CpuKernelContext::* DEVICE_PTR>
struct Friend {
    friend void SetDeviceNull(aicpu::CpuKernelContext& ctx)
    {
        ctx.*DEVICE_PTR = nullptr;
    }
};

template struct Friend<&aicpu::CpuKernelContext::device_>;
void SetDeviceNull(aicpu::CpuKernelContext& ctx);

inline void RunKernelSquare(
    std::shared_ptr<aicpu::NodeDef> node_def, aicpu::DeviceType device_type, uint32_t expect, bool bad_kernel = false)
{
    std::string node_def_str;
    node_def->SerializeToString(node_def_str);
    aicpu::CpuKernelContext ctx(device_type);
    EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KERNEL_STATUS_OK);
    if (bad_kernel) {
        SetDeviceNull(ctx);
    }
    std::uint32_t ret{aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx)};
    EXPECT_EQ(ret, expect);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquare(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    const auto data_type_in{ToDataType<Tin>()};
    const auto data_type_out{ToDataType<Tout>()};
    EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
    EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Square", "Square")
        .Input({"x", data_type_in, dims_in, input})
        .Output({"y", data_type_out, dims_out, output});
    RunKernelSquare(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquare(
    const std::vector<std::int64_t>& dims, Tin* input, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    CreateAndRunKernelSquare(dims, dims, input, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareParamInvalid(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input, Tout* output)
{
    CreateAndRunKernelSquare(dims_in, dims_out, input, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareParamInvalid(const std::vector<std::int64_t>& dims, Tin* input, Tout* output)
{
    CreateAndRunKernelSquareParamInvalid(dims, dims, input, output);
}

template <typename T>
void RunTestSquare(
    const std::uint64_t* dim_data, const std::uint64_t* shape_data, const T* input_data, const T* output_exp_data)
{
    std::uint64_t dim[1];
    dim[0] = dim_data[0];

    std::uint64_t shape[dim[0]];
    for (std::uint64_t i = 0; i < dim[0]; i++) {
        shape[i] = shape_data[i];
    }

    std::vector<std::int64_t> dims(shape, shape + dim[0]);
    auto input1_size{SizeOf(dims)};

    T* data1 = (T*)malloc(input1_size * sizeof(T));
    if (data1) {
        for (std::uint64_t i = 0; i < input1_size; i++) {
            data1[i] = input_data[i];
        }
    }

    T* output = (T*)malloc(input1_size * sizeof(T));
    CreateAndRunKernelSquare(dims, data1, output);

    T* expect_out = (T*)malloc(input1_size * sizeof(T));
    if (expect_out) {
        for (std::uint64_t i = 0; i < input1_size; i++) {
            expect_out[i] = output_exp_data[i];
        }
    }

    EXPECT_EQ(CompareResult(output, expect_out, input1_size), true);
    free(data1);
    free(output);
    free(expect_out);
}

TEST_F(TEST_SQUARE_UT, INPUT_SHAPE_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DIM_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({2, 6}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DIMSIZE_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({2, 2, 3}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DTYPE_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_SQUARE_UT, INPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_SQUARE_UT, OUTPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelSquareParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_SQUARE_UT, NO_OUTPUT_EXCEPTION)
{
    const auto data_type_in{ToDataType<std::float_t>()};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Square", "Square").Input({"x", data_type_in, {2, 6}, float_12_});
    RunKernelSquare(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQUARE_UT, INPUT_BOOL_UNSUPPORT)
{
    CreateAndRunKernelSquareParamInvalid({2, 11}, bool_22_, bool_22_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DOUBLE_SUCC)
{
    const std::uint64_t dim_data[] = {3};
    const std::uint64_t shape_data[] = {3, 2, 3};
    const std::double_t input_data[] = {-16402.371557568098, 31184.491305749427, 20654.268657188688, -99896.5374553138,
                                        -78771.8193964102,   10158.863572535221, 66828.98600681915,  81997.93141635446,
                                        35223.615180814,     -54309.72660861235, -16453.66222489126, -52077.48775084444,
                                        1879.8105129145988,  60066.19899461744,  -98088.152813315,   -16334.09901193701,
                                        98939.76615307,      5456.29496717015};

    const std::double_t output_exp_data[] = {
        269037792.71251893, 972472497.9983616, 426598813.763327,  9979318195.560911,  6204999531.020665,
        103202509.08538309, 4466113370.69963,  6723660756.561171, 1240703066.4060705, 2949546404.302216,
        270723000.6108136,  2712064730.439353, 3533687.564464247, 3607948261.6609817, 9621285722.328236,
        266802790.53176165, 9789077326.424175, 29771154.768766306};

    RunTestSquare<std::double_t>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_COMPLEX64_SUCC)
{
    const std::uint64_t dim_data[] = {3};
    const std::uint64_t shape_data[] = {3, 2, 3};
    const std::complex<std::float_t> input_data[] = {
        {-1.6402372f, 0.0f}, {3.1184492f, 0.0f}, {2.0654268f, 0.0f},  {-9.989654f, 0.0f}, {-7.877182f, 0.0f},
        {1.0158863f, 0.0f},  {6.6828985f, 0.0f}, {8.199793f, 0.0f},   {3.5223615f, 0.0f}, {-5.4309726f, 0.0f},
        {-1.6453662f, 0.0f}, {-5.207749f, 0.0f}, {0.18798105f, 0.0f}, {6.00662f, 0.0f},   {-9.808815f, 0.0f},
        {-1.6334099f, 0.0f}, {9.893976f, 0.0f},  {0.5456295f, 0.0f}};

    const std::complex<std::float_t> output_exp_data[] = {
        {2.6903782f, 0.0f}, {9.724726f, 0.0f},  {4.265988f, 0.0f},   {99.79318f, 0.0f},  {62.049995f, 0.0f},
        {1.032025f, 0.0f},  {44.661133f, 0.0f}, {67.2366f, 0.0f},    {12.407031f, 0.0f}, {29.495462f, 0.0f},
        {2.7072299f, 0.0f}, {27.12065f, 0.0f},  {0.03533688f, 0.0f}, {36.079483f, 0.0f}, {96.21285f, 0.0f},
        {2.6680279f, 0.0f}, {97.89076f, 0.0f},  {0.29771155f, 0.0f}};

    RunTestSquare<std::complex<std::float_t>>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_INT32_SUCC)
{
    const std::uint64_t dim_data[] = {2};
    const std::uint64_t shape_data[] = {3, 4};
    const std::int32_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::int32_t output_exp_data[] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144};
    RunTestSquare<std::int32_t>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_INT64_SUCC)
{
    const std::uint64_t dim_data[] = {2};
    const std::uint64_t shape_data[] = {3, 4};
    const std::int64_t input_data[] = {1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L};
    const std::int64_t output_exp_data[] = {1L, 4L, 9L, 16L, 25L, 36L, 49L, 64L, 81L, 100L, 121L, 144L};
    RunTestSquare<std::int64_t>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_FLOAT16_SUCC)
{
    const std::uint64_t dim_data[] = {2};
    const std::uint64_t shape_data[] = {3, 4};
    const Eigen::half input_data[] = {Eigen::half(1.0f), Eigen::half(2.0f), Eigen::half(3.0f), Eigen::half(4.0f),
                                       Eigen::half(5.0f), Eigen::half(6.0f), Eigen::half(7.0f), Eigen::half(8.0f),
                                       Eigen::half(9.0f), Eigen::half(10.0f), Eigen::half(11.0f), Eigen::half(12.0f)};
    const Eigen::half output_exp_data[] = {Eigen::half(1.0f), Eigen::half(4.0f), Eigen::half(9.0f), Eigen::half(16.0f),
                                            Eigen::half(25.0f), Eigen::half(36.0f), Eigen::half(49.0f), Eigen::half(64.0f),
                                            Eigen::half(81.0f), Eigen::half(100.0f), Eigen::half(121.0f), Eigen::half(144.0f)};
    RunTestSquare<Eigen::half>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_FLOAT_SUCC)
{
    const std::uint64_t dim_data[] = {2};
    const std::uint64_t shape_data[] = {3, 4};
    const std::float_t input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::float_t output_exp_data[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f, 100.0f, 121.0f, 144.0f};
    RunTestSquare<std::float_t>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_COMPLEX128_SUCC)
{
    const std::uint64_t dim_data[] = {2};
    const std::uint64_t shape_data[] = {2, 3};
    const std::complex<std::double_t> input_data[] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    const std::complex<std::double_t> output_exp_data[] = {{-3.0, 4.0}, {-7.0, 24.0}, {-11.0, 60.0}, {-15.0, 112.0}, {-19.0, 180.0}, {-23.0, 264.0}};
    RunTestSquare<std::complex<std::double_t>>(dim_data, shape_data, input_data, output_exp_data);
}

TEST_F(TEST_SQUARE_UT, INPUT_DTYPE_UNSUPPORTED)
{
    const auto data_type_in{aicpu::DataType::DT_INT8};
    const auto data_type_out{aicpu::DataType::DT_INT8};
    std::int8_t input[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::int8_t output[12] = {0};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Square", "Square")
        .Input({"x", data_type_in, {3, 4}, input})
        .Output({"y", data_type_out, {3, 4}, output});
    RunKernelSquare(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

