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

#include <Eigen/Core>
#include <iostream>

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_ATAN_UT : public testing::Test {
protected:
    std::float_t* float_null_{nullptr};
    std::float_t float_0_[0];
    std::float_t float_12_[12]{1.0f};
    std::float_t float_12_nan_[12]{NAN};
    std::complex<std::float_t> complex_float_0_[0];
    std::complex<std::float_t> complex_float_12_[12]{1.0f};
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

inline void RunKernelAtan(
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
void CreateAndRunKernelAtan(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    const auto data_type_in{ToDataType<Tin>()};
    const auto data_type_out{ToDataType<Tout>()};
    EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
    EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Atan", "Atan")
        .Input({"x", data_type_in, dims_in, input0})
        .Output({"output", data_type_out, dims_out, output});
    RunKernelAtan(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAtan(
    const std::vector<std::int64_t>& dims, Tin* input0, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    CreateAndRunKernelAtan(dims, dims, input0, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAtanParamInvalid(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output)
{
    CreateAndRunKernelAtan(dims_in, dims_out, input0, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAtanParamInvalid(const std::vector<std::int64_t>& dims, Tin* input0, Tout* output)
{
    CreateAndRunKernelAtanParamInvalid(dims, dims, input0, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAtanInnerError(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output)
{
    CreateAndRunKernelAtan(dims_in, dims_out, input0, output, aicpu::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAtanInnerError(const std::vector<std::int64_t>& dims, Tin* input0, Tout* output)
{
    CreateAndRunKernelAtanInnerError(dims, dims, input0, output);
}

template <typename Tin, typename Tout>
void RunTestAtan(
    const std::uint64_t* dim_data, const std::uint64_t* shape_data, const Tin* input_data, const Tout* output_exp_data)
{
    std::uint64_t dim[1];
    dim[0] = dim_data[0];

    std::uint64_t shape[dim[0]];
    for (std::uint64_t i = 0; i < dim[0]; i++) {
        shape[i] = shape_data[i];
    }

    std::vector<std::int64_t> dims(shape, shape + dim[0]);
    auto output_size{SizeOf(dims)};
    auto input_size{output_size};
    Tin data0[input_size];
    for (std::uint64_t i = 0; i < input_size; i++) {
        data0[i] = input_data[i];
    }

    Tout output[output_size];
    CreateAndRunKernelAtan(dims, data0, output);

    Tout expect_out[output_size];
    for (std::uint64_t i = 0; i < output_size; i++) {
        expect_out[i] = output_exp_data[i];
    }

    EXPECT_EQ(CompareResult(output, expect_out, output_size), true);
}

TEST_F(TEST_ATAN_UT, DATA_TYPE_DT_DOUBLE)
{
    const std::uint64_t dim_data[] = {3};
    const std::uint64_t shape_data[] = {2, 3, 3};
    const std::double_t input_data[] = {
        62.675395490512244,  -14.176291530865612, 34.62959626471891,  -53.943411609379964, -28.289938691491926,
        -55.214854898692224, -79.45087511045936,  23.826790303052462, 66.40656119021179,   -0.8286997928567814,
        37.43862361839359,   34.86716158412614,   -7.270658933236234, 8.758318294360052,   97.34852974591485,
        41.45011300751571,   85.15187995597074,   85.09662539769744};
    const std::double_t output_exp_data[] = {
        1.5548424560815512,  -1.5003726700038402, 1.541927316300742,   -1.5522605048663116, -1.5354627821331792,
        -1.5526872384631802, -1.558210597589787,  1.5288513791562992,  1.5557387119496027,  -0.6919974888861665,
        1.5440922925581044,  1.5421239054206541,  -1.4341147746468865, 1.4571114654460435,  1.5605243192876117,
        1.5466756191838582,  1.5590531446989462,  1.559045520370406};
    RunTestAtan<std::double_t>(dim_data, shape_data, input_data, output_exp_data);
}
// exception instance
TEST_F(TEST_ATAN_UT, BAD_KERNEL_EXCEPTION)
{
    CreateAndRunKernelAtanInnerError({2000, 6000}, float_0_, float_0_);
}

TEST_F(TEST_ATAN_UT, INPUT_SHAPE_EXCEPTION)
{
    CreateAndRunKernelAtanParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_ATAN_UT, INPUT_DTYPE_EXCEPTION)
{
    CreateAndRunKernelAtanParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_ATAN_UT, INPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelAtanParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_ATAN_UT, OUTPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelAtanParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_ATAN_UT, NO_OUTPUT_EXCEPTION)
{
    const auto data_type_in{ToDataType<std::float_t>()};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Atan", "Atan").Input({"x", data_type_in, {2, 6}, float_12_});
    RunKernelAtan(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ATAN_UT, INPUT_BOOL_UNSUPPORT)
{
    CreateAndRunKernelAtanParamInvalid({2, 11}, bool_22_, bool_22_);
}
