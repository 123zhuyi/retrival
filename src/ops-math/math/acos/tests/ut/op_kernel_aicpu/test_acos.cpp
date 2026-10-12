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

class TEST_ACOS_UT : public testing::Test {
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

inline void RunKernelAcos(
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
void CreateAndRunKernelAcos(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    const auto data_type_in{ToDataType<Tin>()};
    const auto data_type_out{ToDataType<Tout>()};
    EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
    EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Acos", "Acos")
        .Input({"x", data_type_in, dims_in, input0})
        .Output({"output", data_type_out, dims_out, output});
    RunKernelAcos(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcos(
    const std::vector<std::int64_t>& dims, Tin* input0, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    CreateAndRunKernelAcos(dims, dims, input0, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcosParamInvalid(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output)
{
    CreateAndRunKernelAcos(dims_in, dims_out, input0, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcosParamInvalid(const std::vector<std::int64_t>& dims, Tin* input0, Tout* output)
{
    CreateAndRunKernelAcosParamInvalid(dims, dims, input0, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcosInnerError(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input0, Tout* output)
{
    CreateAndRunKernelAcos(dims_in, dims_out, input0, output, aicpu::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcosInnerError(const std::vector<std::int64_t>& dims, Tin* input0, Tout* output)
{
    CreateAndRunKernelAcosInnerError(dims, dims, input0, output);
}

template <typename Tin, typename Tout>
void RunTestAcos(
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
    CreateAndRunKernelAcos(dims, data0, output);

    Tout expect_out[output_size];
    for (std::uint64_t i = 0; i < output_size; i++) {
        expect_out[i] = output_exp_data[i];
    }
    EXPECT_EQ(CompareResult(output, expect_out, output_size), true);
}

TEST_F(TEST_ACOS_UT, DATA_TYPE_DT_DOUBLE)
{
    const std::uint64_t dim_data[] = {3};
    const std::uint64_t shape_data[] = {2, 3, 3};
    const std::double_t input_data[] = {
        0.6267539549051224,  -0.14176291530865615, 0.3462959626471891,  -0.5394341160937997, -0.2828993869149192,
        -0.5521485489869222, -0.7945087511045936,  0.23826790303052459, 0.664065611902118,   -0.008286997928567796,
        0.37438623618393585, 0.3486716158412613,   -0.0727065893323624, 0.08758318294360046, 0.9734852974591486,
        0.4145011300751571,  0.8515187995597076,   0.8509662539769742};
    const std::double_t output_exp_data[] = {
        0.8934159068591445, 1.7130384166744665, 1.2171764545701869, 2.1405612424601053, 1.8576119667098079,
        2.155735355726101,  2.48899450816864,   1.3302143286567885, 0.844552929583032,  1.579083419577074,
        1.1870615428795213, 1.2146429261651235, 1.6435671265012362, 1.4831007831801721, 0.23079302292663156,
        1.143401771614123,  0.551921129577479,  0.5529740540846267};

    RunTestAcos<std::double_t>(dim_data, shape_data, input_data, output_exp_data);
}
// exception instance
TEST_F(TEST_ACOS_UT, BAD_KERNEL_EXCEPTION)
{
    CreateAndRunKernelAcosInnerError({2000, 6000}, float_0_, float_0_);
}

TEST_F(TEST_ACOS_UT, INPUT_SHAPE_EXCEPTION)
{
    CreateAndRunKernelAcosParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_ACOS_UT, INPUT_DTYPE_EXCEPTION)
{
    CreateAndRunKernelAcosParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_ACOS_UT, INPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelAcosParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_ACOS_UT, OUTPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelAcosParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_ACOS_UT, NO_OUTPUT_EXCEPTION)
{
    const auto data_type_in{ToDataType<std::float_t>()};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Acos", "Acos").Input({"x", data_type_in, {2, 6}, float_12_});
    RunKernelAcos(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ACOS_UT, INPUT_BOOL_UNSUPPORT)
{
    CreateAndRunKernelAcosParamInvalid({2, 11}, bool_22_, bool_22_);
}
