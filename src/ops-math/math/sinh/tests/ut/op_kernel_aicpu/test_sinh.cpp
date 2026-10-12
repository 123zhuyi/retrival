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
#include <complex>
#include <iostream>
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_SINH_UT : public testing::Test {
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

inline void RunKernelSinh(
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
void CreateAndRunKernelSinh(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    const auto data_type_in{ToDataType<Tin>()};
    const auto data_type_out{ToDataType<Tout>()};
    EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
    EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Sinh", "Sinh")
        .Input({"x", data_type_in, dims_in, input})
        .Output({"y", data_type_out, dims_out, output});
    RunKernelSinh(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSinh(
    const std::vector<std::int64_t>& dims, Tin* input, Tout* output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
    CreateAndRunKernelSinh(dims, dims, input, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSinhParamInvalid(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input, Tout* output)
{
    CreateAndRunKernelSinh(dims_in, dims_out, input, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSinhParamInvalid(const std::vector<std::int64_t>& dims, Tin* input, Tout* output)
{
    CreateAndRunKernelSinhParamInvalid(dims, dims, input, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSinhInnerError(
    const std::vector<std::int64_t>& dims_in, const std::vector<std::int64_t>& dims_out, Tin* input, Tout* output)
{
    CreateAndRunKernelSinh(dims_in, dims_out, input, output, aicpu::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSinhInnerError(const std::vector<std::int64_t>& dims, Tin* input, Tout* output)
{
    CreateAndRunKernelSinhInnerError(dims, dims, input, output);
}

template <typename T>
void RunTestSinh(
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
    T data1[input1_size];
    for (std::uint64_t i = 0; i < input1_size; i++) {
        data1[i] = input_data[i];
    }

    T output[input1_size];
    CreateAndRunKernelSinh(dims, data1, output);

    T expect_out[input1_size];
    for (std::uint64_t i = 0; i < input1_size; i++) {
        expect_out[i] = output_exp_data[i];
    }

    EXPECT_EQ(CompareResult(output, expect_out, input1_size), true);
}

// exception inssinhce
TEST_F(TEST_SINH_UT, BAD_KERNEL_EXCEPTION)
{
    CreateAndRunKernelSinhInnerError({2, 6}, float_12_, float_12_);
}

TEST_F(TEST_SINH_UT, INPUT_SHAPE_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_SINH_UT, INPUT_DIM_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({2, 6}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SINH_UT, INPUT_DIMSIZE_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({2, 2, 3}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SINH_UT, INPUT_DTYPE_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_SINH_UT, INPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({0, 0}, float_null_, float_null_);
}

TEST_F(TEST_SINH_UT, OUTPUT_NULL_EXCEPTION)
{
    CreateAndRunKernelSinhParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_SINH_UT, NO_OUTPUT_EXCEPTION)
{
    const auto data_type_in{ToDataType<std::float_t>()};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Sinh", "Sinh").Input({"x", data_type_in, {2, 6}, float_12_});
    RunKernelSinh(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_SINH_UT, NO_INPUT_EXCEPTION)
{
    const auto data_type_in{ToDataType<std::float_t>()};
    auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
    aicpu::NodeDefBuilder(node_def.get(), "Sinh", "Sinh").Input({"y", data_type_in, {2, 6}, float_12_});
    RunKernelSinh(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_SINH_UT, INPUT_BOOL_UNSUPPORT)
{
    CreateAndRunKernelSinhParamInvalid({2, 11}, bool_22_, bool_22_);
}

//  TEST_F(TEST_SINH_UT, FLOAT16_SUCC) {
//   const std::uint64_t dim_data[] = {2};
//   const std::uint64_t shape_data[] = {2, 10};
//   const Eigen::half input_data[] =
//   {3.76f,-0.8506f,2.078f,-3.236f,-1.697f,-3.312f,-4.766f,1.43f,3.984f,-0.0497f,2.246f,2.092f,-0.4363f,0.5254f,5.84f,2.486f,5.11f,5.105f,5.97f,5.824f};
//   const Eigen::half output_exp_data[] = {};

//   RunTestSinh<Eigen::half>();
// }
//  ADD_CASE(Eigen::half, DT_FLOAT16)
//  ADD_CASE(std::float_t, DT_FLOAT)
//  ADD_CASE(std::double_t, DT_DOUBLE)
//  ADD_CASE(std::complex<std::float_t>, DT_COMPLEX64)
//  ADD_CASE(std::complex<std::double_t>, DT_COMPLEX128)

// TEST_F(TEST_SINH_UT, COMPLEX128_SUCC)
// {
//     const std::uint64_t dim_data[] = {3};
//     const std::uint64_t shape_data[] = {2, 2, 2};
//     const std::complex<std::double_t> input_data[] = {
//         {626.1272009502172, 663.401546290216},    {-141.62115239334753 - 8.27871093063925},
//         {345.94966668454185, 374.0118499477519},  {-538.8946819777059, 348.3229442254201},
//         {-282.6164875280043 - 72.63388274303009}, {-551.5964004379352, 87.49559976065689},
//         {-793.714242353489, 972.5118121616895},   {238.02963512749398, 414.0866289450819}};

//     const std::complex<std::double_t> output_exp_data[] = {
//         {-3.627539321924039e+271, -2.1035490744014714e+271},
//         {6.59523653887469e+60, -1.4582938348725418e+61},
//         {-8.65474576177777e+149, -1.417344954753251e+149},
//         {5.0508727836068e+233, 2.0988964619810047e+233},
//         {2.547327391365503e+122, 1.0093262129865386e+122},
//         {-1.6018398745286443e+239, -8.116548979291175e+238},
//         {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
//         {9.76075696123225e+102, -6.729425000862294e+102}};

//     RunTestSinh<std::complex<std::double_t>>(dim_data, shape_data, input_data, output_exp_data);
// }

TEST_F(TEST_SINH_UT, INPUT_COMPLEX64_SUCC)
{
    const std::uint64_t dim_data[] = {3};
    const std::uint64_t shape_data[] = {5, 2, 3};
    const std::complex<std::float_t> input_data[] = {
        {6.2675395f, -0.73880816f}, {-1.4176291f, -7.09447f},     {3.4629595f, -5.5420628f}, {-5.394341f, -0.9415152f},
        {-2.8289938f, 8.635306f},   {-5.5214853f, 7.3453755f},    {-7.9450874f, 5.7894673f}, {2.382679f, 5.836191f},
        {6.640656f, 7.799314f},     {-0.08286998f, -7.54986f},    {3.7438624f, -8.039563f},  {3.4867163f, -5.760953f},
        {-0.7270659f, 4.9880633f},  {0.87583184f, 2.9628363f},    {9.734853f, 0.082913846f}, {4.1450114f, 0.04049822f},
        {8.515188f, -1.6889501f},   {8.509663f, 4.2291408f},      {9.948988f, 3.7656374f},   {9.706265f, 4.2289f},
        {-0.6232854f, 6.9682574f},  {-7.017069f, 8.263729f},      {5.6835365f, -3.8873646f}, {-0.7113509f, 3.3878508f},
        {8.911847f, -8.261057f},    {-2.7508345f, -0.026875649f}, {0.6667206f, 0.12001145f}, {0.57304317f, -0.2181459f},
        {-8.078061f, 2.0403728f},   {-8.676903f, 2.6348522f}};

    const std::complex<std::float_t> output_exp_data[] = {
        {194.86343f, -177.50362f},   {-1.3375545f, -1.5843607f},  {11.759096f, 10.782495f},
        {-64.78688f, -88.9949f},     {5.939871f, 6.0304003f},     {-60.870544f, 109.18243f},
        {-1242.3528f, -668.6013f},   {4.843115f, -2.361461f},     {20.916225f, 382.22733f},
        {-0.02484426f, -0.9573884f}, {-3.8967683f, -20.779251f},  {14.148078f, 8.157906f},
        {-0.21580727f, -1.2279775f}, {-0.97636837f, 0.25047404f}, {8419.151f, 699.6683f},
        {31.525381f, 1.2780616f},    {-294.1073f, -2477.5977f},   {-1152.9298f, -2197.118f},
        {-8492.992f, -6115.234f},    {-3816.6304f, -7269.03f},    {-0.5145191f, 0.7596623f},
        {222.19743f, 511.58627f},    {-107.975044f, 99.74209f},   {0.7495635f, -0.30809975f},
        {-1468.7549f, -3406.5322f},  {-7.7930927f, -0.21121131f}, {0.71206605f, 0.14733355f},
        {0.5905882f, -0.25293672f},  {729.21277f, 1437.0616f},    {2564.3481f, 1423.4365f}};

    RunTestSinh<std::complex<std::float_t>>(dim_data, shape_data, input_data, output_exp_data);
}