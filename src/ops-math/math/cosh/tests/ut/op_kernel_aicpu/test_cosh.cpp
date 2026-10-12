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

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

#include "unsupported/Eigen/CXX11/Tensor"

class TEST_COSH_UT : public testing::Test {
 protected:
  std::float_t *float_null_{nullptr};
  std::float_t float_0_[0];
  std::float_t float_12_[12]{1.0f};
  std::float_t float_16_[16]{0.0f};
  std::int32_t int32_22_[22]{1};
  std::int64_t int64_22_[22]{0L};
  bool bool_22_[22]{true};
};

namespace {

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

inline std::uint64_t SizeOf(const std::vector<std::int64_t> &shape)
{
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>());
}

template <std::shared_ptr<aicpu::Device> aicpu::CpuKernelContext::*DEVICE_PTR>
struct Friend {
  friend void SetDeviceNull(aicpu::CpuKernelContext &ctx)
  {
    ctx.*DEVICE_PTR = nullptr;
  }
};

template struct Friend<&aicpu::CpuKernelContext::device_>;
void SetDeviceNull(aicpu::CpuKernelContext &ctx);

inline void RunKernelCosh(
    std::shared_ptr<aicpu::NodeDef> node_def, aicpu::DeviceType device_type, std::uint32_t expect,
    bool bad_kernel = false)
{
  std::string node_def_str;
  node_def->SerializeToString(node_def_str);
  aicpu::CpuKernelContext ctx(device_type);
  EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KERNEL_STATUS_OK);
  if (bad_kernel) {
    SetDeviceNull(ctx);
  }
  const std::uint32_t ret{aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx)};
  EXPECT_EQ(ret, expect);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCosh(
    const std::vector<std::int64_t> &dims_in, const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output,
    aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK, bool bad_kernel = false)
{
  const auto data_type_in{ToDataType<Tin>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Cosh", "Cosh")
      .Input({"x", data_type_in, dims_in, input})
      .Output({"y", data_type_out, dims_out, output});
  RunKernelCosh(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCosh(
    const std::vector<std::int64_t> &dims, Tin *input, Tout *output, aicpu::KernelStatus status = aicpu::KERNEL_STATUS_OK,
    bool bad_kernel = false)
{
  CreateAndRunKernelCosh(dims, dims, input, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCoshParamInvalid(
    const std::vector<std::int64_t> &dims_in, const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output)
{
  CreateAndRunKernelCosh(dims_in, dims_out, input, output, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCoshParamInvalid(const std::vector<std::int64_t> &dims, Tin *input, Tout *output)
{
  CreateAndRunKernelCoshParamInvalid(dims, dims, input, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCoshInnerError(
    const std::vector<std::int64_t> &dims_in, const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output)
{
  CreateAndRunKernelCosh(dims_in, dims_out, input, output, aicpu::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelCoshInnerError(const std::vector<std::int64_t> &dims, Tin *input, Tout *output)
{
  CreateAndRunKernelCoshInnerError(dims, dims, input, output);
}

template <typename T>
void CalcExpect(const T input[], T output[], std::uint64_t num)
{
  std::transform(input, input + num, output, [](const T &x) {
    if constexpr (std::is_same_v<T, Eigen::half>) {
      const Eigen::half val{static_cast<Eigen::half>(std::cosh(static_cast<float>(x)))};
      return Eigen::half_impl::isnan(val) ? Eigen::half(0.0f) : val;
    } else {
      using std::cosh;
      return cosh(x);
    }
  });
}

template <typename T>
void RunTestCosh(const std::vector<std::int64_t> &dims, const T *input_data)
{
  auto shape{dims};
  const auto num{SizeOf(shape)};
  std::vector<T> input(num);
  std::copy(input_data, input_data + num, input.begin());
  std::vector<T> output(num);
  std::vector<T> expect_out(num);

  CreateAndRunKernelCosh(dims, input.data(), output.data());
  CalcExpect(input.data(), expect_out.data(), num);
  EXPECT_EQ(CompareResult(output.data(), expect_out.data(), num), true);
}

}  // namespace

TEST_F(TEST_COSH_UT, DATA_TYPE_DT_FLOAT16)
{
  const std::vector<std::int64_t> dims{2, 3};
  const Eigen::half input_data[] = {Eigen::half(0.0f), Eigen::half(0.5f), Eigen::half(-1.0f),
                                    Eigen::half(1.2f), Eigen::half(-0.75f), Eigen::half(0.25f)};
  RunTestCosh(dims, input_data);
}

TEST_F(TEST_COSH_UT, DATA_TYPE_DT_FLOAT)
{
  const std::vector<std::int64_t> dims{2, 3};
  const std::float_t input_data[] = {0.0f, 0.5f, -1.0f, 1.2f, -0.75f, 0.25f};
  RunTestCosh(dims, input_data);
}

TEST_F(TEST_COSH_UT, DATA_TYPE_DT_DOUBLE)
{
  const std::vector<std::int64_t> dims{2, 3};
  const std::double_t input_data[] = {0.0, 0.5, -1.0, 1.2, -0.75, 0.25};
  RunTestCosh(dims, input_data);
}

TEST_F(TEST_COSH_UT, DATA_TYPE_DT_COMPLEX64)
{
  const std::vector<std::int64_t> dims{4};
  const std::complex<std::float_t> input_data[] = {std::complex<std::float_t>(0.0f, 0.0f),
                                                   std::complex<std::float_t>(0.3f, -0.2f),
                                                   std::complex<std::float_t>(-0.7f, 0.4f),
                                                   std::complex<std::float_t>(1.1f, -0.3f)};
  RunTestCosh(dims, input_data);
}

TEST_F(TEST_COSH_UT, DATA_TYPE_DT_COMPLEX128)
{
  const std::vector<std::int64_t> dims{4};
  const std::complex<std::double_t> input_data[] = {std::complex<std::double_t>(0.0, 0.0),
                                                    std::complex<std::double_t>(0.3, -0.2),
                                                    std::complex<std::double_t>(-0.7, 0.4),
                                                    std::complex<std::double_t>(1.1, -0.3)};
  RunTestCosh(dims, input_data);
}

TEST_F(TEST_COSH_UT, BAD_KERNEL_EXCEPTION)
{
  std::vector<std::float_t> input(257 * 257, 1.0f);
  std::vector<std::float_t> output(257 * 257, 0.0f);
  CreateAndRunKernelCoshInnerError({257, 257}, input.data(), output.data());
}

TEST_F(TEST_COSH_UT, INPUT_SHAPE_EXCEPTION)
{
  CreateAndRunKernelCoshParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_COSH_UT, INPUT_DTYPE_EXCEPTION)
{
  CreateAndRunKernelCoshParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_COSH_UT, INPUT_NULL_EXCEPTION)
{
  CreateAndRunKernelCoshParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_COSH_UT, OUTPUT_NULL_EXCEPTION)
{
  CreateAndRunKernelCoshParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_COSH_UT, NO_OUTPUT_EXCEPTION)
{
  const auto data_type_in{ToDataType<std::float_t>()};
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Cosh", "Cosh").Input({"x", data_type_in, {2, 6}, float_12_});
  RunKernelCosh(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COSH_UT, INPUT_BOOL_UNSUPPORT)
{
  CreateAndRunKernelCoshParamInvalid({2, 11}, bool_22_, bool_22_);
}
