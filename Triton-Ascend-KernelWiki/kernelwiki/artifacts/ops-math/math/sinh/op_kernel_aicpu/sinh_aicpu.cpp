/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sinh_aicpu.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kSinhInputNum{1};
const std::uint32_t kSinhOutputNum{1};
const std::uint32_t ParallelNum{4096};
const char *kSinh{"Sinh"};
}  // namespace

namespace internal {
template <typename T>
inline auto ScalarSinh(T x) -> T {
  return Eigen::numext::sinh(x);
}

template <>
inline Eigen::half ScalarSinh(Eigen::half x) {
  const Eigen::half val{Eigen::numext::sinh(static_cast<float>(x))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}
}  // namespace internal

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSinhKernel(const CpuKernelContext &ctx) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  const auto ScalarSinh = internal::ScalarSinh<T>;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (total_size > ParallelNum * sizeof(T)) {
    std::int64_t per_unit_size{total /
                               std::min(std::max(1L, cores - 2L), total)};
    return ParallelFor(ctx, total, per_unit_size,
                       [&](std::int64_t begin, std::int64_t end) {
                         std::transform(input + begin, input + end,
                                        output + begin, ScalarSinh);
                       });
  } else if (static_cast<int>(cores) != 0) {
    std::transform(input, input + total, output, ScalarSinh);
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSinh(const CpuKernelContext &ctx) {
  uint32_t result = ComputeSinhKernel<T>(ctx);
  if (static_cast<int>(result) != 0) {
    KERNEL_LOG_ERROR("Sinh compute failed.");
  }
  return result;
}

inline std::uint32_t SinhExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the ouput [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%llu] need be the same as the ouput "
        "[%llu].",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input_dims =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR(
        "The data dim size of the input [%llu] need be the same as the output "
        "[%llu].",
        input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR(
          "The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t SinhCheck(CpuKernelContext &ctx, uint32_t inputs_num,
                        uint32_t outputs_num) {
  (void)inputs_num;
  (void)outputs_num;
  return NormalCheck(ctx, kSinhInputNum, kSinhOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : SinhExtraCheck(ctx);
}

std::uint32_t SinhCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeSinh<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeSinh<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeSinh<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeSinh<std::complex<std::float_t> >(ctx);
    case DT_COMPLEX128:
      return ComputeSinh<std::complex<std::double_t> >(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SinhCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SinhCheck(ctx, kSinhInputNum, kSinhOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::SinhCompute(ctx);
}

REGISTER_CPU_KERNEL(kSinh, SinhCpuKernel);
}  // namespace aicpu
