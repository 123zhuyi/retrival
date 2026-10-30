/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_n_aicpu.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAddNInputNum{static_cast<uint32_t>(aicpu::kDynamicInput)};
const std::uint32_t kAddNOutputNum{1u};
const char *kAddN{"AddN"};
const std::int64_t kAddNParallelNum{16 * 1024};
}  // namespace

namespace aicpu {
namespace detail {

inline std::uint32_t ParallelForAddN(
    const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
    const std::function<void(std::int64_t, std::int64_t)> &work) {
  const std::uint32_t result = [&]() {
    if (total > kAddNParallelNum)
      return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
    else
      work(0, total);
    return KERNEL_STATUS_OK;
  }();
  return result;
}

template <typename T>
inline std::uint32_t ComputeAddNKernel(const CpuKernelContext &ctx) {
  AttrValue *n_ptr{ctx.GetAttr("N")};
  KERNEL_CHECK_NULLPTR(n_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr N failed.");
  std::int64_t per_batch_elements{n_ptr->GetInt()};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Output(0)->NumElements()};
  KERNEL_CHECK_FALSE((total != 0L), KERNEL_STATUS_PARAM_INVALID, "Output element number can not be 0.");
  auto cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAddN(
      ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i{begin}; i < end; i++) {
          output[i] = static_cast<T>(0);
          for (std::int64_t j{0}; j < per_batch_elements; j++) {
            output[i] += static_cast<T *>(ctx.Input(j)->GetData())[i];
          }
        }
      });
}

template <typename T>
inline std::uint32_t ComputeAddN(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAddNKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("AddN compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAddN(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the output [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *n_ptr = ctx.GetAttr("N");
  KERNEL_CHECK_NULLPTR(n_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr N failed.");
  std::int64_t per_batch_elements = n_ptr->GetInt();
  std::uint32_t actual_inputs_num = ctx.GetInputsSize();
  KERNEL_CHECK_FALSE((per_batch_elements == static_cast<std::int64_t>(actual_inputs_num)),
                      KERNEL_STATUS_PARAM_INVALID,
                      "Attr N [%ld] does not match actual input number [%u].",
                      per_batch_elements, actual_inputs_num);
  for (std::int64_t j = 0; j < per_batch_elements; j++) {
    KERNEL_CHECK_NULLPTR(ctx.Input(j), KERNEL_STATUS_PARAM_INVALID,
                          "Input [%u] is null.", j);
    KERNEL_CHECK_NULLPTR(ctx.Input(j)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                          "Input [%u] data is null.", j);
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%llu] need be the same as the output "
        "[%llu].",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAddN(CpuKernelContext &ctx, std::uint32_t inputs_num,
                                std::uint32_t outputs_num) {
  return NormalCheck(ctx, kAddNInputNum, kAddNOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckAddN(ctx);
}

inline std::uint32_t ComputeAddN(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_INT8:
      return ComputeAddN<std::int8_t>(ctx);
    case DT_INT16:
      return ComputeAddN<std::int16_t>(ctx);
    case DT_INT32:
      return ComputeAddN<std::int32_t>(ctx);
    case DT_INT64:
      return ComputeAddN<std::int64_t>(ctx);
    case DT_UINT8:
      return ComputeAddN<std::uint8_t>(ctx);
    case DT_UINT16:
      return ComputeAddN<std::uint16_t>(ctx);
    case DT_UINT32:
      return ComputeAddN<std::uint32_t>(ctx);
    case DT_UINT64:
      return ComputeAddN<std::uint64_t>(ctx);
    case DT_FLOAT16:
      return ComputeAddN<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAddN<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAddN<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeAddN<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeAddN<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

}  // namespace detail

std::uint32_t AddNCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAddN(ctx, kAddNInputNum, kAddNOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::ComputeAddN(ctx);
}

REGISTER_CPU_KERNEL(kAddN, AddNCpuKernel);
}  // namespace aicpu
