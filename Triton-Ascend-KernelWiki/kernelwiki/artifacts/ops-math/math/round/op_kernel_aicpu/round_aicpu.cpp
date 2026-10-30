/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "round_aicpu.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *const kRound = "Round";

bool IsValueEqualForDouble(double a, double b) { 
  bool has_inf = std::isinf(a) || std::isinf(b);
  bool has_nan = std::isnan(a) || std::isnan(b);
  if (has_inf || has_nan) {
    return a == b;
  }
  constexpr double epsilon = 1e-14;
  return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
}

template <typename T>
inline auto DoCompute(const T &x) -> const T {
  T roundval = Eigen::numext::floor(x);
  const T fraction = x - roundval;
  const T point_five = T(.5);
  bool is_point_five = std::is_same<T, double>::value ?
        IsValueEqualForDouble(fraction, point_five) : aicpu::IsValueEqual<T>(fraction, point_five);
  if (fraction > point_five) {
    roundval += T(1.0);
  } else if (is_point_five) {
    const T nearest_even_int =
        roundval - T(2) * Eigen::numext::floor(point_five * x);
    bool is_odd = std::is_same<T, double>::value ? 
        IsValueEqualForDouble(nearest_even_int, T(1)) : aicpu::IsValueEqual<T>(nearest_even_int, T(1));
    if (is_odd) {
      roundval += T(1);
    }
  }
  return roundval;
}

template <typename T>
inline auto ScalarRoundWithDecimals(const T &x, const T &pow_decimals, const bool &neg_flag) -> const T {
  T roundval = neg_flag ? DoCompute<T>(x / pow_decimals) : DoCompute<T>(x * pow_decimals);
  return neg_flag ? roundval * pow_decimals : roundval / pow_decimals;
}
}

namespace aicpu {
template <typename T>
uint32_t RangeRound(CpuKernelContext &ctx, T *input, T *out, int64_t decimals) {
  uint32_t ret = 0;
  uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx));
  int64_t input_datasize = ctx.Input(0)->NumElements();
  auto per_unit_size = CeilMultiple(input_datasize, max_core_num);
  if (decimals == 0) {
    auto shard_copy = [&input, &out](int64_t start, int64_t end) {
      constexpr bool isInt = (Eigen::NumTraits<T>::IsInteger != 0);
      if (isInt) {
        for (int64_t i = start; i < end; ++i) {
          out[i] = input[i];
        }
      } else {
        for (int64_t i = start; i < end; ++i) {
          out[i] = DoCompute<T>(input[i]);
        }
      }
    };
    ret =  CpuKernelUtils::ParallelFor(ctx, input_datasize, per_unit_size, shard_copy);
  } else {
    bool neg_flag = false;
    if (decimals < 0) {
      decimals = -decimals;
      neg_flag = true;
    }
    const T pow_decimals = static_cast<T>(std::pow(10, decimals));
    auto shard_copy = [&input, &out, &pow_decimals, &neg_flag](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        out[i] = ScalarRoundWithDecimals<T>(input[i], pow_decimals, neg_flag);
      }
    };
    ret = CpuKernelUtils::ParallelFor(ctx, input_datasize, per_unit_size, shard_copy);
  }
  return ret;
}

bool RoundCpuKernel::CheckSupported(DataType input_type) const {
  switch (input_type) {
    case DT_FLOAT16:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
    case DT_INT64:
      return true;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%d]", static_cast<int32_t>(input_type));
      return false;
  }
}

uint32_t RoundCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID, "Get input[0] failed")
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID, "Get output[0] failed")
  auto inputdata = input_tensor->GetData();
  KERNEL_CHECK_NULLPTR(inputdata, KERNEL_STATUS_PARAM_INVALID, "Get input[0] data failed")
  auto outputdata = output_tensor->GetData();
  KERNEL_CHECK_NULLPTR(outputdata, KERNEL_STATUS_PARAM_INVALID, "Get output[0] data failed")
  auto decimals_attr = ctx.GetAttr("decimals");
  int64_t decimals = 0;
  if (decimals_attr != nullptr) {
    decimals = decimals_attr->GetInt();
  }
  DataType inputtype = input_tensor->GetDataType();
  if (!CheckSupported(inputtype)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t ret = 0;
  switch (inputtype) {
    case DT_FLOAT16:
      ret = RangeRound(ctx, static_cast<Eigen::half *>(inputdata),
                       static_cast<Eigen::half *>(outputdata), decimals);
      break;
    case DT_FLOAT:
      ret = RangeRound(ctx, static_cast<float *>(inputdata),
                       static_cast<float *>(outputdata), decimals);
      break;
    case DT_DOUBLE:
      ret = RangeRound(ctx, static_cast<double *>(inputdata),
                       static_cast<double *>(outputdata), decimals);
      break;
    case DT_INT32:
      ret = RangeRound(ctx, static_cast<int32_t *>(inputdata),
                       static_cast<int32_t *>(outputdata), decimals);
      break;
    case DT_INT64:
      ret = RangeRound(ctx, static_cast<int64_t *>(inputdata),
                       static_cast<int64_t *>(outputdata), decimals);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%d]", static_cast<int32_t>(inputtype));
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kRound, RoundCpuKernel);
}  // namespace aicpu
