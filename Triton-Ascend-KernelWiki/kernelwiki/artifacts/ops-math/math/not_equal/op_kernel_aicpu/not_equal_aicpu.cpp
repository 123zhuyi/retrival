/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include "not_equal_aicpu.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kInputNum = 2;
const char *const kNotEqual = "NotEqual";
const int64_t kParallelDataNum = INT64_C(6) * INT64_C(1024);
const int64_t kParallelDataNumSameShape = INT64_C(7) * INT64_C(1024);
}

namespace aicpu {
template <typename T>
inline void NotEqualImpl(T a, T b, bool *output) {
  *output = !IsValueEqual<T>(a, b);
}

template <typename T>
uint32_t NotEqualBcastCompute(const CpuKernelContext &ctx, const Bcast &bcast) {
  auto x1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto x2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  bool *out = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    auto sharder_not_equal = [&x1, &x2, &out, &bcast](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        NotEqualImpl(*(x1 + bcast.GetBroadcastXIndex(i)), *(x2 + bcast.GetBroadcastYIndex(i)), (out + i));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 1, sharder_not_equal),
                        "Equal Compute failed.")
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      NotEqualImpl(*(x1 + bcast.GetBroadcastXIndex(i)), *(x2 + bcast.GetBroadcastYIndex(i)), (out + i));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void NotEqualSpecialCompute(BcastShapeType type, int64_t start, int64_t end, const CpuKernelContext &ctx) {
  auto x1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto x2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  bool *output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        NotEqualImpl(*(x1 + i), *(x2 + i), (output + i));
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        NotEqualImpl(*x1, *(x2 + i), (output + i));
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        NotEqualImpl(*(x1 + i), *x2, (output + i));
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
  }
}

template <typename T>
uint32_t NotEqualNoBcastCompute(const CpuKernelContext &ctx) {
  int64_t element_num_x1 = ctx.Input(0)->NumElements();
  int64_t element_num_x2 = ctx.Input(1)->NumElements();
  int64_t data_num = static_cast<int64_t>(ctx.Output(0)->NumElements());
  BcastShapeType type = (element_num_x1 == element_num_x2 ? BcastShapeType::SAME_SHAPE :
      (element_num_x1 == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT));
  if (data_num >= kParallelDataNumSameShape) {
    auto sharder_not_equal = [&type, &ctx](int64_t start, int64_t end) {
      NotEqualSpecialCompute<T>(type, start, end, ctx);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 1, sharder_not_equal),
                        "Equal Compute failed.")
  } else {
    NotEqualSpecialCompute<T>(type, 0, data_num, ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t NotEqualCompute(const CpuKernelContext &ctx) {
  Tensor *tensorx1 = ctx.Input(0);
  auto shapex1 = tensorx1->GetTensorShape()->GetDimSizes();
  Tensor *tensorx2 = ctx.Input(1);
  auto shapex2 = tensorx2->GetTensorShape()->GetDimSizes();
  Tensor *output = ctx.Output(0);
  KERNEL_LOG_INFO("CpuKernel[%s], input x1 : size[%lu], input x2: size[%lu], output: size[%lu]",
                   ctx.GetOpType().c_str(), tensorx1->GetDataSize(),
                   tensorx2->GetDataSize(), output->GetDataSize());

  bool no_need_bcast = (shapex1 == shapex2) || (tensorx1->NumElements() == 1) ||
                     (tensorx2->NumElements() == 1);
  if (no_need_bcast) {
    return NotEqualNoBcastCompute<T>(ctx);
  }

  Bcast bcast(shapex1, shapex2);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return NotEqualBcastCompute<T>(ctx, bcast);
}

template <typename T>
uint32_t NotEqualComputeCase(const CpuKernelContext &ctx) {
    uint32_t result = NotEqualCompute<T>(ctx);
    if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("NotEqual kernel compute failed, result = [%d].", result);
        return result;
    }
    return KERNEL_STATUS_OK;
}

uint32_t NotEqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check NotEqual params failed.");

  DataType x1_type = ctx.Input(0)->GetDataType();
  DataType x2_type = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE((x1_type == x2_type), KERNEL_STATUS_PARAM_INVALID,
                     "DataType of x1 [%d] should be same as x2 [%d].",
                     static_cast<int32_t>(x1_type), static_cast<int32_t>(x2_type))

  switch (x1_type) {
        case DT_INT8:
            return NotEqualComputeCase<int8_t>(ctx);
        case DT_INT16:
            return NotEqualComputeCase<int16_t>(ctx);
        case DT_INT32:
            return NotEqualComputeCase<int32_t>(ctx);
        case DT_INT64:
            return NotEqualComputeCase<int64_t>(ctx);
        case DT_UINT8:
            return NotEqualComputeCase<uint8_t>(ctx);
        case DT_FLOAT16:
            return NotEqualComputeCase<Eigen::half>(ctx);
        case DT_FLOAT:
            return NotEqualComputeCase<float>(ctx);
        case DT_DOUBLE:
            return NotEqualComputeCase<double>(ctx);
        case DT_BOOL:
            return NotEqualComputeCase<bool>(ctx);
        default:
            KERNEL_LOG_WARN("NotEqual kernel data type [%u] not support.", x1_type);
            return KERNEL_STATUS_PARAM_INVALID;
    }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNotEqual, NotEqualCpuKernel);
}  // namespace aicpu