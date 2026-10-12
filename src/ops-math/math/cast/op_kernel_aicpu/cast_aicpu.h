/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_CAST_AICPU_H_
#define AICPU_KERNELS_NORMALIZED_CAST_AICPU_H_

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "Eigen/Core"

namespace aicpu {

// ============ hif8 type definition (minimal exposure) ============
namespace hif8_impl {
struct Hif8Raw {
  int8_t val;
  Hif8Raw();
};
struct Hif8Base : public Hif8Raw {
  Hif8Base();
  explicit Hif8Base(const Hif8Raw &v);
};
Hif8Raw Fp32ToHif8Rtne(float f);
Hif8Raw Fp16ToHif8Rtne(Eigen::half f);
Hif8Raw Bf16ToHif8Rtne(Eigen::bfloat16 f);
} // namespace hif8_impl

struct hif8 : public hif8_impl::Hif8Base {
  explicit hif8() {}
  explicit hif8(float f);
  explicit hif8(Eigen::half f);
  explicit hif8(Eigen::bfloat16 f);
};

// ============ CastCpuKernel class definition ============
class CastCpuKernel : public CpuKernel {
 public:
  CastCpuKernel();
  ~CastCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  void SetMap();
  void SetInt8Map();
  void SetInt16Map();
  void SetInt32Map();
  void SetInt64Map();
  void SetFloat16Map();
  void SetFloatMap();
  void SetDoubleMap();
  void SetUInt8Map();
  void SetUInt16Map();
  void SetUInt32Map();
  void SetUInt64Map();
  void SetBoolMap();
  void SetComplexMap();
  void SetBfloat16Map();
  void SetHif8Map();
  uint32_t ValidateTensors(CpuKernelContext &ctx);
  uint32_t ValidateDataType();
  uint32_t ExecuteCast(CpuKernelContext &ctx);
  std::map<int, std::map<int, std::function<void(Tensor *&, Tensor *&,
                                                        int64_t &, int64_t &)>>>
      calls_;
  Tensor *x_tensor_;
  Tensor *y_tensor_;
  DataType x_data_type_;
  DataType y_data_type_;
  uint64_t x_data_size_;
  uint64_t y_data_size_;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_CAST_AICPU_H_