/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_RIGHTSHIFT_H
#define AICPU_KERNELS_NORMALIZED_RIGHTSHIFT_H

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {

class RightShiftCpuKernel : public CpuKernel {
 public:
  RightShiftCpuKernel() = default;
  ~RightShiftCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t RightShiftParamCheck(const CpuKernelContext &ctx);

  template <typename T>
  void SpecialCompute(BcastShapeType type, int64_t start, int64_t end,
                      const T *input1, const T *input2, T *output);

  template <typename T>
  uint32_t NoBcastCompute(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t BcastCompute(const CpuKernelContext &ctx, const Bcast &bcast);

  template <typename T>
  uint32_t RightShiftCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif