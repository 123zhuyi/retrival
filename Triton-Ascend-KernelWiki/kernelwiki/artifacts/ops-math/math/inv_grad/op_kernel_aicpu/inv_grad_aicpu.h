/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_INVGRAD_H_
#define AICPU_KERNELS_NORMALIZED_INVGRAD_H_

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class InvGradCpuKernel : public CpuKernel {
 public:
  InvGradCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t InvGradParamCheck(const CpuKernelContext &ctx) const;

  template <typename T>
  void SpecialCompute(int64_t start, int64_t end,
                      const T *input1, const T *input2, T *output) const;

  template <typename T>
  uint32_t NoBcastCompute(const CpuKernelContext &ctx) const;

  template <typename T>
  uint32_t InvGradCompute(const CpuKernelContext &ctx) const;
};
}  // namespace aicpu
#endif