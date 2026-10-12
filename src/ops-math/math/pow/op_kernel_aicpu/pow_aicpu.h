/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_POW_H_
#define AICPU_KERNELS_NORMALIZED_POW_H_
#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class PowCpuKernel : public CpuKernel {
 public:
  PowCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

  template <typename TIn1, typename TIn2, typename TOut>
  static uint32_t PowCompute(CpuKernelContext &ctx);

 private:
  template <typename TIn1, typename TIn2, typename TOut>
  static void SpecialCompute(BcastShapeType type, int64_t start, int64_t end,
                             CpuKernelContext &ctx);

  template <typename TIn1, typename TIn2, typename TOut>
  static uint32_t NoBcastCompute(CpuKernelContext &ctx);

  template <typename TIn1, typename TIn2, typename TOut>
  static uint32_t BcastCompute(CpuKernelContext &ctx, Bcast &bcast);
};
}  // namespace aicpu
#endif