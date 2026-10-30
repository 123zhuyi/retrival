/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_CROSS_H_
#define AICPU_KERNELS_NORMALIZED_CROSS_H_
#include "cpu_kernel.h"
#include "utils/status.h"
#include "utils/bcast.h"

namespace aicpu {
class CrossCpuKernel : public CpuKernel {
 public:
  CrossCpuKernel() = default;
  ~CrossCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T1>
  uint32_t CrossBcastCompute(const CpuKernelContext &ctx);

  template <typename T1>
  uint32_t CrossCompute(const CpuKernelContext &ctx, const BCalcInfo &calc_info);

  KernelStatus GetDimAndCheck(const CpuKernelContext &ctx, const BCalcInfo &calc_info);

  std::vector<int64_t> StridesCompute(int64_t stride_length,
                                      std::vector<int64_t> data_dims);

  void InitStride(const BCalcInfo &calc_info);
  template <typename T1>
  void SetResult(const BCalcInfo &calc_info, int64_t data_start_bcast) const;

  template <typename T1>
  void ParallelCompute(const BCalcInfo &calc_info, int64_t start, int64_t end);
  int64_t dim_ = 0;
  std::vector<int64_t> stride_; 
  int64_t data_stride_ = 0;
};
}  // namespace aicpu
#endif