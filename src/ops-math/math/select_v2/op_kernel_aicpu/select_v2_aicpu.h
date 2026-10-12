/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_SELECTV2_CPU_KERNEL_H
#define AICPU_KERNELS_NORMALIZED_SELECTV2_CPU_KERNEL_H

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"

namespace aicpu {
struct SelectV2BCalcInfo {
  SelectV2BCalcInfo()
      : input_0(nullptr), input_1(nullptr), input_2(nullptr), output(nullptr) {}
  Tensor *input_0;
  Tensor *input_1;
  Tensor *input_2;
  Tensor *output;
  std::vector<int64_t> reshape_0;
  std::vector<int64_t> reshape_1;
  std::vector<int64_t> reshape_2;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> bcast_0;
  std::vector<int64_t> bcast_1;
  std::vector<int64_t> bcast_2;
};
class Selectv2CpuKernel : public CpuKernel {
public:
  Selectv2CpuKernel() = default;
  ~Selectv2CpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
  std::vector<int64_t> x_reshape_;
  std::vector<int64_t> y_reshape_;
  std::vector<int64_t> z_reshape_;
  std::vector<int64_t> shape_out_;
  std::vector<int64_t> x_bcast_;
  std::vector<int64_t> y_bcast_;
  std::vector<int64_t> z_bcast_;

private:
  KernelStatus Selectv2GenerateBcastInfo(const SelectV2BCalcInfo &calc_info);
  void SelectV2GetBcastVec(SelectV2BCalcInfo &calc_info) const;
  KernelStatus Selectv2ParamCheck(const CpuKernelContext &ctx) const;
  KernelStatus SelectV2BcastCheck(const int64_t x, const int64_t y,
                                  const int64_t z) const;

  template <typename T>
  KernelStatus Selectv2BuildBcast(const CpuKernelContext &ctx);

  bool AlignedCheck(const SelectV2BCalcInfo &calc_info) const;

  template <int32_t RANK, typename T>
  KernelStatus SelectV2CalculateWithAlignedCheck(SelectV2BCalcInfo &calc_info);

  template <int32_t RANK, typename T, int32_t OPTION>
  KernelStatus SelectV2Calculate(SelectV2BCalcInfo &calc_info);
  KernelStatus SelectV2BcastInfo(size_t max_size);
  enum Dim {
    SCALAR,
    ONE_DIM,
    TWO_DIM,
    THREE_DIM,
    FOUR_DIM,
    FIVE_DIM,
    SIX_DIM,
    SEVEN_DIM,
    EIGHT_DIM
  };
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_SELECTV2_CPU_KERNEL_H