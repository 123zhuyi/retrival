/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_MATH_CONVERSION_SLICE_WRITE_AICPU_H
#define OPS_MATH_CONVERSION_SLICE_WRITE_AICPU_H

#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {
class SliceWriteCpuKernel : public CpuKernel {
 public:
  SliceWriteCpuKernel() = default;
  ~SliceWriteCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ValidateInputOutput(CpuKernelContext &ctx, Tensor *&x, Tensor *&begin,
                               const Tensor *&value, Tensor *&output);
  bool CheckValueSupported(const DataType input_x_type) const;
  KernelStatus Check(const Tensor *x, const Tensor *value,
    int64_t row_offset, int64_t col_offset);
  KernelStatus GetBeginValue(const Tensor *begin, int64_t &row_offset,
    int64_t &col_offset);
};
}  // namespace aicpu
#endif