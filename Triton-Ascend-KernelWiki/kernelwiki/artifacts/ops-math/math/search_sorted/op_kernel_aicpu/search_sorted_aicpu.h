/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_SEARCH_SORTED_H
#define AICPU_KERNELS_NORMALIZED_SEARCH_SORTED_H

#include <type_traits>
#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {

class SearchSortedKernel : public CpuKernel {
 public:
  ~SearchSortedKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  KernelStatus GetInputAndCheck(const CpuKernelContext &ctx);

  bool right_ = false;
  DataType sequence_dtype_ = DT_INT32;
  DataType values_dtype_ = DT_INT32;
  DataType output_dtype_ = DT_INT32;

  Tensor *sequence_t_ = nullptr;
  Tensor *values_t_ = nullptr;
  Tensor *sorter_t_ = nullptr;
  Tensor *output_t_ = nullptr;
};
}  // namespace aicpu
#endif
