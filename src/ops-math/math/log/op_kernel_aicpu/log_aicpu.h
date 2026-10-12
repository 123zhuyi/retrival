/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_MATH_LO_AICPU_H
#define OPS_MATH_MATH_LO_AICPU_H

#include "cpu_kernel.h"

namespace aicpu {
class LogCpuKernel final : public CpuKernel {
public:
  std::uint32_t Compute(CpuKernelContext &ctx) override;

private:
  uint32_t LogOpCheck(CpuKernelContext &ctx);
  
  float base_ = -1.0F;
  float shift_ = 0.0F;
  float scale_ = 1.0F;
};
} // namespace aicpu

#endif
