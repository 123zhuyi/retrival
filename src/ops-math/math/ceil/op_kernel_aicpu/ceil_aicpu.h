/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_CEIL_H
#define AICPU_KERNELS_CEIL_H

#include "cpu_kernel.h"

namespace aicpu {
class CeilCpuKernel : public CpuKernel {
public:
    ~CeilCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    uint32_t ComputeCeil(Tensor* x, Tensor* y, uint64_t data_size, const CpuKernelContext& ctx) const;
};
} // namespace aicpu
#endif // AICPU_KERNELS_CEIL_H
