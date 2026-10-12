/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_SQRT_H
#define AICPU_KERNELS_NORMALIZED_SQRT_H

#include <complex>
#include "cpu_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {

class SqrtCpuKernel : public CpuKernel {
public:
    SqrtCpuKernel() = default;
    ~SqrtCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template <typename T>
    uint32_t DoComputeReal(const CpuKernelContext &ctx);

    template <typename T>
    uint32_t DoComputeComplex(const CpuKernelContext &ctx);
};

}  // namespace aicpu
#endif
