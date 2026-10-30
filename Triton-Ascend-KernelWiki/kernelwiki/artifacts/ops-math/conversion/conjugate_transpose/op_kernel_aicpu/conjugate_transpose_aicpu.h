/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_CONJUGATETRANSPOSE_H_
#define AICPU_KERNELS_NORMALIZED_CONJUGATETRANSPOSE_H_

#include <vector>

#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {
class ConjugateTranspose : public CpuKernel {
public:
    ~ConjugateTranspose() = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    std::vector<int64_t> perm;
    KernelStatus ConjugateTransposeParamCheck(const CpuKernelContext& ctx);
    KernelStatus GetConjugateTransposeValue(Tensor* tensor, std::vector<int64_t>& value);

    template <typename T>
    KernelStatus ConjugateTransposeCompute(const CpuKernelContext& ctx);
};
} // namespace aicpu
#endif // AICPU_KERNELS_NORMALIZED_CONJUGATETRANSPOSE_H_
