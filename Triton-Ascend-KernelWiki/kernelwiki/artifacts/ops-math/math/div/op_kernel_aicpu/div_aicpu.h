/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DIV_AICPU_H_
#define OPS_MATH_DIV_AICPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class DivCpuKernel : public CpuKernel {
public:
    DivCpuKernel() = default;
    ~DivCpuKernel() override = default;

protected:
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    uint32_t DivParamCheck(CpuKernelContext& ctx);

    template <typename T>
    uint32_t DivParamCheckZero(CpuKernelContext& ctx);

    template <typename T>
    uint32_t SpecialComputeInt(
        BcastShapeType type, int64_t start, int64_t end, const T* input1, const T* input2, T* output);

    template <typename T>
    uint32_t SpecialCompute(
        BcastShapeType type, int64_t start, int64_t end, const T* input1, const T* input2, T* output);

    template <typename T>
    uint32_t NoBcastComputeInt(CpuKernelContext& ctx);

    template <typename T>
    uint32_t NoBcastCompute(CpuKernelContext& ctx);

    template <typename T>
    uint32_t BcastComputeInt(CpuKernelContext& ctx, Bcast& bcast);

    template <typename T>
    uint32_t BcastCompute(CpuKernelContext& ctx, Bcast& bcast);

    template <typename T>
    uint32_t DivComputeInt(CpuKernelContext& ctx);

    template <typename T>
    uint32_t DivCompute(CpuKernelContext& ctx);
};
} // namespace aicpu

#endif