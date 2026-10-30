/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_
#define OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class MulCpuKernel : public CpuKernel {
public:
    MulCpuKernel() = default;
    ~MulCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    uint32_t MulCompute(const CpuKernelContext& ctx);

    template <typename T>
    uint32_t MulDispatch(BCalcInfo& calc_info);

    bool AlignedCheck(const BCalcInfo& calc_info) const;

    template <int32_t RANK, typename T>
    uint32_t MulCalculateWithAlignedCheck(BCalcInfo& calc_info);

    template <int32_t RANK, typename T, int32_t OPTION>
    uint32_t MulCalculate(BCalcInfo& calc_info);

    uint32_t MulSameTypeCompute(const CpuKernelContext& ctx);
};
} // namespace aicpu

#endif // OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_
