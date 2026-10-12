/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CUMSUM_AICPU_H_
#define CUMSUM_AICPU_H_

#include "cpu_kernel.h"

namespace aicpu {
class CumsumCpuKernel : public CpuKernel {
public:
    CumsumCpuKernel() = default;
    ~CumsumCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    uint32_t CumsumCheck(const CpuKernelContext& ctx);

    void CumsumGetAttr(const CpuKernelContext& ctx, bool& exclusive, bool& reverse) const;

    template <typename T>
    uint32_t CumsumCompute(CpuKernelContext& ctx) const;

    void AxesCal(
        const CpuKernelContext& ctx, int64_t& inner, int64_t& outer, int64_t& depth, const int32_t& axis) const;
};
} // namespace aicpu
#endif
