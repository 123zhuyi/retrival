/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_CALC_BUCKETS_LIMIT_AND_OFFSET_H_
#define AICPU_KERNELS_CALC_BUCKETS_LIMIT_AND_OFFSET_H_

#include <cstdint>

#include "cpu_kernel.h"

namespace aicpu {
class CalcBucketsLimitAndOffsetCpuKernel : public CpuKernel {
public:
    CalcBucketsLimitAndOffsetCpuKernel() = default;
    ~CalcBucketsLimitAndOffsetCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    static constexpr const char* kOpName = "CalcBucketsLimitAndOffset";
    static constexpr uint32_t kInputNum = 3;
    static constexpr uint32_t kOutputNum = 2;

    uint32_t InitParams(const CpuKernelContext& ctx);
    template <typename T>
    uint32_t DoCompute();

    int64_t input_num_elements_[kInputNum]{0};
    void* datas_[kInputNum + kOutputNum]{nullptr};
    int64_t total_limit_{0};
};
} // namespace aicpu
#endif // AICPU_KERNELS_CALC_BUCKETS_LIMIT_AND_OFFSET_H_
