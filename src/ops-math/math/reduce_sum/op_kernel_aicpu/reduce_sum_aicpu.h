/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_REDUCE_SUM_AICPU_H
#define AICPU_KERNELS_NORMALIZED_REDUCE_SUM_AICPU_H

#include <complex>
#include <vector>

#include "cpu_kernel.h"

namespace aicpu {
class ReduceSumCpuKernel : public CpuKernel {
public:
    ReduceSumCpuKernel() = default;
    ~ReduceSumCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t ReduceSumCheck(const CpuKernelContext &ctx) const;
    bool GetNoopWithEmptyAxes(const CpuKernelContext &ctx) const;

    template <typename T>
    uint32_t ReduceSumCompute(const CpuKernelContext &ctx);

    template <typename T>
    bool ReduceSumSimpleCompute(const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape,
        T *output_data, bool force_full_compute = false);

    template <typename T>
    bool IsReduceSumFullCompute(const CpuKernelContext &ctx, std::vector<int64_t> &input_shape);

    template <typename T>
    uint32_t ReduceSumLastAxes(
        const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape, T *output_data);

    template <typename T>
    uint32_t ReduceSumOneAxes(const T *input_data, std::vector<int64_t> &input_shape, T *output_data,
        int64_t output_num, std::vector<int64_t> &axes, uint32_t &axes_idx);

    template <typename T, typename T2>
    uint32_t ReduceSumCompute2(const CpuKernelContext &ctx);

    template <typename T, typename T2>
    bool ReduceSumSimpleCompute2(const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape,
        T *output_data, bool force_full_compute = false);

    template <typename T, typename T2>
    uint32_t ReduceSumOneAxes2(const T *input_data, int64_t input_num, std::vector<int64_t> input_shape,
        T *output_data, int64_t output_num, std::vector<int64_t> &axes, uint32_t &axes_idx);

    template <typename T>
    uint32_t ReduceSumDedupAxes(const CpuKernelContext &ctx, std::vector<int64_t> &axes);

    uint32_t ReduceSumParseAxes(std::vector<int64_t> &input_shape, std::vector<int64_t> &axes, uint32_t &axes_idx,
        int64_t &inner, int64_t &outer, int64_t &depth) const;
};
}  // namespace aicpu

#endif