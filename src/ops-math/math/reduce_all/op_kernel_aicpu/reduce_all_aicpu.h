/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_REDUCE_ALL_AICPU_H
#define AICPU_KERNELS_NORMALIZED_REDUCE_ALL_AICPU_H

#include <map>
#include <vector>

#include "cpu_kernel.h"

namespace aicpu {
class ReduceAllCpuKernel : public CpuKernel {
public:
    ReduceAllCpuKernel() = default;
    ~ReduceAllCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template <typename T, typename T2>
    uint32_t ReduceAllCompute(const CpuKernelContext &ctx);

    uint32_t GenDataNoAxis(const CpuKernelContext &ctx) const;

    template <typename T>
    uint32_t AxisCal(T axis, const std::vector<int64_t> &data_dims, int64_t &head_dim, int64_t &end_dim) const;

    template <typename T, typename T2>
    uint32_t ReduceAllOneAxes(
        const T *input_data, std::vector<int64_t> &input_dims, T *output_data, const int64_t &output_num,
        std::vector<T2> &axes);

    template <typename T>
    std::vector<int64_t> GetOutputShape(const std::vector<int64_t> &input_shape, const T &axis);

    template <typename T>
    uint32_t AxesRankCheckAndReverse(
        const CpuKernelContext &ctx, const T *axis_data, const int64_t &axes_num, std::map<T, int64_t> &axis_map,
        int32_t &rank);

    uint32_t ReduceAllCheck(const CpuKernelContext &ctx) const;

    bool keep_dims_ = false;
    size_t axes_idx_ = 0;
};
}  // namespace aicpu

#endif