/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_INPLACE_TOP_K_DISTANCE_H_
#define AICPU_KERNELS_NORMALIZED_INPLACE_TOP_K_DISTANCE_H_

#include <cstdint>
#include <vector>

#include "cpu_kernel.h"

namespace aicpu {
class InplaceTopKDistanceCpuKernel : public CpuKernel {
public:
    InplaceTopKDistanceCpuKernel() = default;
    ~InplaceTopKDistanceCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    class Inputs {
    public:
        Tensor* topk_pq_distance = nullptr;
        Tensor* topk_pq_index = nullptr;
        Tensor* topk_pq_ivf = nullptr;
        Tensor* pq_distance = nullptr;
        Tensor* pq_index = nullptr;
        Tensor* pq_ivf = nullptr;
        AttrValue* order = nullptr;
    };

    template <typename T>
    class Item {
    public:
        T value;
        int32_t index;
        int32_t ivf;
    };

    uint32_t GetInputAndCheck(const CpuKernelContext& ctx, Inputs& inputs) const;

    uint32_t CheckInputDataType(const Inputs& inputs) const;

    uint32_t CheckInputElementNum(const Inputs& inputs) const;

    template <typename T>
    uint32_t DoCompute(const Inputs& inputs) const;

    template <typename T>
    uint32_t ModifyInput(const std::vector<Item<T>>& items_vec, const Inputs& inputs) const;
};
} // namespace aicpu
#endif
