/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_TOPK_AICPU_H
#define AICPU_KERNELS_NORMALIZED_TOPK_AICPU_H

#include "cpu_kernel.h"
#include "status.h"

namespace aicpu {
class TopkCpuKernel : public CpuKernel {
public:
    ~TopkCpuKernel() = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

    KernelStatus GetInputAndCheck(const CpuKernelContext& ctx);
    template <typename T>
    KernelStatus DoCompute(const CpuKernelContext& ctx);
    template <typename T>
    void TopKForNVector(size_t start, size_t end);
    template <typename T>
    void TopKForNVectorImpl(T* in, T* val, int32_t* indice, int64_t start, int64_t end) const;
    KernelStatus DispatchByDtype(const CpuKernelContext& ctx);
    KernelStatus ParseShapeAndDim(const CpuKernelContext& ctx);

private:
    int32_t k_ = 0;
    bool sorted_ = true;
    bool largest_ = true;
    int32_t dim_ = 0;
    int32_t input_rank_ = 0;
    DataType data_type_ = DT_DOUBLE;
    Tensor* input_tensor_ = nullptr;
    Tensor* output_values_ = nullptr;
    Tensor* output_indices_ = nullptr;
    int32_t head_ = 1;
    int32_t tail_ = 1;
    int32_t n_ = 1;
};
} // namespace aicpu
#endif // AICPU_KERNELS_NORMALIZED_TOPK_AICPU_H
