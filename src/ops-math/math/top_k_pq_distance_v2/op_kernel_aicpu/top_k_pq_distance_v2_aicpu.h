/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_TOP_K_PQ_DISTANCE_V2_H_
#define AICPU_KERNELS_NORMALIZED_TOP_K_PQ_DISTANCE_V2_H_

#include <cstdint>
#include <string>

#include "cpu_kernel.h"

namespace aicpu {
class TopKPQDistanceV2CpuKernel : public CpuKernel {
public:
    TopKPQDistanceV2CpuKernel() = default;
    ~TopKPQDistanceV2CpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    struct Item {
        T val;
        int32_t idx;
    };

    template <typename T>
    struct InputsData {
        const T* pq_distances = nullptr;
        const T* grouped_extreme_distances = nullptr;
    };

    std::string order_;
    int32_t k_ = 0;
    int32_t group_size_ = 0;
    // element_num / group_size_ and k_ % group_size_ are loop invariant: computed once next to the group_size_
    // guard so no division by group_size_ is left in the per-row compute path.
    int32_t extreme_size_ = 0;
    int32_t init_group_offset_ = 0;
    bool is_min_heap_ = true;

    uint32_t CheckInputAndAttr(const CpuKernelContext& ctx);

    uint32_t ParseAndCheckAttr(const CpuKernelContext& ctx, int64_t element_num);

    uint32_t CheckBufferElementNum(const CpuKernelContext& ctx, int64_t batch, int64_t element_num) const;

    uint32_t CheckDataType(const CpuKernelContext& ctx) const;

    template <typename T>
    uint32_t DoCompute(const CpuKernelContext& ctx);

    template <typename T>
    void ComputeWithBlock(const CpuKernelContext& ctx, int64_t start, int64_t end) const;

    template <typename T>
    void InitTopKHeap(int32_t& group_idx, Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                      const InputsData<T>& inputs_data) const;

    template <typename T>
    void GetDistanceTopKHeap(Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                             const InputsData<T>& inputs_data) const;

    template <typename T>
    void GetGroupedDistanceTopKHeap(Item<T> grp_extreme_ptr[], const InputsData<T>& input_data) const;

    template <typename T>
    void InitGrpExtreme(Item<T> grp_extreme_ptr[], const InputsData<T>& input_data, int32_t size, int32_t& idx) const;

    template <typename T>
    void MakeHeap(Item<T> arr_ptr[], int32_t n) const;

    template <typename T>
    void PopHeap(Item<T> arr_ptr[], int32_t n, Item<T>* res) const;

    template <typename T>
    void HeapFixdown(Item<T> a[], int32_t index, int32_t n) const;

    template <typename T>
    void SortHeap(Item<T> arr_ptr[], int32_t n) const;

    template <typename T>
    void ProcessResultWithBlock(Item<T> topk_ptr[], T* topk_distance_ptr, int32_t* topk_index_ptr) const;
};
} // namespace aicpu
#endif
