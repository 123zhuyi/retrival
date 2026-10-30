/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "top_k_pq_distance_v2_aicpu.h"

#include <algorithm>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char* const kTopKPQDistanceV2 = "TopKPQDistanceV2";
constexpr uint32_t kPqDistanceInputIndex = 0;
constexpr uint32_t kGroupedExtremeDistanceInputIndex = 1;
constexpr uint32_t kTopkDistanceOutputIndex = 0;
constexpr uint32_t kTopkIndexOutputIndex = 1;
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 2;
constexpr int32_t kBatchDimIndex = 0;
constexpr int32_t kElementDimIndex = 1;
constexpr int32_t kPqDistanceDimNum = 2;
constexpr const char* kOrderAsc = "ASC";
constexpr const char* kOrderDes = "DES";
} // namespace

namespace aicpu {
uint32_t TopKPQDistanceV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                        kTopKPQDistanceV2);
    Tensor* pq_distance = ctx.Input(kPqDistanceInputIndex);
    KERNEL_CHECK_NULLPTR(pq_distance, KERNEL_STATUS_PARAM_INVALID, "Get input pq_distance failed.")
    DataType data_type = pq_distance->GetDataType();
    uint32_t res = KERNEL_STATUS_OK;
    switch (data_type) {
        case DT_FLOAT16:
            res = DoCompute<Eigen::half>(ctx);
            break;
        case DT_FLOAT:
            res = DoCompute<float>(ctx);
            break;
        case DT_INT32:
            res = DoCompute<int32_t>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR(
                "[%s] input pq_distance only support type[DT_FLOAT16, DT_FLOAT, DT_INT32], but got type[%s].",
                kTopKPQDistanceV2, DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] kernel compute failed, KernelStatus is [%u].", kTopKPQDistanceV2, res);
    }
    return res;
}

uint32_t TopKPQDistanceV2CpuKernel::CheckInputAndAttr(const CpuKernelContext& ctx)
{
    Tensor* pq_distance_tensor = ctx.Input(kPqDistanceInputIndex);
    auto pq_distance_tensor_shape = pq_distance_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(pq_distance_tensor_shape, KERNEL_STATUS_PARAM_INVALID, "Get input pq_distance shape failed.")
    KERNEL_CHECK_FALSE((pq_distance_tensor_shape->GetDims() == kPqDistanceDimNum), KERNEL_STATUS_PARAM_INVALID,
                       "Input pq_distance should be 2D, but got dim num[%d].", pq_distance_tensor_shape->GetDims())
    int64_t element_num = pq_distance_tensor_shape->GetDimSize(kElementDimIndex);

    KERNEL_CHECK_FALSE((ParseAndCheckAttr(ctx, element_num) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "Parse and check attr failed.")

    auto grouped_extreme_shape = ctx.Input(kGroupedExtremeDistanceInputIndex)->GetTensorShape();
    KERNEL_CHECK_NULLPTR(grouped_extreme_shape, KERNEL_STATUS_PARAM_INVALID,
                         "Get input grouped_extreme_distance shape failed.")
    KERNEL_CHECK_FALSE((grouped_extreme_shape->GetDims() == kPqDistanceDimNum), KERNEL_STATUS_PARAM_INVALID,
                       "Input grouped_extreme_distance should be 2D, but got dim num[%d].",
                       grouped_extreme_shape->GetDims())

    int64_t batch = pq_distance_tensor_shape->GetDimSize(kBatchDimIndex);
    KERNEL_CHECK_FALSE((CheckBufferElementNum(ctx, batch, element_num) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "Check buffer element num failed.")

    KERNEL_LOG_DEBUG("[%s] order[%s], is_min_heap_ is[%d], element_num is[%ld], group_size is[%d]", kTopKPQDistanceV2,
                     order_.c_str(), is_min_heap_, element_num, group_size_);
    return KERNEL_STATUS_OK;
}

uint32_t TopKPQDistanceV2CpuKernel::ParseAndCheckAttr(const CpuKernelContext& ctx, int64_t element_num)
{
    AttrValue* order = ctx.GetAttr("order");
    KERNEL_CHECK_NULLPTR(order, KERNEL_STATUS_PARAM_INVALID, "Get attr [order] failed.")
    order_ = order->GetString();
    KERNEL_CHECK_FALSE(((order_ == kOrderAsc) || (order_ == kOrderDes)), KERNEL_STATUS_PARAM_INVALID,
                       "order should be ASC or DES, but got [%s].", order_.c_str())
    // ASC keeps the k smallest values, so the heap root must hold the largest candidate to evict.
    is_min_heap_ = (order_ == kOrderAsc) ? false : true;

    AttrValue* k = ctx.GetAttr("k");
    KERNEL_CHECK_NULLPTR(k, KERNEL_STATUS_PARAM_INVALID, "Get attr [k] failed.")
    k_ = static_cast<int32_t>(k->GetInt());
    KERNEL_CHECK_FALSE((k_ > 0 && k_ <= element_num), KERNEL_STATUS_PARAM_INVALID,
                       "k should be in range (0, element_num], but got k[%d], element_num[%ld].", k_, element_num)

    AttrValue* group_size = ctx.GetAttr("group_size");
    KERNEL_CHECK_NULLPTR(group_size, KERNEL_STATUS_PARAM_INVALID, "Get attr [group_size] failed.")
    group_size_ = static_cast<int32_t>(group_size->GetInt());
    // group_size divides element_num / k below; keep the `== 0` check standalone so the div-by-zero static check
    // recognizes it as the guard, and reject negatives with a separate `< 0` check (a merged `<= 0` would no longer
    // match the recognized guard form).
    if (group_size_ == 0) {
        KERNEL_LOG_ERROR("[%s] group_size must not be 0.", kTopKPQDistanceV2);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (group_size_ < 0) {
        KERNEL_LOG_ERROR("[%s] group_size should be greater than zero, but got group_size[%d].", kTopKPQDistanceV2,
                         group_size_);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if ((element_num % static_cast<int64_t>(group_size_)) != 0) {
        KERNEL_LOG_ERROR(
            "[%s] element num should be an integer multiple of group size, but element_num is [%ld], group_size is "
            "[%d].",
            kTopKPQDistanceV2, element_num, group_size_);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    extreme_size_ = static_cast<int32_t>(element_num / static_cast<int64_t>(group_size_));
    init_group_offset_ = k_ % group_size_;
    return KERNEL_STATUS_OK;
}

uint32_t TopKPQDistanceV2CpuKernel::CheckBufferElementNum(const CpuKernelContext& ctx, int64_t batch,
                                                          int64_t element_num) const
{
    // Every shard walks pq_distance / grouped_extreme_distance and both outputs by batch, so the buffers must be
    // large enough for the whole batch before any parallel work starts.
    int64_t extreme_num_per_batch = static_cast<int64_t>(extreme_size_);
    KERNEL_CHECK_FALSE((ctx.Input(kPqDistanceInputIndex)->NumElements() >= batch * element_num),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Input pq_distance element num[%ld] is less than batch[%ld] * element_num[%ld].",
                       ctx.Input(kPqDistanceInputIndex)->NumElements(), batch, element_num)
    KERNEL_CHECK_FALSE((ctx.Input(kGroupedExtremeDistanceInputIndex)->NumElements() >= batch * extreme_num_per_batch),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Input grouped_extreme_distance element num[%ld] is less than batch[%ld] * [%ld].",
                       ctx.Input(kGroupedExtremeDistanceInputIndex)->NumElements(), batch, extreme_num_per_batch)
    KERNEL_CHECK_FALSE((ctx.Output(kTopkDistanceOutputIndex)->NumElements() >= batch * static_cast<int64_t>(k_)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Output topk_distance element num[%ld] is less than batch[%ld] * k[%d].",
                       ctx.Output(kTopkDistanceOutputIndex)->NumElements(), batch, k_)
    KERNEL_CHECK_FALSE((ctx.Output(kTopkIndexOutputIndex)->NumElements() >= batch * static_cast<int64_t>(k_)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Output topk_index element num[%ld] is less than batch[%ld] * k[%d].",
                       ctx.Output(kTopkIndexOutputIndex)->NumElements(), batch, k_)
    return KERNEL_STATUS_OK;
}

uint32_t TopKPQDistanceV2CpuKernel::CheckDataType(const CpuKernelContext& ctx) const
{
    // pq_distance, grouped_extreme_distance and topk_distance are all read/written through T.
    DataType pq_distance_type = ctx.Input(kPqDistanceInputIndex)->GetDataType();
    DataType grouped_extreme_type = ctx.Input(kGroupedExtremeDistanceInputIndex)->GetDataType();
    KERNEL_CHECK_FALSE((grouped_extreme_type == pq_distance_type), KERNEL_STATUS_PARAM_INVALID,
                       "Input grouped_extreme_distance type[%s] must be the same as pq_distance type[%s].",
                       DTypeStr(grouped_extreme_type).c_str(), DTypeStr(pq_distance_type).c_str())
    DataType topk_distance_type = ctx.Output(kTopkDistanceOutputIndex)->GetDataType();
    KERNEL_CHECK_FALSE((topk_distance_type == pq_distance_type), KERNEL_STATUS_PARAM_INVALID,
                       "Output topk_distance type[%s] must be the same as pq_distance type[%s].",
                       DTypeStr(topk_distance_type).c_str(), DTypeStr(pq_distance_type).c_str())
    DataType topk_index_type = ctx.Output(kTopkIndexOutputIndex)->GetDataType();
    KERNEL_CHECK_FALSE((topk_index_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Output topk_index only support type[DT_INT32], but got type[%s].",
                       DTypeStr(topk_index_type).c_str())

    KERNEL_CHECK_NULLPTR(ctx.Input(kPqDistanceInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input pq_distance data failed.")
    KERNEL_CHECK_NULLPTR(ctx.Input(kGroupedExtremeDistanceInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input grouped_extreme_distance data failed.")
    KERNEL_CHECK_NULLPTR(ctx.Output(kTopkDistanceOutputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get output topk_distance data failed.")
    KERNEL_CHECK_NULLPTR(ctx.Output(kTopkIndexOutputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get output topk_index data failed.")
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKPQDistanceV2CpuKernel::DoCompute(const CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(CheckInputAndAttr(ctx), "[%s] check input and attr failed.", kTopKPQDistanceV2);
    KERNEL_HANDLE_ERROR(CheckDataType(ctx), "[%s] check data type failed.", kTopKPQDistanceV2);

    auto pq_distance_tensor_shape = ctx.Input(kPqDistanceInputIndex)->GetTensorShape();
    int64_t batch = pq_distance_tensor_shape->GetDimSize(kBatchDimIndex);
    if (batch == 0) {
        KERNEL_LOG_DEBUG("[%s] batch is zero, nothing to compute.", kTopKPQDistanceV2);
        return KERNEL_STATUS_OK;
    }

    // Every parameter the shard needs was validated above, so the shard body itself cannot fail.
    auto sharder = [&](int64_t start, int64_t end) { ComputeWithBlock<T>(ctx, start, end); };
    int64_t max_core_num = std::min(batch, static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)));
    if (max_core_num <= 0) {
        max_core_num = 1;
    }
    int64_t per_unit_size = CeilMultiple(batch, max_core_num);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, per_unit_size, sharder),
                        "[%s] parallel compute failed.", kTopKPQDistanceV2);
    return KERNEL_STATUS_OK;
}

template <typename T>
void TopKPQDistanceV2CpuKernel::ComputeWithBlock(const CpuKernelContext& ctx, int64_t start, int64_t end) const
{
    Tensor* pq_distance_tensor = ctx.Input(kPqDistanceInputIndex);
    int64_t element_num = pq_distance_tensor->GetTensorShape()->GetDimSize(kElementDimIndex);
    int64_t extreme_num_per_batch = static_cast<int64_t>(extreme_size_);

    const T* pq_distance_addr = static_cast<const T*>(pq_distance_tensor->GetData()) + start * element_num;
    const T* grouped_extreme_addr = static_cast<const T*>(ctx.Input(kGroupedExtremeDistanceInputIndex)->GetData()) +
                                    start * extreme_num_per_batch;
    T* output_topk_distance_addr = static_cast<T*>(ctx.Output(kTopkDistanceOutputIndex)->GetData()) +
                                   start * static_cast<int64_t>(k_);
    int32_t* output_topk_index_addr = static_cast<int32_t*>(ctx.Output(kTopkIndexOutputIndex)->GetData()) +
                                      start * static_cast<int64_t>(k_);

    // Both heaps are fully rewritten for every batch row, so one allocation per shard is enough.
    std::vector<Item<T>> grp_extreme(static_cast<size_t>(k_));
    std::vector<Item<T>> topk(static_cast<size_t>(k_));

    for (int64_t i = start; i < end; i++) {
        InputsData<T> input_data;
        input_data.pq_distances = pq_distance_addr;
        input_data.grouped_extreme_distances = grouped_extreme_addr;

        GetGroupedDistanceTopKHeap(grp_extreme.data(), input_data);
        GetDistanceTopKHeap(topk.data(), grp_extreme.data(), input_data);
        ProcessResultWithBlock(topk.data(), output_topk_distance_addr, output_topk_index_addr);

        pq_distance_addr += element_num;
        grouped_extreme_addr += extreme_num_per_batch;
        output_topk_distance_addr += k_;
        output_topk_index_addr += k_;
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::ProcessResultWithBlock(Item<T> topk_ptr[], T* topk_distance_ptr,
                                                       int32_t* topk_index_ptr) const
{
    // Popping the heap yields the extremes in reverse, so fill the output back to front.
    for (int32_t i = k_; i > 0; i--) {
        Item<T> res;
        PopHeap<T>(topk_ptr, i, &res);
        topk_distance_ptr[i - 1] = res.val;
        topk_index_ptr[i - 1] = res.idx;
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::InitTopKHeap(int32_t& group_idx, Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                                             const InputsData<T>& inputs_data) const
{
    const T* ptr = inputs_data.pq_distances;
    int32_t cnt = 0;
    for (; group_idx < k_; group_idx++) {
        int32_t idx = grp_extreme_ptr[group_idx].idx * group_size_;
        const T* itemvalptr = ptr + idx;
        for (int32_t index = 0; index < group_size_; index++, cnt++) {
            if (cnt == k_) {
                return;
            }
            topk_ptr[cnt] = {itemvalptr[index], idx + index};
        }
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::GetDistanceTopKHeap(Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                                                    const InputsData<T>& inputs_data) const
{
    int32_t group_idx = 0;
    const T* ptr = inputs_data.pq_distances;
    InitTopKHeap(group_idx, topk_ptr, grp_extreme_ptr, inputs_data);
    MakeHeap(topk_ptr, k_);
    // Offset inside the group that InitTopKHeap stopped at.
    int32_t index = init_group_offset_;
    int32_t size = std::min(k_, extreme_size_);
    for (; group_idx < size; group_idx++, index = 0) {
        if (is_min_heap_) {
            if (grp_extreme_ptr[group_idx].val <= topk_ptr[0].val) {
                continue;
            }
        } else if (grp_extreme_ptr[group_idx].val >= topk_ptr[0].val) {
            continue;
        }
        int32_t idx = grp_extreme_ptr[group_idx].idx * group_size_;
        const T* itemvalptr = ptr + idx;
        for (; index < group_size_; index++) {
            const T& itemval = itemvalptr[index];
            if (is_min_heap_) {
                if (itemval <= topk_ptr[0].val) {
                    continue;
                }
            } else if (itemval >= topk_ptr[0].val) {
                continue;
            }
            topk_ptr[0] = {itemval, idx + index};
            HeapFixdown(topk_ptr, 0, k_);
        }
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::GetGroupedDistanceTopKHeap(Item<T> grp_extreme_ptr[],
                                                           const InputsData<T>& input_data) const
{
    const T* ptr = input_data.grouped_extreme_distances;
    int32_t size = std::min(k_, extreme_size_);
    // idx points at the position inside grouped_extreme_distances.
    int32_t idx = 0;
    InitGrpExtreme<T>(grp_extreme_ptr, input_data, size, idx);
    MakeHeap(grp_extreme_ptr, size);
    for (; idx < extreme_size_; idx++) {
        T temp = ptr[idx];
        if (is_min_heap_) {
            if (grp_extreme_ptr[0].val > temp) {
                continue;
            }
        } else if (grp_extreme_ptr[0].val < temp) {
            continue;
        }
        grp_extreme_ptr[0] = {temp, idx};
        HeapFixdown(grp_extreme_ptr, 0, size);
    }
    SortHeap(grp_extreme_ptr, size);
}

template <typename T>
void TopKPQDistanceV2CpuKernel::InitGrpExtreme(Item<T> grp_extreme_ptr[], const InputsData<T>& input_data, int32_t size,
                                               int32_t& idx) const
{
    int32_t n = 0;
    const T* ptr = input_data.grouped_extreme_distances;
    for (; idx < size; idx++, n++) {
        grp_extreme_ptr[n] = {ptr[idx], idx};
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::MakeHeap(Item<T> arr_ptr[], const int32_t n) const
{
    for (int32_t i = static_cast<int32_t>(static_cast<uint32_t>(n) >> 1) - 1; i >= 0; i--) {
        HeapFixdown(arr_ptr, i, n);
    }
}

template <typename T>
void TopKPQDistanceV2CpuKernel::PopHeap(Item<T> arr_ptr[], const int32_t n, Item<T>* const res) const
{
    *res = arr_ptr[0];
    arr_ptr[0] = arr_ptr[n - 1];
    HeapFixdown(arr_ptr, 0, n - 1);
}

template <typename T>
void TopKPQDistanceV2CpuKernel::HeapFixdown(Item<T> a[], const int32_t index, const int32_t n) const
{
    int32_t i = index;
    Item<T> temp = a[i];

    int32_t j = (static_cast<int32_t>(static_cast<uint32_t>(i) << 1)) + 1;
    while (j < n) {
        if (is_min_heap_) {
            if (j + 1 < n && a[j].val > a[j + 1].val) {
                j++;
            }
            if (a[j].val >= temp.val) {
                break;
            }
        } else {
            if (j + 1 < n && a[j].val < a[j + 1].val) {
                j++;
            }
            if (a[j].val <= temp.val) {
                break;
            }
        }

        a[i] = a[j];
        i = j;
        j = (static_cast<int32_t>(static_cast<uint32_t>(i) << 1)) + 1;
    }
    a[i] = temp;
}

template <typename T>
void TopKPQDistanceV2CpuKernel::SortHeap(Item<T> arr_ptr[], const int32_t n) const
{
    for (int32_t i = n - 1; i >= 0; i--) {
        Item<T> temp = arr_ptr[0];
        arr_ptr[0] = arr_ptr[i];
        arr_ptr[i] = temp;
        HeapFixdown(arr_ptr, 0, i);
    }
}

REGISTER_CPU_KERNEL(kTopKPQDistanceV2, TopKPQDistanceV2CpuKernel);
} // namespace aicpu
