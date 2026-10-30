/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "top_k_pq_distance_aicpu.h"

#include <securec.h>

#include "algorithm"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "log.h"
namespace {
const char* const kTopKPQDistance = "TopKPQDistance";
constexpr uint32_t kParamNum = 5;
const uint32_t kOutputNum = 3;
} // namespace

namespace aicpu {
uint32_t TopKPQDistanceCpuKernel::Compute(CpuKernelContext& ctx)
{
    uint32_t inputs_size = ctx.GetInputsSize();
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, inputs_size, kOutputNum), "Check TopKPQDistance params failed.");
    data_batch_ = inputs_size / kParamNum;
    Tensor* pq_distance = ctx.Input(data_batch_ * 1);

    DataType data_type = pq_distance->GetDataType();
    uint32_t res = KERNEL_STATUS_OK;
    switch (data_type) {
        case DT_FLOAT16:
            KERNEL_LOG_DEBUG("TopKPQDistance compute DT_FLOAT16");
            res = DoCompute<FP16>(ctx);
            break;
        case DT_FLOAT:
            KERNEL_LOG_DEBUG("TopKPQDistance compute DT_FLOAT");
            res = DoCompute<float>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("TopKPQDistance input pq_distance only support type[DT_FLOAT16, "
                             "DT_FLOAT], but got type[%s]",
                             DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("TopKPQDistance kernel compute failed, KernelStatus is [%d]", res);
    }
    return res;
}

template <typename T>
uint32_t TopKPQDistanceCpuKernel::DoCompute(const CpuKernelContext& ctx)
{
    InputsData<T> input_data;
    uint32_t ret = GetInputAndCheck(ctx, input_data);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    Item<T> grp_extreme_ptr[k_];
    Item<T> topk_ptr[k_];
    ret = GetGroupedDistanceTopKHeap(grp_extreme_ptr, input_data);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    ret = GetDistanceTopKHeap(topk_ptr, grp_extreme_ptr, input_data);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    ret = ProcessResult(ctx, input_data, topk_ptr);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKPQDistanceCpuKernel::GetInputAndCheck(const CpuKernelContext& ctx, InputsData<T>& input_data)
{
    auto* order = ctx.GetAttr("order");
    KERNEL_CHECK_NULLPTR(order, KERNEL_STATUS_PARAM_INVALID, "Get attr [order] failed.");
    order_ = order->GetString();
    is_min_heap_ = ("ASC" == order_) ? false : true;
    KERNEL_LOG_DEBUG("TopKPQDistance getInputAndCheck order_[%s], is_min_heap_ is[%d], "
                     "data_batch_ is[%d]",
                     order_.c_str(), is_min_heap_, data_batch_);
    auto* k = ctx.GetAttr("k");
    KERNEL_CHECK_NULLPTR(k, KERNEL_STATUS_PARAM_INVALID, "Get attr [k] failed.");
    k_ = static_cast<int32_t>(k->GetInt());
    KERNEL_CHECK_FALSE((k_ > 0), KERNEL_STATUS_PARAM_INVALID, "k_[%d] should be bigger than zero.", k_);
    auto* group_size = ctx.GetAttr("group_size");
    KERNEL_CHECK_NULLPTR(group_size, KERNEL_STATUS_PARAM_INVALID, "Get attr [group_size] failed.");
    group_size_ = static_cast<int32_t>(group_size->GetInt());
    KERNEL_CHECK_FALSE((group_size_ > 0), KERNEL_STATUS_PARAM_INVALID, "group_size[%d] should be bigger than zero.",
                       group_size_);
    for (uint32_t i = 0; i < data_batch_; i++) {
        Tensor* actual_count_tensor = ctx.Input(data_batch_ * 0 + i);
        KERNEL_CHECK_NULLPTR(actual_count_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID,
                             "actual_count tensor data is null.");
        int32_t actual_count = *(static_cast<int32_t*>(actual_count_tensor->GetData()));
        KERNEL_CHECK_FALSE((actual_count >= 0 && actual_count % group_size_ == 0), KERNEL_STATUS_PARAM_INVALID,
                           "actual_count[%d] should be non-negative and an integer multiple of group_size[%d].",
                           actual_count, group_size_);
        input_data.actual_count += actual_count;

        Tensor* pq_distance_tensor = ctx.Input(data_batch_ * 1 + i);
        input_data.pq_distances.Add(static_cast<T*>(pq_distance_tensor->GetData()), actual_count);

        Tensor* grouped_extreme_distance_tensor = ctx.Input(data_batch_ * 2 + i);
        input_data.grouped_extreme_distances.Add(static_cast<T*>(grouped_extreme_distance_tensor->GetData()),
                                                 actual_count / group_size_);

        Tensor* pq_ivf_tensor = ctx.Input(data_batch_ * 3 + i);
        input_data.pq_ivfs.Add(static_cast<int32_t*>(pq_ivf_tensor->GetData()), actual_count);

        Tensor* pq_index_tensor = ctx.Input(data_batch_ * 4 + i);
        input_data.pq_indexs.Add(static_cast<int32_t*>(pq_index_tensor->GetData()), actual_count);
    }

    KERNEL_CHECK_FALSE((input_data.actual_count >= k_), KERNEL_STATUS_PARAM_INVALID,
                       "k_[%d] should not be greater than actual_count[%d].", k_, input_data.actual_count);
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKPQDistanceCpuKernel::ProcessResult(const CpuKernelContext& ctx, const InputsData<T>& input_data,
                                                Item<T> topk_ptr[])
{
    Tensor* topk_distance = ctx.Output(0);
    Tensor* topk_ivf = ctx.Output(1);
    Tensor* topk_index = ctx.Output(2);

    if (topk_distance->GetDataSize() < static_cast<uint64_t>(sizeof(T) * k_) ||
        topk_ivf->GetDataSize() < static_cast<uint64_t>(sizeof(int32_t) * k_) ||
        topk_index->GetDataSize() < static_cast<uint64_t>(sizeof(int32_t) * k_)) {
        KERNEL_LOG_ERROR("outputs data size error");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    T* topk_distance_ptr = static_cast<T*>(topk_distance->GetData());
    int32_t* topk_ivf_ptr = static_cast<int32_t*>(topk_ivf->GetData());
    int32_t* topk_index_ptr = static_cast<int32_t*>(topk_index->GetData());

    for (int32_t i = k_; i > 0; i--) {
        Item<T> res;
        PopHeap<T>(topk_ptr, i, &res);
        int32_t grp = res.grp;
        int32_t grpi = res.grpi;

        topk_distance_ptr[i - 1] = res.val;
        topk_ivf_ptr[i - 1] = input_data.pq_ivfs.Getv(grp, grpi);
        topk_index_ptr[i - 1] = input_data.pq_indexs.Getv(grp, grpi);
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
void TopKPQDistanceCpuKernel::InitTopKHeap(int& cnt, int& cntk, Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                                           const InputsData<T>& inputs_data)
{
    T** ptr = inputs_data.pq_distances.GetPointer();
    for (; cntk < k_; cntk++) {
        int32_t grp = grp_extreme_ptr[cntk].grp;
        int32_t grpi = grp_extreme_ptr[cntk].grpi * group_size_;
        T* itemvalptr = ptr[grp] + grpi;
        for (int32_t index = 0; index < group_size_; index++, cnt++) {
            if (cnt == k_) {
                return;
            }
            topk_ptr[cnt] = {itemvalptr[index], grp, grpi + index};
        }
    }
}

template <typename T>
uint32_t TopKPQDistanceCpuKernel::GetDistanceTopKHeap(Item<T> topk_ptr[], const Item<T> grp_extreme_ptr[],
                                                      const InputsData<T>& inputs_data)
{
    int cnt = 0;
    int cntk = 0;
    T** ptr = inputs_data.pq_distances.GetPointer();
    InitTopKHeap(cnt, cntk, topk_ptr, grp_extreme_ptr, inputs_data);
    MakeHeap(topk_ptr, k_);
    int32_t extreme_size = inputs_data.actual_count / group_size_;
    int32_t size = std::min(k_, extreme_size);
    int32_t index = k_ % group_size_;
    for (; cntk < size; cntk++, index = 0) {
        if (is_min_heap_) {
            if (grp_extreme_ptr[cntk].val <= topk_ptr[0].val) {
                continue;
            }
        } else if (grp_extreme_ptr[cntk].val >= topk_ptr[0].val) {
            continue;
        }
        int32_t grp = grp_extreme_ptr[cntk].grp;
        int32_t grpi = grp_extreme_ptr[cntk].grpi * group_size_;
        T* itemvalptr = ptr[grp] + grpi;
        for (; index < group_size_; index++) {
            T& itemval = itemvalptr[index];
            if (is_min_heap_) {
                if (itemval <= topk_ptr[0].val) {
                    continue;
                }
            } else if (itemval >= topk_ptr[0].val) {
                continue;
            }
            topk_ptr[0] = {itemval, grp, grpi + index};
            HeapFixdown(topk_ptr, 0, k_);
        }
    }
    KERNEL_LOG_DEBUG("GetDistanceTopKHeap end");
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKPQDistanceCpuKernel::GetGroupedDistanceTopKHeap(Item<T> grp_extreme_ptr[], const InputsData<T>& input_data)
{
    T** ptr = input_data.grouped_extreme_distances.GetPointer();
    int32_t extreme_size = input_data.actual_count / group_size_;
    int32_t size = std::min(k_, extreme_size);
    int32_t grp = 0;
    int32_t grpi = 0;
    InitGrpExtreme<T>(grp_extreme_ptr, input_data, grp, grpi);
    MakeHeap(grp_extreme_ptr, size);
    int32_t grp_size = static_cast<int32_t>(input_data.grouped_extreme_distances.data_count.size());
    for (; grp < grp_size; grpi = 0, grp++) {
        for (; grpi < input_data.grouped_extreme_distances.data_count[grp]; grpi++) {
            T temp = ptr[grp][grpi];
            if (is_min_heap_) {
                if (grp_extreme_ptr[0].val > temp) {
                    continue;
                }
            } else if (grp_extreme_ptr[0].val < temp) {
                continue;
            }
            grp_extreme_ptr[0] = {temp, grp, grpi};
            HeapFixdown(grp_extreme_ptr, 0, size);
        }
    }
    SortHeap(grp_extreme_ptr, size);
    return KERNEL_STATUS_OK;
}

template <typename T>
void TopKPQDistanceCpuKernel::InitGrpExtreme(Item<T> grp_extreme_ptr[], const InputsData<T>& input_data, int32_t& grp,
                                             int32_t& grpi)
{
    int32_t extreme_size = input_data.actual_count / group_size_;
    int32_t size = std::min(k_, extreme_size);
    int32_t n = 0;
    T** ptr = input_data.grouped_extreme_distances.GetPointer();
    int32_t grp_size = static_cast<int32_t>(input_data.grouped_extreme_distances.data_count.size());
    for (; grp < grp_size; grpi = 0, grp++) {
        for (; grpi < input_data.grouped_extreme_distances.data_count[grp]; grpi++, n++) {
            if (n == size) {
                return;
            }
            grp_extreme_ptr[n] = {ptr[grp][grpi], grp, grpi};
        }
    }
}

template <typename T>
void TopKPQDistanceCpuKernel::MakeHeap(Item<T> arr_ptr[], const int32_t n)
{
    for (int32_t i = (static_cast<uint32_t>(n) >> 1) - 1; i >= 0; i--) {
        HeapFixdown(arr_ptr, i, n);
    }
}

template <typename T>
void TopKPQDistanceCpuKernel::PopHeap(Item<T> arr_ptr[], const int32_t n, Item<T>* res)
{
    *res = arr_ptr[0];
    arr_ptr[0] = arr_ptr[n - 1];
    HeapFixdown(arr_ptr, 0, n - 1);
}

template <typename T>
inline void TopKPQDistanceCpuKernel::HeapFixdown(Item<T> a[], const int32_t index, const int32_t n)
{
    int32_t j = 0;
    int32_t i = index;
    Item<T> temp = a[i];

    j = (i << 1) + 1;
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
        j = (static_cast<uint32_t>(i) << 1) + 1;
    }
    a[i] = temp;
}

template <typename T>
void TopKPQDistanceCpuKernel::SortHeap(Item<T> arr_ptr[], const int32_t n)
{
    Item<T> temp;
    for (int i = n - 1; i >= 0; i--) {
        temp = arr_ptr[0];
        arr_ptr[0] = arr_ptr[i];
        arr_ptr[i] = temp;
        HeapFixdown(arr_ptr, 0, i);
    }
}
REGISTER_CPU_KERNEL(kTopKPQDistance, TopKPQDistanceCpuKernel);
} // namespace aicpu
