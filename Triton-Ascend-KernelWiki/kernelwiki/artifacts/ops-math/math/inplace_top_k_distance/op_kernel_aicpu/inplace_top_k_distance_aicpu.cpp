/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inplace_top_k_distance_aicpu.h"

#include <algorithm>
#include <string>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char* const kInplaceTopKDistance = "InplaceTopKDistance";
constexpr uint32_t kTopkPqDistanceInputIndex = 0;
constexpr uint32_t kTopkPqIndexInputIndex = 1;
constexpr uint32_t kTopkPqIvfInputIndex = 2;
constexpr uint32_t kPqDistanceInputIndex = 3;
constexpr uint32_t kPqIndexInputIndex = 4;
constexpr uint32_t kPqIvfInputIndex = 5;
constexpr const char* kOrderAsc = "asc";
} // namespace

namespace aicpu {
uint32_t InplaceTopKDistanceCpuKernel::Compute(CpuKernelContext& ctx)
{
    Inputs inputs;
    uint32_t res = GetInputAndCheck(ctx, inputs);
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] GetInputAndCheck failed, KernelStatus is [%u].", kInplaceTopKDistance, res);
        return res;
    }

    DataType data_type = inputs.topk_pq_distance->GetDataType();
    switch (data_type) {
        case DT_FLOAT16:
            res = DoCompute<Eigen::half>(inputs);
            break;
        case DT_FLOAT:
            res = DoCompute<float>(inputs);
            break;
        default:
            KERNEL_LOG_ERROR("[%s] input topk_pq_distance only support type[DT_FLOAT16, DT_FLOAT], but got type[%s].",
                             kInplaceTopKDistance, DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] kernel compute failed, KernelStatus is [%u].", kInplaceTopKDistance, res);
    }
    return res;
}

uint32_t InplaceTopKDistanceCpuKernel::GetInputAndCheck(const CpuKernelContext& ctx, Inputs& inputs) const
{
    inputs.topk_pq_distance = ctx.Input(kTopkPqDistanceInputIndex);
    inputs.topk_pq_index = ctx.Input(kTopkPqIndexInputIndex);
    inputs.topk_pq_ivf = ctx.Input(kTopkPqIvfInputIndex);
    inputs.pq_distance = ctx.Input(kPqDistanceInputIndex);
    inputs.pq_index = ctx.Input(kPqIndexInputIndex);
    inputs.pq_ivf = ctx.Input(kPqIvfInputIndex);

    KERNEL_CHECK_NULLPTR(inputs.topk_pq_distance, KERNEL_STATUS_PARAM_INVALID,
                         "Get input[0], name[topk_pq_distance] failed.")
    KERNEL_CHECK_NULLPTR(inputs.topk_pq_index, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[topk_pq_index] failed.")
    KERNEL_CHECK_NULLPTR(inputs.topk_pq_ivf, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[topk_pq_ivf] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_distance, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[pq_distance] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_index, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[pq_index] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_ivf, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[pq_ivf] failed.")

    KERNEL_CHECK_NULLPTR(inputs.topk_pq_distance->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[0] data, name[topk_pq_distance] failed.")
    KERNEL_CHECK_NULLPTR(inputs.topk_pq_index->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[1] data, name[topk_pq_index] failed.")
    KERNEL_CHECK_NULLPTR(inputs.topk_pq_ivf->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[2] data, name[topk_pq_ivf] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_distance->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[3] data, name[pq_distance] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_index->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[4] data, name[pq_index] failed.")
    KERNEL_CHECK_NULLPTR(inputs.pq_ivf->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "Get input[5] data, name[pq_ivf] failed.")

    KERNEL_CHECK_FALSE((CheckInputDataType(inputs) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "Check input data type failed.")
    KERNEL_CHECK_FALSE((CheckInputElementNum(inputs) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "Check input element num failed.")

    inputs.order = ctx.GetAttr("order");
    KERNEL_CHECK_NULLPTR(inputs.order, KERNEL_STATUS_PARAM_INVALID, "Get attr, name[order] failed.")
    return KERNEL_STATUS_OK;
}

uint32_t InplaceTopKDistanceCpuKernel::CheckInputDataType(const Inputs& inputs) const
{
    // topk_pq_distance and pq_distance are read through the same element type, so their dtypes must agree.
    DataType topk_distance_type = inputs.topk_pq_distance->GetDataType();
    DataType pq_distance_type = inputs.pq_distance->GetDataType();
    KERNEL_CHECK_FALSE((topk_distance_type == pq_distance_type), KERNEL_STATUS_PARAM_INVALID,
                       "Input topk_pq_distance type[%s] must be the same as pq_distance type[%s].",
                       DTypeStr(topk_distance_type).c_str(), DTypeStr(pq_distance_type).c_str())

    const Tensor* const int32_inputs[] = {inputs.topk_pq_index, inputs.topk_pq_ivf, inputs.pq_index, inputs.pq_ivf};
    const char* const int32_input_names[] = {"topk_pq_index", "topk_pq_ivf", "pq_index", "pq_ivf"};
    for (size_t i = 0; i < sizeof(int32_inputs) / sizeof(int32_inputs[0]); i++) {
        DataType index_type = int32_inputs[i]->GetDataType();
        KERNEL_CHECK_FALSE((index_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                           "Input [%s] only support type[DT_INT32], but got type[%s].", int32_input_names[i],
                           DTypeStr(index_type).c_str())
    }
    return KERNEL_STATUS_OK;
}

uint32_t InplaceTopKDistanceCpuKernel::CheckInputElementNum(const Inputs& inputs) const
{
    // topk_pq_distance / topk_pq_index / topk_pq_ivf form one triple per top-k result, so their element counts must
    // match; pq_distance / pq_index likewise pair up one distance with one index.
    int64_t topk_elements_num = inputs.topk_pq_distance->NumElements();
    KERNEL_CHECK_FALSE((inputs.topk_pq_index->NumElements() == topk_elements_num), KERNEL_STATUS_PARAM_INVALID,
                       "Input topk_pq_index element num[%ld] must be equal to topk_pq_distance element num[%ld].",
                       inputs.topk_pq_index->NumElements(), topk_elements_num)
    KERNEL_CHECK_FALSE((inputs.topk_pq_ivf->NumElements() == topk_elements_num), KERNEL_STATUS_PARAM_INVALID,
                       "Input topk_pq_ivf element num[%ld] must be equal to topk_pq_distance element num[%ld].",
                       inputs.topk_pq_ivf->NumElements(), topk_elements_num)
    int64_t elements_num = inputs.pq_distance->NumElements();
    KERNEL_CHECK_FALSE((inputs.pq_index->NumElements() == elements_num), KERNEL_STATUS_PARAM_INVALID,
                       "Input pq_index element num[%ld] must be equal to pq_distance element num[%ld].",
                       inputs.pq_index->NumElements(), elements_num)
    // pq_ivf is a single bucket number shared by every pq_distance element; only its first element is read, so it must
    // be a scalar or a one-element 1D tensor. Extra elements would be silently dropped, so reject them up front.
    KERNEL_CHECK_FALSE((inputs.pq_ivf->NumElements() == 1), KERNEL_STATUS_PARAM_INVALID,
                       "Input pq_ivf must be a scalar or hold exactly one element, but got element num[%ld].",
                       inputs.pq_ivf->NumElements())
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InplaceTopKDistanceCpuKernel::DoCompute(const Inputs& inputs) const
{
    int64_t topk_elements_num = inputs.topk_pq_distance->NumElements();
    int64_t elements_num = inputs.pq_distance->NumElements();

    std::vector<Item<T>> items;
    items.reserve(static_cast<size_t>(topk_elements_num + elements_num));

    const T* topk_value_ptr = static_cast<const T*>(inputs.topk_pq_distance->GetData());
    const int32_t* topk_index_ptr = static_cast<const int32_t*>(inputs.topk_pq_index->GetData());
    const int32_t* topk_ivf_ptr = static_cast<const int32_t*>(inputs.topk_pq_ivf->GetData());
    for (int64_t i = 0; i < topk_elements_num; i++) {
        items.push_back({topk_value_ptr[i], topk_index_ptr[i], topk_ivf_ptr[i]});
    }

    const T* new_value_ptr = static_cast<const T*>(inputs.pq_distance->GetData());
    const int32_t* new_index_ptr = static_cast<const int32_t*>(inputs.pq_index->GetData());
    // pq_ivf is a scalar: every new element shares the same bucket number.
    const int32_t new_ivf = *static_cast<const int32_t*>(inputs.pq_ivf->GetData());
    for (int64_t i = 0; i < elements_num; i++) {
        items.push_back({new_value_ptr[i], new_index_ptr[i], new_ivf});
    }

    std::sort(items.begin(), items.end(), [](const Item<T>& a, const Item<T>& b) { return a.value < b.value; });
    return ModifyInput(items, inputs);
}

template <typename T>
uint32_t InplaceTopKDistanceCpuKernel::ModifyInput(const std::vector<Item<T>>& items_vec, const Inputs& inputs) const
{
    int64_t topk_elements_num = inputs.topk_pq_distance->NumElements();
    T* topk_value_ptr = static_cast<T*>(inputs.topk_pq_distance->GetData());
    int32_t* topk_index_ptr = static_cast<int32_t*>(inputs.topk_pq_index->GetData());
    int32_t* topk_ivf_ptr = static_cast<int32_t*>(inputs.topk_pq_ivf->GetData());

    std::string order = inputs.order->GetString();
    if (order == kOrderAsc) {
        // items_vec is sorted ascending: the first topk_elements_num items are the smallest ones.
        for (int64_t i = 0; i < topk_elements_num; i++) {
            topk_value_ptr[i] = items_vec[i].value;
            topk_index_ptr[i] = items_vec[i].index;
            topk_ivf_ptr[i] = items_vec[i].ivf;
        }
    } else {
        // Take the largest topk_elements_num items and write them back reversed to get descending order.
        size_t start = items_vec.size() - static_cast<size_t>(topk_elements_num);
        int64_t j = topk_elements_num - 1;
        for (size_t i = start; i < items_vec.size(); i++, j--) {
            topk_value_ptr[j] = items_vec[i].value;
            topk_index_ptr[j] = items_vec[i].index;
            topk_ivf_ptr[j] = items_vec[i].ivf;
        }
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kInplaceTopKDistance, InplaceTopKDistanceCpuKernel);
} // namespace aicpu
