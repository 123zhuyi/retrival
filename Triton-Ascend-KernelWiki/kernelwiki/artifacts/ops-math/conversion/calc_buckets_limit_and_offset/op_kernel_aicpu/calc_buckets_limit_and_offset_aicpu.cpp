/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_buckets_limit_and_offset_aicpu.h"

#include <algorithm>
#include <vector>

#include "utils/kernel_util.h"

namespace {
const char* const kCalcBucketsLimitAndOffset = "CalcBucketsLimitAndOffset";
}

namespace aicpu {
uint32_t CalcBucketsLimitAndOffsetCpuKernel::InitParams(const CpuKernelContext& ctx)
{
    KERNEL_CHECK_FALSE((ctx.GetInputsSize() == kInputNum), KERNEL_STATUS_PARAM_INVALID,
                       "%s op need has %u inputs, but got %u inputs", kOpName, kInputNum, ctx.GetInputsSize());
    KERNEL_CHECK_FALSE((ctx.GetOutputsSize() == kOutputNum), KERNEL_STATUS_PARAM_INVALID,
                       "%s op need has %u outputs, but got %u outputs", kOpName, kOutputNum, ctx.GetOutputsSize());
    for (uint32_t i = 0; i < kInputNum; ++i) {
        auto input = ctx.Input(i);
        int64_t num_elements = input->NumElements();
        KERNEL_CHECK_FALSE((num_elements >= 0), KERNEL_STATUS_PARAM_INVALID,
                           "%s op input[%u] elements num should >= 0, but got [%ld]", kOpName, i, num_elements);
        auto input_data = input->GetData();
        KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID, "%s op input[%u] data is nullptr.", kOpName, i);
        input_num_elements_[i] = num_elements;
        datas_[i] = input_data;
    }
    for (uint32_t i = 0; i < kOutputNum; ++i) {
        auto output = ctx.Output(i);
        auto output_data = output->GetData();
        KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID, "%s op output[%u] data is nullptr.", kOpName, i);
        datas_[kInputNum + i] = output_data;
    }
    auto attr = ctx.GetAttr("total_limit");
    KERNEL_CHECK_NULLPTR(attr, KERNEL_STATUS_PARAM_INVALID, "%s op get total_limit attr failed.", kOpName);
    total_limit_ = attr->GetInt();
    KERNEL_CHECK_FALSE((total_limit_ >= 0), KERNEL_STATUS_PARAM_INVALID, "%s op total_limit should >= 0, but got %ld",
                       kOpName, total_limit_);
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CalcBucketsLimitAndOffsetCpuKernel::DoCompute()
{
    int32_t* counts = new (std::nothrow) int32_t[input_num_elements_[0]];
    if (counts == nullptr) {
        KERNEL_LOG_ERROR("%s op alloc counts memory failed.", kOpName);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    int32_t* bucket_list = reinterpret_cast<int32_t*>(datas_[0]);
    int32_t* ivf_counts = reinterpret_cast<int32_t*>(datas_[1]);
    T* ivf_offset = reinterpret_cast<T*>(datas_[2]);
    int32_t* buckets_limit = reinterpret_cast<int32_t*>(datas_[3]);
    T* buckets_offset = reinterpret_cast<T*>(datas_[4]);
    const uint32_t input_num_2 = 2;

    for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
        if ((bucket_list[i] < 0) || (bucket_list[i] >= input_num_elements_[1]) ||
            (bucket_list[i] >= input_num_elements_[input_num_2])) {
            KERNEL_LOG_ERROR("%s op input0[%ld] = %d is out of range input1 num elements [0, %ld) "
                             "or input2 num elements [0, %ld).",
                             kOpName, i, bucket_list[i], input_num_elements_[1], input_num_elements_[2]);
            delete[] counts;
            return KERNEL_STATUS_PARAM_INVALID;
        }
        counts[i] = ivf_counts[bucket_list[i]];
        buckets_limit[i] = counts[i];
        buckets_offset[i] = ivf_offset[bucket_list[i]];
    }
    std::sort(counts, counts + input_num_elements_[0]);
    int64_t rest = total_limit_;
    int64_t limit = 0;
    for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
        limit = rest / (input_num_elements_[0] - i);
        if (counts[i] > limit) {
            break;
        }
        rest -= counts[i];
    }
    for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
        if (static_cast<int64_t>(buckets_limit[i]) > limit) {
            buckets_limit[i] = static_cast<int32_t>(limit);
        }
    }
    delete[] counts;
    return KERNEL_STATUS_OK;
}

uint32_t CalcBucketsLimitAndOffsetCpuKernel::Compute(CpuKernelContext& ctx)
{
    auto ret = InitParams(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    if (ctx.Input(2)->GetDataType() == DT_INT32) {
        return DoCompute<int32_t>();
    }
    return DoCompute<int64_t>();
}

REGISTER_CPU_KERNEL(kCalcBucketsLimitAndOffset, CalcBucketsLimitAndOffsetCpuKernel);
} // namespace aicpu
