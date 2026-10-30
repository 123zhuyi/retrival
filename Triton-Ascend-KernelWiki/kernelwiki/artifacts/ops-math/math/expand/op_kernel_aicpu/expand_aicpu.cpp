/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expand_aicpu.h"

#include <map>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
const char* const kExpand = "Expand";

#define EXPAND_EMPTY_TENSOR_CASE(DTYPE, TYPE, CTX) \
    case (DTYPE): {                                \
        EmptyTensorCompute<TYPE>(CTX);             \
        break;                                     \
    }
} // namespace

namespace expand {
template <typename IndexT>
uint32_t NormalizeExpandShape(std::vector<IndexT>& input_shape, std::vector<IndexT>& target_shape)
{
    if (target_shape.size() < input_shape.size()) {
        KERNEL_LOG_ERROR(
            "Param error, target rank [%zu] cannot be less than input rank [%zu].",
            target_shape.size(), input_shape.size());
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }

    const size_t diff = target_shape.size() - input_shape.size();
    if (diff > 0) {
        input_shape.insert(input_shape.begin(), diff, static_cast<IndexT>(1));
    }

    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] < static_cast<IndexT>(-1)) {
            KERNEL_LOG_ERROR(
                "Param error, target_shape[%zu] [%ld] is invalid.", i, static_cast<int64_t>(target_shape[i]));
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }

        if (i < diff) {
            if (aicpu::IsValueEqual<IndexT>(target_shape[i], static_cast<IndexT>(-1))) {
                target_shape[i] = static_cast<IndexT>(1);
            }
            continue;
        }

        if (aicpu::IsValueEqual<IndexT>(target_shape[i], static_cast<IndexT>(-1))) {
            target_shape[i] = input_shape[i];
            continue;
        }

        if (aicpu::IsValueEqual<IndexT>(target_shape[i], static_cast<IndexT>(1)) &&
            !aicpu::IsValueEqual<IndexT>(input_shape[i], static_cast<IndexT>(1))) {
            target_shape[i] = input_shape[i];
            continue;
        }

        if (!aicpu::IsValueEqual<IndexT>(input_shape[i], static_cast<IndexT>(1)) &&
            !aicpu::IsValueEqual<IndexT>(input_shape[i], target_shape[i])) {
            KERNEL_LOG_ERROR(
                "Param error, input_shape[%zu] [%ld] cannot broadcast to target_shape[%zu] [%ld].", i,
                static_cast<int64_t>(input_shape[i]), i, static_cast<int64_t>(target_shape[i]));
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
    }

    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t CalculateOutIndex(
    const std::vector<T>& input_values, std::vector<T>& output_values, const std::vector<IndexT>& input_shape,
    uint64_t copy_size, uint64_t expand_factor)
{
    uint64_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= static_cast<uint64_t>(dim);
    }

    uint64_t copy_num = input_size / copy_size;
    for (uint64_t i = 0; i < copy_num; ++i) {
        for (uint64_t j = 0; j < expand_factor; ++j) {
            output_values.insert(
                output_values.end(), input_values.begin() + (i * copy_size),
                input_values.begin() + ((i + 1) * copy_size));
        }
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t CopyExpandIndex(
    const std::vector<T>& input_values, std::vector<T>& output_values, const std::vector<IndexT>& input_shape,
    const std::vector<IndexT>& target_shape)
{
    output_values.clear();
    uint64_t expand_factor = 1;
    uint64_t copy_size = 1;
    uint64_t break_axis = 0;

    for (int64_t i = static_cast<int64_t>(input_shape.size()) - 1; i >= 0; --i) {
        if (!aicpu::IsValueEqual<IndexT>(input_shape[static_cast<size_t>(i)], target_shape[static_cast<size_t>(i)])) {
            if (!aicpu::IsValueEqual<IndexT>(input_shape[static_cast<size_t>(i)], static_cast<IndexT>(1))) {
                KERNEL_LOG_ERROR(
                    "Param error, input_shape[%ld] != 1 when input_shape[%ld] != target_shape[%ld].", i, i, i);
                return aicpu::KERNEL_STATUS_PARAM_INVALID;
            }
            if (target_shape[static_cast<size_t>(i)] < 0) {
                KERNEL_LOG_ERROR(
                    "Param error, target_shape[%ld] is invalid at axis[%ld].",
                    static_cast<int64_t>(target_shape[static_cast<size_t>(i)]), i);
                return aicpu::KERNEL_STATUS_PARAM_INVALID;
            }
            expand_factor = static_cast<uint64_t>(target_shape[static_cast<size_t>(i)]);
            break_axis = static_cast<uint64_t>(i);
            break;
        }
    }

    if (!input_shape.empty()) {
        if (break_axis == 0) {
            for (uint64_t i = input_shape.size() - 1; i > 0; --i) {
                copy_size *= static_cast<uint64_t>(input_shape[i]);
            }
        } else {
            for (uint64_t i = input_shape.size() - 1; i >= break_axis; --i) {
                copy_size *= static_cast<uint64_t>(input_shape[i]);
                if (i == break_axis) {
                    break;
                }
            }
        }
    }

    return CalculateOutIndex<T, IndexT>(input_values, output_values, input_shape, copy_size, expand_factor);
}

template <typename T, typename IndexT>
uint32_t GetExpandIndex(
    const std::vector<T>& input_values, std::vector<T>& output_values, const std::vector<IndexT>& input_shape,
    const std::vector<IndexT>& target_shape, std::vector<IndexT>& expanded_shape)
{
    IndexT expand_factor = static_cast<IndexT>(1);
    uint64_t break_axis = 0;

    for (int64_t i = static_cast<int64_t>(input_shape.size()) - 1; i >= 0; --i) {
        if (!aicpu::IsValueEqual<IndexT>(input_shape[static_cast<size_t>(i)], target_shape[static_cast<size_t>(i)])) {
            if (!aicpu::IsValueEqual<IndexT>(input_shape[static_cast<size_t>(i)], static_cast<IndexT>(1))) {
                KERNEL_LOG_ERROR(
                    "Param error, input_shape[%ld] != 1 when input_shape[%ld] != target_shape[%ld].", i, i, i);
                return aicpu::KERNEL_STATUS_PARAM_INVALID;
            }
            expand_factor = target_shape[static_cast<size_t>(i)];
            break_axis = static_cast<uint64_t>(i);
            break;
        }
    }

    std::vector<IndexT> temp_shape = input_shape;
    temp_shape[break_axis] = expand_factor;
    if (CopyExpandIndex<T, IndexT>(input_values, output_values, input_shape, temp_shape) != aicpu::KERNEL_STATUS_OK) {
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    expanded_shape = std::move(temp_shape);
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t ExpandByLayer(
    std::vector<T> input_values, std::vector<T>& output_values, std::vector<IndexT> input_shape,
    const std::vector<IndexT>& target_shape)
{
    std::vector<IndexT> expanded_shape;
    bool need_continue = true;
    while (need_continue) {
        if (GetExpandIndex<T, IndexT>(input_values, output_values, input_shape, target_shape, expanded_shape) !=
            aicpu::KERNEL_STATUS_OK) {
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
        need_continue = false;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (!aicpu::IsValueEqual<IndexT>(target_shape[i], expanded_shape[i])) {
                need_continue = true;
                break;
            }
        }
        if (need_continue) {
            input_values = output_values;
            input_shape = expanded_shape;
        }
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t PrepareExpandShape(
    std::vector<T>& input_values, std::vector<IndexT>& input_shape, std::vector<IndexT>& target_shape,
    const std::vector<int64_t>& origin_shape, const T* input_data)
{
    for (auto dim : origin_shape) {
        input_shape.push_back(static_cast<IndexT>(dim));
    }

    uint64_t input_num = 1;
    for (auto dim : input_shape) {
        input_num *= static_cast<uint64_t>(dim);
    }

    for (uint64_t i = 0; i < input_num; ++i) {
        input_values.push_back(input_data[i]);
    }

    return NormalizeExpandShape(input_shape, target_shape);
}

template <typename T, typename IndexT>
uint32_t DoExpandCompute(const aicpu::CpuKernelContext& ctx)
{
    const auto* input_data = static_cast<const T*>(ctx.Input(0)->GetData());
    const auto* shape_data = static_cast<const IndexT*>(ctx.Input(1)->GetData());
    auto* output_data = static_cast<T*>(ctx.Output(0)->GetData());

    const auto* input_tensor = ctx.Input(0);
    const auto* shape_tensor = ctx.Input(1);

    std::vector<int64_t> origin_shape = input_tensor->GetTensorShape()->GetDimSizes();
    const std::vector<int64_t> shape_shape = shape_tensor->GetTensorShape()->GetDimSizes();
    if (origin_shape.empty()) {
        origin_shape.push_back(static_cast<int64_t>(1));
    }

    std::vector<T> output_values;
    std::vector<T> input_values;
    std::vector<IndexT> input_shape;
    std::vector<IndexT> target_shape;
    for (int64_t i = 0; i < shape_shape[0]; ++i) {
        target_shape.push_back(shape_data[i]);
    }

    uint32_t ret = PrepareExpandShape<T, IndexT>(input_values, input_shape, target_shape, origin_shape, input_data);
    if (ret != aicpu::KERNEL_STATUS_OK) {
        return ret;
    }

    bool need_expand = false;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (!aicpu::IsValueEqual<IndexT>(input_shape[i], target_shape[i])) {
            need_expand = true;
            break;
        }
    }

    if (!need_expand) {
        for (size_t i = 0; i < input_values.size(); ++i) {
            output_data[i] = input_values[i];
        }
        return aicpu::KERNEL_STATUS_OK;
    }

    ret = ExpandByLayer<T, IndexT>(input_values, output_values, input_shape, target_shape);
    if (ret != aicpu::KERNEL_STATUS_OK) {
        return ret;
    }

    for (size_t i = 0; i < output_values.size(); ++i) {
        output_data[i] = output_values[i];
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename IndexT>
uint32_t IndicesExpandCompute(aicpu::CpuKernelContext& ctx)
{
    auto input_type = static_cast<aicpu::DataType>(ctx.Input(0)->GetDataType());
    std::map<int, std::function<uint32_t(aicpu::CpuKernelContext&)>> calls;
    calls[aicpu::DT_FLOAT16] = DoExpandCompute<Eigen::half, IndexT>;
    calls[aicpu::DT_BFLOAT16] = DoExpandCompute<Eigen::bfloat16, IndexT>;
    calls[aicpu::DT_FLOAT] = DoExpandCompute<float, IndexT>;
    calls[aicpu::DT_INT8] = DoExpandCompute<int8_t, IndexT>;
    calls[aicpu::DT_INT32] = DoExpandCompute<int32_t, IndexT>;
    calls[aicpu::DT_INT64] = DoExpandCompute<int64_t, IndexT>;
    calls[aicpu::DT_UINT8] = DoExpandCompute<uint8_t, IndexT>;
    calls[aicpu::DT_BOOL] = DoExpandCompute<bool, IndexT>;

    auto found = calls.find(input_type);
    if (found == calls.end()) {
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    return found->second(ctx);
}
} // namespace expand

namespace aicpu {
template <typename T>
void ExpandCpuKernel::EmptyTensorCompute(const CpuKernelContext& ctx)
{
    const int64_t shape_num = ctx.Input(kSecondInputIndex)->NumElements();
    KERNEL_LOG_INFO("shape num elements [%ld]", shape_num);
    if (shape_num == 0) {
        auto* output_data = reinterpret_cast<T*>(ctx.Output(kFirstOutputIndex)->GetData());
        const auto* input_data = reinterpret_cast<const T*>(ctx.Input(kFirstInputIndex)->GetData());
        *output_data = *input_data;
        is_empty_tensor_ = true;
    }
}

void ExpandCpuKernel::HandleEmptyTensor(const CpuKernelContext& ctx)
{
    auto input_type = ctx.Input(kFirstInputIndex)->GetDataType();
    switch (input_type) {
        EXPAND_EMPTY_TENSOR_CASE(DT_FLOAT16, Eigen::half, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_BFLOAT16, Eigen::bfloat16, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_FLOAT, float, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT32, int32_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT64, int64_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT8, int8_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_UINT8, uint8_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_BOOL, bool, ctx)
        default:
            KERNEL_LOG_WARN("Expand empty tensor data type [%u] not support.", input_type);
    }
}

uint32_t ExpandCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("ExpandCpuKernel start.");
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check Expand params failed.");

    is_empty_tensor_ = false;
    HandleEmptyTensor(ctx);
    if (is_empty_tensor_) {
        KERNEL_LOG_INFO("shape of expand empty tensor scenario.");
        return KERNEL_STATUS_OK;
    }

    auto shape_type = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
    switch (shape_type) {
        case DT_INT32:
            return expand::IndicesExpandCompute<int32_t>(ctx);
        case DT_INT64:
            return expand::IndicesExpandCompute<int64_t>(ctx);
        default:
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

REGISTER_CPU_KERNEL(kExpand, ExpandCpuKernel);
} // namespace aicpu