/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_sum_aicpu.h"

#include <algorithm>
#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kReduceSumInputNum = 2;
const uint32_t kReduceSumOutputNum = 1;
const char *const kReduceSum = "ReduceSum";
constexpr int64_t kParallelDataNums = 256;
constexpr int64_t kParallelElements = 2 * 1024 * 1024;

#define REDUCESUM_COMPUTE_CASE(DTYPE, TYPE, CTX)                                                                  \
    case (DTYPE): {                                                                                               \
        uint32_t result = ReduceSumCompute<TYPE>(CTX);                                                            \
        if (result != KERNEL_STATUS_OK) {                                                                         \
            KERNEL_LOG_ERROR("ReduceSum kernel compute failed.");                                                \
            return result;                                                                                        \
        }                                                                                                         \
        break;                                                                                                    \
    }

#define REDUCESUM_COMPUTE_CASE_COMPLEX(DTYPE, TYPE, IN_TYPE, CTX)                                                \
    case (DTYPE): {                                                                                               \
        uint32_t result = ReduceSumCompute2<TYPE, IN_TYPE>(CTX);                                                 \
        if (result != KERNEL_STATUS_OK) {                                                                         \
            KERNEL_LOG_ERROR("ReduceSum kernel compute failed.");                                                \
            return result;                                                                                        \
        }                                                                                                         \
        break;                                                                                                    \
    }
}  // namespace

namespace aicpu {
uint32_t ReduceSumCpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_HANDLE_ERROR(
        NormalCheck(ctx, kReduceSumInputNum, kReduceSumOutputNum), "[%s] check input and output failed.", kReduceSum);
    KERNEL_HANDLE_ERROR(ReduceSumCheck(ctx), "[%s] check params failed.", kReduceSum);

    auto input_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
    switch (input_data_type) {
        REDUCESUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        REDUCESUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
        REDUCESUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        REDUCESUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
        REDUCESUM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
        REDUCESUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX64, std::complex<float>, float, ctx)
        REDUCESUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX128, std::complex<double>, double, ctx)
        default:
            KERNEL_LOG_ERROR("ReduceSum kernel data type [%s] not support.", DTypeStr(input_data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ReduceSumCpuKernel::ReduceSumCheck(const CpuKernelContext &ctx) const
{
    auto *x = ctx.Input(kFirstInputIndex);
    auto *axes = ctx.Input(kSecondInputIndex);
    auto *y = ctx.Output(kFirstOutputIndex);

    KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID, "get input failed.");
    KERNEL_CHECK_NULLPTR(x->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input data failed.");
    KERNEL_CHECK_NULLPTR(x->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.");
    KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID, "get output failed.");
    KERNEL_CHECK_NULLPTR(y->GetData(), KERNEL_STATUS_PARAM_INVALID, "get output data failed.");
    if (axes != nullptr && axes->GetData() != nullptr) {
        KERNEL_CHECK_FALSE((axes->GetDataType() == DT_INT32 || axes->GetDataType() == DT_INT64),
            KERNEL_STATUS_PARAM_INVALID, "Data type of axis is not support, axis data type is [%u].",
            axes->GetDataType());
    }
    return KERNEL_STATUS_OK;
}

bool ReduceSumCpuKernel::GetNoopWithEmptyAxes(const CpuKernelContext &ctx) const
{
    bool noop_with_empty_axes = true;
    auto *noop_with_empty_axes_attr = ctx.GetAttr("noop_with_empty_axes");
    if (noop_with_empty_axes_attr != nullptr) {
        noop_with_empty_axes = noop_with_empty_axes_attr->GetBool();
    }
    return noop_with_empty_axes;
}

template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumCompute(const CpuKernelContext &ctx)
{
    auto *x = ctx.Input(kFirstInputIndex);
    auto *axes = ctx.Input(kSecondInputIndex);
    auto *y = ctx.Output(kFirstOutputIndex);

    std::vector<int64_t> input_shape = x->GetTensorShape()->GetDimSizes();
    auto *input_data = reinterpret_cast<T *>(x->GetData());
    auto *output_data = reinterpret_cast<T *>(y->GetData());
    int64_t input_num = x->NumElements();
    const bool noop_with_empty_axes = GetNoopWithEmptyAxes(ctx);
    if (axes == nullptr || axes->GetDataSize() == 0) {
        if (noop_with_empty_axes) {
            for (int64_t i = 0; i < input_num; ++i) {
                output_data[i] = input_data[i];
            }
            return KERNEL_STATUS_OK;
        }
        if (ReduceSumSimpleCompute<T>(ctx, input_data, input_shape, output_data, true)) {
            return KERNEL_STATUS_OK;
        }
        KERNEL_LOG_ERROR("ReduceSum full reduction on empty axes failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ReduceSumSimpleCompute<T>(ctx, input_data, input_shape, output_data)) {
        return KERNEL_STATUS_OK;
    }

    std::vector<int64_t> axes_value;
    if (axes->GetDataType() == DT_INT32) {
        KERNEL_HANDLE_ERROR(ReduceSumDedupAxes<int32_t>(ctx, axes_value), "ReduceSum deduplicate failed.");
    } else {
        KERNEL_HANDLE_ERROR(ReduceSumDedupAxes<int64_t>(ctx, axes_value), "ReduceSum deduplicate failed.");
    }
    if ((axes_value.size() == 1) && ((axes_value[0] + 1) == static_cast<int64_t>(input_shape.size()))) {
        KERNEL_LOG_INFO("Reduce sum last axes compute");
        return ReduceSumLastAxes<T>(ctx, input_data, input_shape, output_data);
    }

    int64_t output_num = y->NumElements();
    uint32_t axes_idx = 0;
    KERNEL_HANDLE_ERROR(ReduceSumOneAxes<T>(input_data, input_shape, output_data, output_num, axes_value, axes_idx),
        "Reduce sum compute failed.");
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumLastAxes(
    const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape, T *output_data)
{
    int64_t rank = input_shape.size() - 1;
    int64_t datanum = 1;
    int64_t i = 0;
    for (i = 0; i < rank; i++) {
        datanum *= input_shape[i];
    }
    int64_t depth = input_shape[i];
    auto shard_reduce = [&](int64_t start, int64_t end) {
        for (int64_t inner_index = start; inner_index < end; inner_index++) {
            auto accumulator = static_cast<T>(0);
            int64_t inout_index = inner_index * depth;
            for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
                int64_t index = inout_index + depth_index;
                accumulator += input_data[index];
            }
            output_data[inner_index] = accumulator;
        }
    };
    if ((datanum < kParallelDataNums) || ((datanum * depth) < kParallelElements)) {
        shard_reduce(0, datanum);
    } else {
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        if (max_core_num > datanum) {
            max_core_num = datanum;
        }
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, datanum, datanum / max_core_num, shard_reduce),
            "ReduceSum Compute failed.");
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
bool ReduceSumCpuKernel::IsReduceSumFullCompute(const CpuKernelContext &ctx, std::vector<int64_t> &input_shape)
{
    auto *axes_data = reinterpret_cast<T *>(ctx.Input(kSecondInputIndex)->GetData());
    if (axes_data == nullptr) {
        return true;
    }
    int64_t rank = input_shape.size();
    int64_t axes_num = ctx.Input(kSecondInputIndex)->NumElements();
    if (rank != axes_num) {
        return false;
    }
    for (int64_t i = 0; i < rank; i++) {
        if (axes_data[i] != i) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool ReduceSumCpuKernel::ReduceSumSimpleCompute(
    const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape, T *output_data,
    bool force_full_compute)
{
    if (input_shape.size() == 0) {
        output_data[0] = input_data[0];
        return true;
    }
    int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements();
    if (data_num <= 0) {
        int64_t out_num = ctx.Output(kFirstOutputIndex)->NumElements();
        KERNEL_LOG_INFO("Reduce sum input is empty tensor, out num is [%ld].", out_num);
        for (int64_t i = 0; i < out_num; i++) {
            output_data[i] = static_cast<T>(0);
        }
        return true;
    }
    bool full_computer = force_full_compute;
    if (!full_computer) {
        if (ctx.Input(kSecondInputIndex)->GetDataType() == DT_INT32) {
            full_computer = IsReduceSumFullCompute<int32_t>(ctx, input_shape);
        } else {
            full_computer = IsReduceSumFullCompute<int64_t>(ctx, input_shape);
        }
    }
    if (full_computer) {
        auto accumulator = static_cast<T>(0);
        for (int64_t i = 0; i < data_num; i++) {
            accumulator += input_data[i];
        }
        output_data[0] = accumulator;
        KERNEL_LOG_INFO("Reduce sum full compute");
        return true;
    }
    return false;
}

template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumOneAxes(const T *input_data, std::vector<int64_t> &input_shape, T *output_data,
    int64_t output_num, std::vector<int64_t> &axes, uint32_t &axes_idx)
{
    if (axes_idx >= axes.size()) {
        for (int64_t i = 0; i < output_num; i++) {
            output_data[i] = input_data[i];
        }
        return KERNEL_STATUS_OK;
    }
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t depth = 1;
    KERNEL_HANDLE_ERROR(ReduceSumParseAxes(input_shape, axes, axes_idx, inner, outer, depth), "parse axes failed.");
    auto *output_data_temp = new (std::nothrow) T[inner * outer];
    KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
            auto accumulator = static_cast<T>(0);
            int64_t inout_index = outer_index + inner_index * depth * outer;
            for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
                int64_t index = inout_index + depth_index * outer;
                accumulator += input_data[index];
            }
            int64_t output_index = outer_index + inner_index * outer;
            output_data_temp[output_index] = accumulator;
        }
    }
    uint32_t result = ReduceSumOneAxes<T>(output_data_temp, input_shape, output_data, output_num, axes, axes_idx);
    delete[] output_data_temp;
    return result;
}

template <typename T, typename T2>
uint32_t ReduceSumCpuKernel::ReduceSumCompute2(const CpuKernelContext &ctx)
{
    auto *x = ctx.Input(kFirstInputIndex);
    auto *axes = ctx.Input(kSecondInputIndex);
    auto *y = ctx.Output(kFirstOutputIndex);

    std::vector<int64_t> input_shape = x->GetTensorShape()->GetDimSizes();
    auto *input_data = reinterpret_cast<T *>(x->GetData());
    auto *output_data = reinterpret_cast<T *>(y->GetData());
    int64_t input_num = x->NumElements();
    if (input_num <= 0) {
        KERNEL_LOG_ERROR("ReduceSum does not support empty complex input.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    const bool noop_with_empty_axes = GetNoopWithEmptyAxes(ctx);
    if (axes == nullptr || axes->GetDataSize() == 0) {
        if (noop_with_empty_axes) {
            for (int64_t i = 0; i < input_num; ++i) {
                output_data[i] = input_data[i];
            }
            return KERNEL_STATUS_OK;
        }
        if (ReduceSumSimpleCompute2<T, T2>(ctx, input_data, input_shape, output_data, true)) {
            return KERNEL_STATUS_OK;
        }
        KERNEL_LOG_ERROR("ReduceSum full reduction on empty axes failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ReduceSumSimpleCompute2<T, T2>(ctx, input_data, input_shape, output_data)) {
        return KERNEL_STATUS_OK;
    }
    std::vector<int64_t> axes_value;
    if (axes->GetDataType() == DT_INT32) {
        KERNEL_HANDLE_ERROR(ReduceSumDedupAxes<int32_t>(ctx, axes_value), "ReduceSum deduplicate failed.");
    } else {
        KERNEL_HANDLE_ERROR(ReduceSumDedupAxes<int64_t>(ctx, axes_value), "ReduceSum deduplicate failed.");
    }

    int64_t output_num = y->NumElements();
    uint32_t axes_idx = 0;
    KERNEL_HANDLE_ERROR((ReduceSumOneAxes2<T, T2>(input_data, input_num, input_shape, output_data, output_num,
                            axes_value, axes_idx)),
        "Reduce sum compute failed.");
    return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
bool ReduceSumCpuKernel::ReduceSumSimpleCompute2(
    const CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape, T *output_data,
    bool force_full_compute)
{
    if (input_shape.size() == 0) {
        output_data[0] = std::complex<T2>(input_data[0].real(), input_data[0].imag());
        return true;
    }
    int64_t input_num = ctx.Input(kFirstInputIndex)->NumElements();
    if (input_num <= 0) {
        KERNEL_LOG_ERROR("ReduceSum does not support empty complex input.");
        return false;
    }
    bool full_computer = force_full_compute;
    if (!full_computer) {
        if (ctx.Input(kSecondInputIndex)->GetDataType() == DT_INT32) {
            full_computer = IsReduceSumFullCompute<int32_t>(ctx, input_shape);
        } else {
            full_computer = IsReduceSumFullCompute<int64_t>(ctx, input_shape);
        }
    }
    if (full_computer) {
        auto accumulator_real = static_cast<T2>(0);
        auto accumulator_imag = static_cast<T2>(0);
        for (int64_t i = 0; i < input_num; i++) {
            accumulator_real += input_data[i].real();
            accumulator_imag += input_data[i].imag();
        }
        output_data[0] = std::complex<T2>(accumulator_real, accumulator_imag);
        KERNEL_LOG_INFO("Reduce sum full compute");
        return true;
    }
    return false;
}

template <typename T, typename T2>
uint32_t ReduceSumCpuKernel::ReduceSumOneAxes2(const T *input_data, int64_t input_num,
    std::vector<int64_t> input_shape, T *output_data, int64_t output_num, std::vector<int64_t> &axes,
    uint32_t &axes_idx)
{
    if (axes_idx >= axes.size()) {
        auto accumulator_real = static_cast<T2>(0);
        auto accumulator_imag = static_cast<T2>(0);
        for (int64_t i = 0; i < output_num; i++) {
            accumulator_real = input_data[i].real();
            accumulator_imag = input_data[i].imag();
            output_data[i] = std::complex<T2>(accumulator_real, accumulator_imag);
        }
        return KERNEL_STATUS_OK;
    }
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t depth = 1;
    KERNEL_HANDLE_ERROR(ReduceSumParseAxes(input_shape, axes, axes_idx, inner, outer, depth), "parse axes failed.");
    std::vector<T2> input_data_real(input_num);
    std::vector<T2> input_data_imag(input_num);
    for (int64_t i = 0; i < input_num; i++) {
        input_data_real[i] = input_data[i].real();
        input_data_imag[i] = input_data[i].imag();
    }
    int64_t output_num_temp = inner * outer;
    auto *output_data_temp = new (std::nothrow) T[output_num_temp];
    KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
    for (int64_t outer_index = 0; outer_index < outer; outer_index++) {
        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
            auto accumulator_real = static_cast<T2>(0);
            auto accumulator_imag = static_cast<T2>(0);
            int64_t inout_index = outer_index + inner_index * depth * outer;
            for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
                int64_t index = inout_index + depth_index * outer;
                accumulator_real += input_data_real[index];
                accumulator_imag += input_data_imag[index];
            }
            int64_t output_index = outer_index + inner_index * outer;
            output_data_temp[output_index] = std::complex<T2>(accumulator_real, accumulator_imag);
        }
    }
    uint32_t result = ReduceSumOneAxes2<T, T2>(
        output_data_temp, output_num_temp, input_shape, output_data, output_num, axes, axes_idx);
    delete[] output_data_temp;
    return result;
}

template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumDedupAxes(const CpuKernelContext &ctx, std::vector<int64_t> &axes)
{
    int32_t rank = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDims();
    auto *axes_data = reinterpret_cast<T *>(ctx.Input(kSecondInputIndex)->GetData());
    int64_t axes_num = ctx.Input(kSecondInputIndex)->NumElements();
    for (int64_t i = 0; i < axes_num; i++) {
        T axis = axes_data[i];
        KERNEL_CHECK_FALSE((axis < rank) && (axis >= -rank), KERNEL_STATUS_PARAM_INVALID,
            "axes[%ld] is out of input dims rank[%d]", static_cast<int64_t>(axis), rank);
        if (axis < 0) {
            axis += rank;
        }
        axes.push_back(axis);
    }
    int64_t j = 1;
    while (j < axes_num) {
        std::vector<int64_t>::iterator iter = std::find(axes.begin(), axes.begin() + j, axes[j]);
        if (iter != axes.begin() + j) {
            axes.erase(iter);
            axes_num--;
        } else {
            j++;
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t ReduceSumCpuKernel::ReduceSumParseAxes(std::vector<int64_t> &input_shape, std::vector<int64_t> &axes,
    uint32_t &axes_idx, int64_t &inner, int64_t &outer, int64_t &depth) const
{
    int64_t axis = axes[axes_idx];
    axes_idx++;
    int64_t rank = input_shape.size();
    for (int64_t i = 0; i < rank; i++) {
        if (i < axis) {
            inner *= input_shape[i];
        } else if (i > axis) {
            outer *= input_shape[i];
        } else {
            depth = input_shape[i];
            input_shape[i] = 1;
        }
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kReduceSum, ReduceSumCpuKernel);
}  // namespace aicpu