/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "square_aicpu.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

using namespace std;
namespace {
const std::uint32_t kSquareInputNum{1};
const std::uint32_t kSquareOutputNum{1};
// parallel 3 level
constexpr int64_t kParallelLevel = 3;
constexpr int64_t kTwo = 2;
const std::int64_t Int32ParallelNums[kParallelLevel] = {96000, 192000, 384000};
const std::int64_t Int64ParallelNums[kParallelLevel] = {24000, 48000, 96000};
const std::int64_t Float16ParallelNums[kParallelLevel] = {24000, 48000, 96000};
const std::int64_t FloatParallelNums[kParallelLevel] = {192000, 384000, 512000};
const std::int64_t DoubleParallelNums[kParallelLevel] = {48000, 96000, 256000};
const std::int64_t ComplexParallelNums[kParallelLevel] = {24000, 96000, 192000};
constexpr int64_t kMinCoreNum = 1;
constexpr int64_t kUseCpuNumlevel1 = 6;
constexpr int64_t kUseCpuNumlevel2 = 8;
const char* const kSquare{"Square"};
} // namespace

namespace aicpu {
namespace detail {
template <typename T>
typename std::enable_if<
    std::is_same<T, std::complex<std::float_t>>::value || std::is_same<T, std::complex<std::double_t>>::value,
    void>::type inline ComputeKernel(T* input, T* output, int64_t begin, int64_t end)
{
    T* inner_input = input + begin;
    T* inner_output = output + begin;
    for (int64_t index = begin; index < end; ++index) {
        (*inner_output)
            .real((*inner_input).real() * (*inner_input).real() - (*inner_input).imag() * (*inner_input).imag());
        (*inner_output).imag(kTwo * ((*inner_input).real() * (*inner_input).imag()));
        inner_input++;
        inner_output++;
    }
}

template <typename T>
typename std::enable_if<std::is_same<T, int32_t>::value || std::is_same<T, double_t>::value, void>::
    type inline ComputeKernel(T* input, T* output, int64_t begin, int64_t end)
{
    int64_t length = end - begin;
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_x(input + begin, length);
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_y(output + begin, length);
    tensor_y = tensor_x.square();
}

template <typename T>
typename std::enable_if<
    std::is_same<T, int64_t>::value || std::is_same<T, float_t>::value || std::is_same<T, Eigen::half>::value,
    void>::type inline ComputeKernel(T* input, T* output, int64_t begin, int64_t end)
{
    T* inner_input = input + begin;
    T* inner_output = output + begin;

    for (int64_t index = begin; index < end; ++index) {
        *inner_output = (*inner_input) * (*inner_input);
        inner_input++;
        inner_output++;
    }
}

template <typename T>
uint32_t ComputeSquareKernel(const CpuKernelContext& ctx, const int64_t parallel[kParallelLevel])
{
    T* input = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    T* output = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    std::int64_t total = ctx.Input(0)->NumElements();
    int64_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    constexpr int PARALLEL_INDEX_ZERO = 0;
    constexpr int PARALLEL_INDEX_ONE = 1;
    constexpr int PARALLEL_INDEX_TWO = 2;
    bool parallel_flag = true;
    if (total > parallel[PARALLEL_INDEX_TWO]) {
    } else if (total > parallel[PARALLEL_INDEX_ONE]) {
        cores = (cores > kUseCpuNumlevel2) ? kUseCpuNumlevel2 : cores;
    } else if (total > parallel[PARALLEL_INDEX_ZERO]) {
        cores = (cores > kUseCpuNumlevel1) ? kUseCpuNumlevel1 : cores;
    } else {
        parallel_flag = false;
    }

    if (parallel_flag) {
        int64_t per_unit_size{total / std::min(std::max(kMinCoreNum, cores - kResvCpuNum), total)};
        return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
            ComputeKernel<T>(input, output, begin, end);
        });
    } else {
        ComputeKernel<T>(input, output, 0, total);
        return KERNEL_STATUS_OK;
    }
}

template <typename T>
inline std::uint32_t ComputeSquare(const CpuKernelContext& ctx, const int64_t parallel[kParallelLevel])
{
    uint32_t result = ComputeSquareKernel<T>(ctx, parallel);
    if (static_cast<int>(result) != 0) {
        KERNEL_LOG_ERROR("Square compute failed.");
    }
    return result;
}

inline std::uint32_t SquareExtraCheck(const CpuKernelContext& ctx)
{
    if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
        KERNEL_LOG_ERROR(
            "The data type of the input [%s] need be the same as the ouput [%s].",
            DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
    KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
    std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
    if (input_dims.size() != output_dims.size()) {
        KERNEL_LOG_ERROR(
            "The data dim of the input size [%lu] need be the same as the output "
            "size [%lu].",
            input_dims.size(), output_dims.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    for (size_t index = 0; index < input_dims.size(); index++) {
        if (input_dims[index] != output_dims[index]) {
            KERNEL_LOG_ERROR(
                "The data dim[%lu]=%ld of the input need be the same as the output "
                "dim[%lu]=%ld.",
                index, input_dims[index], index, output_dims[index]);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

std::uint32_t SquareCheck(CpuKernelContext& ctx)
{
    return NormalCheck(ctx, kSquareInputNum, kSquareOutputNum) ? KERNEL_STATUS_PARAM_INVALID : SquareExtraCheck(ctx);
}

std::uint32_t SquareCompute(const CpuKernelContext& ctx)
{
    DataType input_type{ctx.Input(0)->GetDataType()};
    switch (input_type) {
        case DT_INT32:
            return ComputeSquare<std::int32_t>(ctx, Int32ParallelNums);
        case DT_INT64:
            return ComputeSquare<std::int64_t>(ctx, Int64ParallelNums);
        case DT_FLOAT16:
            return ComputeSquare<Eigen::half>(ctx, Float16ParallelNums);
        case DT_FLOAT:
            return ComputeSquare<std::float_t>(ctx, FloatParallelNums);
        case DT_DOUBLE:
            return ComputeSquare<std::double_t>(ctx, DoubleParallelNums);
        case DT_COMPLEX64:
            return ComputeSquare<std::complex<std::float_t>>(ctx, ComplexParallelNums);
        case DT_COMPLEX128:
            return ComputeSquare<std::complex<std::double_t>>(ctx, ComplexParallelNums);
        default:
            KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}
} // namespace detail

std::uint32_t SquareCpuKernel::Compute(CpuKernelContext& ctx)
{
    return detail::SquareCheck(ctx) ? KERNEL_STATUS_PARAM_INVALID : detail::SquareCompute(ctx);
}

REGISTER_CPU_KERNEL(kSquare, SquareCpuKernel);
} // namespace aicpu
