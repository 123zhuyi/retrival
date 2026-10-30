/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "sqrt_aicpu.h"

#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *const kSqrt = "Sqrt";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 8 * 1024;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t SqrtCpuKernel::DoComputeReal(const CpuKernelContext &ctx) {
    auto *input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    auto *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Input(0)->NumElements();

    if (data_num <= kParallelDataNums) {
        Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> tensor_x(input, data_num);
        Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> tensor_y(output, data_num);
        tensor_y = tensor_x.sqrt();
    } else {
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        max_core_num = max_core_num > data_num ? data_num : max_core_num;
        auto shard_sqrt = [&input, &output](int64_t begin, int64_t end) {
            int64_t length = end - begin;
            Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> tensor_x(input + begin, length);
            Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> tensor_y(output + begin, length);
            tensor_y = tensor_x.sqrt();
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sqrt),
                            "Sqrt Compute failed.");
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SqrtCpuKernel::DoComputeComplex(const CpuKernelContext &ctx) {
    auto *input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    auto *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Input(0)->NumElements();

    auto shard_sqrt = [&input, &output](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    };

    if (data_num <= kParallelDataNums) {
        shard_sqrt(0, data_num);
    } else {
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        max_core_num = max_core_num > data_num ? data_num : max_core_num;
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sqrt),
                            "Sqrt Compute failed.");
    }
    return KERNEL_STATUS_OK;
}

uint32_t SqrtCpuKernel::Compute(CpuKernelContext &ctx) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check Sqrt params failed.");

    DataType input_type = ctx.Input(0)->GetDataType();
    DataType output_type = ctx.Output(0)->GetDataType();
    KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of input [%s] must be the same as output [%s].",
                       DTypeStr(input_type).c_str(), DTypeStr(output_type).c_str());
    KERNEL_CHECK_FALSE((ctx.Input(0)->GetDataSize() == ctx.Output(0)->GetDataSize()), KERNEL_STATUS_PARAM_INVALID,
                       "The data size of input [%lu] must be the same as output [%lu].",
                       ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());

    KERNEL_LOG_DEBUG("%s op input[x] data type is [%s].", kSqrt, DTypeStr(input_type).c_str());
    switch (input_type) {
        case DT_FLOAT16:
            return DoComputeReal<Eigen::half>(ctx);
        case DT_FLOAT:
            return DoComputeReal<float>(ctx);
        case DT_DOUBLE:
            return DoComputeReal<double>(ctx);
        case DT_COMPLEX64:
            return DoComputeComplex<complex<float>>(ctx);
        case DT_COMPLEX128:
            return DoComputeComplex<complex<double>>(ctx);
        default:
            KERNEL_LOG_ERROR("Sqrt invalid input type [%s].", DTypeStr(input_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

REGISTER_CPU_KERNEL(kSqrt, SqrtCpuKernel);
}  // namespace aicpu
