/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zeta_aicpu.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char* const kZeta = "Zeta";
const int64_t kZetaParallelNum = 64 * 1024;
} // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarZeta(T a, T b)
{
    return Eigen::numext::zeta(a, b);
}

inline uint32_t ParallelForZeta(const CpuKernelContext& ctx, int64_t total, int64_t per_unit_size,
                                const std::function<void(int64_t, int64_t)>& work)
{
    if (total > kZetaParallelNum) {
        return CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
    }
    work(0, total);
    return KERNEL_STATUS_OK;
}

template <typename T>
inline uint32_t ComputeZetaKernel(const CpuKernelContext& ctx)
{
    T* input0 = static_cast<T*>(ctx.Input(0)->GetData());
    T* input1 = static_cast<T*>(ctx.Input(1)->GetData());
    T* output = static_cast<T*>(ctx.Output(0)->GetData());
    int64_t total = ctx.Input(0)->NumElements();
    if (total == 0) {
        return KERNEL_STATUS_OK;
    }
    uint32_t cores = CpuKernelUtils::GetCPUNum(ctx);
    int64_t per_unit_size = total / std::min(std::max(1L, static_cast<long>(cores) - 2L), total);
    return ParallelForZeta(ctx, total, per_unit_size, [&](int64_t begin, int64_t end) {
        (void)std::transform(input0 + begin, input0 + end, input1 + begin, output + begin, ScalarZeta<T>);
    });
}

template <typename T>
inline uint32_t ComputeZeta(const CpuKernelContext& ctx)
{
    uint32_t result = ComputeZetaKernel<T>(ctx);
    if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Zeta compute failed.");
    }
    return result;
}

inline uint32_t ExtraCheckZeta(const CpuKernelContext& ctx)
{
    if (ctx.Input(0)->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Get input data failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Output(0)->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Get output data failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(0)->GetDataType() != ctx.Input(1)->GetDataType()) {
        KERNEL_LOG_ERROR("The data type of the first input [%s] need be the same as the second input [%s].",
                         DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Input(1)->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
        KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                         DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(0)->GetDataSize() != ctx.Input(1)->GetDataSize()) {
        KERNEL_LOG_ERROR("The data size of the first input [%lu] need be the same as the second input [%lu].",
                         ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
        KERNEL_LOG_ERROR("The data size of the input [%lu] need be the same as the output [%lu].",
                         ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

inline uint32_t CheckZeta(CpuKernelContext& ctx)
{
    return NormalCheck(ctx, kInputNum, kOutputNum) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckZeta(ctx);
}

inline uint32_t ComputeZeta(const CpuKernelContext& ctx)
{
    DataType input_type = ctx.Input(0)->GetDataType();
    switch (input_type) {
        case DT_FLOAT:
            return ComputeZeta<float>(ctx);
        case DT_DOUBLE:
            return ComputeZeta<double>(ctx);
        default:
            KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}
} // namespace detail

uint32_t ZetaCpuKernel::Compute(CpuKernelContext& ctx)
{
    return detail::CheckZeta(ctx) ? static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID) : detail::ComputeZeta(ctx);
}

REGISTER_CPU_KERNEL(kZeta, ZetaCpuKernel);
} // namespace aicpu
