/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rsqrt_aicpu.h"

#include <cfloat>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *const kRsqrt = "Rsqrt";
const std::uint32_t kRsqrtInputNum = 1;
const std::uint32_t kRsqrtOutputNum = 1;
constexpr int64_t kParallelDataNums = 8 * 1024;
constexpr int64_t kParallelComplexDataNums = 4 * 1024;
}  // namespace

namespace aicpu {
namespace {
// Compute the maximum number of parallel cores for the given data size.
int64_t GetMaxCoreNum(const CpuKernelContext &ctx, int64_t dataNum) {
    const std::uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(static_cast<int64_t>(min_core_num),
                                    static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)) - kResvCpuNum);
    return std::min(max_core_num, dataNum);
}
}  // namespace

template <typename T>
std::uint32_t RsqrtCpuKernel::RsqrtCompute(const Tensor *x, const Tensor *y, int64_t dataNum,
                                            const CpuKernelContext &ctx) const {
    auto inputx = reinterpret_cast<T *>(x->GetData());
    KERNEL_CHECK_NULLPTR(inputx, static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get input data failed")
    auto outputy = reinterpret_cast<T *>(y->GetData());
    KERNEL_CHECK_NULLPTR(outputy, static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get output data failed")
    if (dataNum <= kParallelDataNums) {
        for (int64_t i = 0; i < dataNum; i++) {
            outputy[i] = static_cast<T>(1) / sqrt(inputx[i]);
        }
    } else {
        int64_t max_core_num = GetMaxCoreNum(ctx, dataNum);
        auto shard_rsqrt = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                outputy[i] = static_cast<T>(1) / sqrt(inputx[i]);
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dataNum, dataNum / max_core_num, shard_rsqrt),
                            "Rsqrt Compute failed.");
    }
    return static_cast<std::uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
std::uint32_t RsqrtCpuKernel::RsqrtComputeComplex(const Tensor *x, const Tensor *y, int64_t dataNum,
                                                  const CpuKernelContext &ctx) const {
    auto inputx = reinterpret_cast<T *>(x->GetData());
    KERNEL_CHECK_NULLPTR(inputx, static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get input data failed")
    auto outputy = reinterpret_cast<T *>(y->GetData());
    KERNEL_CHECK_NULLPTR(outputy, static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get output data failed")
    if (dataNum <= kParallelComplexDataNums) {
        for (int64_t i = 0; i < dataNum; i++) {
            outputy[i] =
                sqrt(conj(inputx[i])) / sqrt(inputx[i].real() * inputx[i].real() + inputx[i].imag() * inputx[i].imag());
        }
    } else {
        int64_t max_core_num = GetMaxCoreNum(ctx, dataNum);
        auto shard_rsqrt = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                outputy[i] =
                    sqrt(conj(inputx[i])) / sqrt(inputx[i].real() * inputx[i].real() + inputx[i].imag() * inputx[i].imag());
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dataNum, dataNum / max_core_num, shard_rsqrt),
                            "Rsqrt Compute failed.");
    }
    return static_cast<std::uint32_t>(KERNEL_STATUS_OK);
}

std::uint32_t RsqrtCpuKernel::Compute(CpuKernelContext &ctx) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kRsqrtInputNum, kRsqrtOutputNum), "Check Rsqrt params failed.");
    if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
        KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s]",
                         DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                         DTypeStr(ctx.Output(0)->GetDataType()).c_str());
        return static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
        KERNEL_LOG_ERROR("The data size of the input [%lu] need be the same as the output [%lu]",
                         ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
        return static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    const Tensor *x = ctx.Input(0);
    const Tensor *y = ctx.Output(0);
    int64_t dataNum = x->NumElements();
    DataType datatype = x->GetDataType();
    std::uint32_t res = static_cast<std::uint32_t>(KERNEL_STATUS_OK);

    switch (datatype) {
        case DT_FLOAT16:
            res = RsqrtCompute<Eigen::half>(x, y, dataNum, ctx);
            break;
        case DT_FLOAT:
            res = RsqrtCompute<float>(x, y, dataNum, ctx);
            break;
        case DT_DOUBLE:
            res = RsqrtCompute<double>(x, y, dataNum, ctx);
            break;
        case DT_COMPLEX64:
            res = RsqrtComputeComplex<std::complex<float>>(x, y, dataNum, ctx);
            break;
        case DT_COMPLEX128:
            res = RsqrtComputeComplex<std::complex<double>>(x, y, dataNum, ctx);
            break;
        default:
            KERNEL_LOG_ERROR("Rsqrt invalid input type [%s]", DTypeStr(datatype).c_str());
            return static_cast<std::uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    if (res != static_cast<std::uint32_t>(KERNEL_STATUS_OK)) {
        return static_cast<std::uint32_t>(KERNEL_STATUS_INNER_ERROR);
    }
    return static_cast<std::uint32_t>(KERNEL_STATUS_OK);
}

REGISTER_CPU_KERNEL(kRsqrt, RsqrtCpuKernel);
}  // namespace aicpu
