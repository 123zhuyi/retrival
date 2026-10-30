/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "clip_by_value_v2_aicpu.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>   // for Eigen::half / Eigen::bfloat16 type definitions only
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char* const kClipByValueV2 = "ClipByValueV2";
constexpr uint32_t kInputNum = 3U;
constexpr uint32_t kOutputNum = 1U;
constexpr uint32_t kIndexTwo = 2U;
constexpr int64_t kParallelBytesThresh = 200LL * 1024LL;
constexpr int64_t kComplexParallelBytesThresh = 40LL * 1024LL;
constexpr int64_t kCpuReserveK = 2;
} // namespace

namespace aicpu {
template <typename T>
__attribute__((always_inline)) inline T ClampScalar(T x, T mn, T mx) noexcept
{
    if constexpr (std::is_floating_point_v<T>) {
        // Standard floating point: any NaN yields NaN. std::isnan(NaN) == true.
        if (std::isnan(x) || std::isnan(mn) || std::isnan(mx)) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        return std::min(std::max(x, mn), mx);
    } else if constexpr (std::is_same_v<T, Eigen::half> || std::is_same_v<T, Eigen::bfloat16>) {
        // Eigen half precision: compute in float to preserve NaN semantics.
        float xf = static_cast<float>(x);
        float mnf = static_cast<float>(mn);
        float mxf = static_cast<float>(mx);
        if (std::isnan(xf) || std::isnan(mnf) || std::isnan(mxf)) {
            return static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        }
        return static_cast<T>(std::min(std::max(xf, mnf), mxf));
    } else {
        return std::min(std::max(x, mn), mx);
    }
}

template <typename T>
__attribute__((always_inline)) inline std::complex<T> ClampComplex(
    std::complex<T> x, std::complex<T> mn, std::complex<T> mx) noexcept
{
    const T nx = std::norm(x);
    const T nmn = std::norm(mn);
    const T nmx = std::norm(mx);
    std::complex<T> tmp = (nx <= nmn) ? mn : x;
    const T ntmp = (nx <= nmn) ? nmn : nx;
    return (ntmp <= nmx) ? tmp : mx;
}

template <typename T>
__attribute__((hot)) static void KernelScalarBound(
    const T* __restrict__ x, T* __restrict__ y, int64_t beg, int64_t end, T mn_v, T mx_v) noexcept
{
    for (int64_t i = beg; i < end; ++i) {
        y[i] = ClampScalar<T>(x[i], mn_v, mx_v);
    }
}

template <typename T>
__attribute__((hot)) static void KernelElemBound(
    const T* __restrict__ x, const T* __restrict__ mn, const T* __restrict__ mx, T* __restrict__ y, int64_t beg,
    int64_t end) noexcept
{
    for (int64_t i = beg; i < end; ++i) {
        y[i] = ClampScalar<T>(x[i], mn[i], mx[i]);
    }
}

template <typename T>
__attribute__((hot)) static void KernelComplexScalar(
    const std::complex<T>* __restrict__ x, std::complex<T>* __restrict__ y, int64_t beg, int64_t end,
    std::complex<T> mn_v, std::complex<T> mx_v) noexcept
{
    for (int64_t i = beg; i < end; ++i) {
        y[i] = ClampComplex<T>(x[i], mn_v, mx_v);
    }
}

template <typename T>
__attribute__((hot)) static void KernelComplexElem(
    const std::complex<T>* __restrict__ x, const std::complex<T>* __restrict__ mn,
    const std::complex<T>* __restrict__ mx, std::complex<T>* __restrict__ y, int64_t beg, int64_t end) noexcept
{
    for (int64_t i = beg; i < end; ++i) {
        y[i] = ClampComplex<T>(x[i], mn[i], mx[i]);
    }
}

// parallel dispatch wrapper -- adaptive serial fallback + byte-level grain.
template <typename Body>
static uint32_t DispatchParallel(
    const CpuKernelContext& ctx, const char* branch_tag, int64_t total, int64_t per_unit_bytes,
    int64_t bytes_thresh, Body&& body)
{
    const int64_t total_bytes = total * per_unit_bytes;
    if (total_bytes < bytes_thresh) {
        KERNEL_LOG_INFO(
            "[%s] branch=%s serial path: total=%ld, total_bytes=%ld < thresh=%ld.",
            ctx.GetOpType().c_str(), branch_tag, total, total_bytes, bytes_thresh);
        body(static_cast<int64_t>(0), total);
        return KERNEL_STATUS_OK;
    }

    const int64_t cpu_num = static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx));
    const int64_t max_core_num = std::max<int64_t>(1, std::max(cpu_num, kCpuReserveK) - kCpuReserveK);
    const int64_t per_unit = CeilMultiple(total, max_core_num);
    const int64_t grain_bytes = per_unit * per_unit_bytes;

    KERNEL_LOG_INFO(
        "[%s] branch=%s parallel path: total=%ld, total_bytes=%ld, cpu_num=%ld, cores=%ld, grain_bytes=%ld.",
        ctx.GetOpType().c_str(), branch_tag, total, total_bytes, cpu_num, max_core_num, grain_bytes);

    auto sharde = [&body](size_t b, size_t e) {
        body(static_cast<int64_t>(b), static_cast<int64_t>(e));
    };
    const uint32_t rc = CpuKernelUtils::ParallelFor(ctx, total, grain_bytes, sharde);
    if (rc != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR(
            "[%s] ParallelFor failed, branch=%s, rc=%u, total=%ld, grain_bytes=%ld, cores=%ld.",
            ctx.GetOpType().c_str(), branch_tag, rc, total, grain_bytes, max_core_num);
        return rc;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
static uint32_t DoCompute(const CpuKernelContext& ctx)
{
    const T* __restrict__ x_ptr = aicpu::PtrToPtr<void, const T>(ctx.Input(0)->GetData());
    const T* __restrict__ mn_ptr = aicpu::PtrToPtr<void, const T>(ctx.Input(1)->GetData());
    const T* __restrict__ mx_ptr = aicpu::PtrToPtr<void, const T>(ctx.Input(kIndexTwo)->GetData());
    T* __restrict__ y_ptr = aicpu::PtrToPtr<void, T>(ctx.Output(0)->GetData());

    const int64_t data_num = ctx.Input(0)->NumElements();
    const bool is_scalar = (ctx.Input(1)->NumElements() == 1);

    // Degenerate form: single-element output computed directly.
    if (__builtin_expect(data_num == 1, 0)) {
        y_ptr[0] = ClampScalar<T>(x_ptr[0], mn_ptr[0], mx_ptr[0]);
        return KERNEL_STATUS_OK;
    }

    constexpr int64_t elem_bytes = static_cast<int64_t>(sizeof(T));

    if (is_scalar) {
        // Specialized branch: min/max broadcast; preload the scalars into local registers.
        const T mn_v = mn_ptr[0];
        const T mx_v = mx_ptr[0];
        auto body = [x_ptr, y_ptr, mn_v, mx_v](int64_t b, int64_t e) {
            KernelScalarBound<T>(x_ptr, y_ptr, b, e, mn_v, mx_v);
        };
        return DispatchParallel(ctx, "real-scalar", data_num, elem_bytes, kParallelBytesThresh, body);
    }

    auto body = [x_ptr, mn_ptr, mx_ptr, y_ptr](int64_t b, int64_t e) {
        KernelElemBound<T>(x_ptr, mn_ptr, mx_ptr, y_ptr, b, e);
    };
    return DispatchParallel(ctx, "real-elem", data_num, elem_bytes, kParallelBytesThresh, body);
}

template <typename T>
static uint32_t DoComplexCompute(const CpuKernelContext& ctx)
{
    using C = std::complex<T>;
    const C* __restrict__ x_ptr = aicpu::PtrToPtr<void, const C>(ctx.Input(0)->GetData());
    const C* __restrict__ mn_ptr = aicpu::PtrToPtr<void, const C>(ctx.Input(1)->GetData());
    const C* __restrict__ mx_ptr = aicpu::PtrToPtr<void, const C>(ctx.Input(kIndexTwo)->GetData());
    C* __restrict__ y_ptr = aicpu::PtrToPtr<void, C>(ctx.Output(0)->GetData());

    const int64_t data_num = ctx.Input(0)->NumElements();
    const bool is_scalar = (ctx.Input(1)->NumElements() == 1);

    if (__builtin_expect(data_num == 1, 0)) {
        y_ptr[0] = ClampComplex<T>(x_ptr[0], mn_ptr[0], mx_ptr[0]);
        return KERNEL_STATUS_OK;
    }

    constexpr int64_t elem_bytes = static_cast<int64_t>(sizeof(C));

    if (is_scalar) {
        const C mn_v = mn_ptr[0];
        const C mx_v = mx_ptr[0];
        auto body = [x_ptr, y_ptr, mn_v, mx_v](int64_t b, int64_t e) {
            KernelComplexScalar<T>(x_ptr, y_ptr, b, e, mn_v, mx_v);
        };
        return DispatchParallel(ctx, "complex-scalar", data_num, elem_bytes, kComplexParallelBytesThresh, body);
    }

    auto body = [x_ptr, mn_ptr, mx_ptr, y_ptr](int64_t b, int64_t e) {
        KernelComplexElem<T>(x_ptr, mn_ptr, mx_ptr, y_ptr, b, e);
    };
    return DispatchParallel(ctx, "complex-elem", data_num, elem_bytes, kComplexParallelBytesThresh, body);
}

static uint32_t GetInputAndCheck(const CpuKernelContext& ctx)
{
    auto x_tensor = ctx.Input(0);
    auto min_tensor = ctx.Input(1);
    auto max_tensor = ctx.Input(kIndexTwo);
    auto y_tensor = ctx.Output(0);
    const auto dtype = x_tensor->GetDataType();
    if ((dtype != min_tensor->GetDataType()) || (dtype != max_tensor->GetDataType()) ||
        (dtype != y_tensor->GetDataType())) {
        KERNEL_LOG_ERROR(
            "[%s] dtype mismatch: x=[%s], clip_value_min=[%s], clip_value_max=[%s], y=[%s], expect all equal.",
            ctx.GetOpType().c_str(), DTypeStr(dtype).c_str(), DTypeStr(min_tensor->GetDataType()).c_str(),
            DTypeStr(max_tensor->GetDataType()).c_str(), DTypeStr(y_tensor->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (x_tensor->NumElements() != y_tensor->NumElements()) {
        KERNEL_LOG_ERROR(
            "[%s] numelements mismatch: x=%ld, y=%ld, expect equal.",
            ctx.GetOpType().c_str(), x_tensor->NumElements(), y_tensor->NumElements());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if ((min_tensor->NumElements() == 1) && (max_tensor->NumElements() == 1)) {
        return KERNEL_STATUS_OK;
    }

    const auto& x_shape = x_tensor->GetTensorShape()->GetDimSizes();
    const auto& min_shape = min_tensor->GetTensorShape()->GetDimSizes();
    const auto& max_shape = max_tensor->GetTensorShape()->GetDimSizes();
    if (x_shape != min_shape || x_shape != max_shape) {
        KERNEL_LOG_ERROR(
            "[%s] shape mismatch: x=[%s], min=[%s], max=[%s], expect min/max to be scalar or same shape as x.",
            ctx.GetOpType().c_str(), VectorToString(x_shape).c_str(), VectorToString(min_shape).c_str(),
            VectorToString(max_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

static uint32_t DispatchByDtype(const CpuKernelContext& ctx, DataType dtype)
{
    switch (dtype) {
        case DT_DOUBLE:
            return DoCompute<double>(ctx);
        case DT_FLOAT:
            return DoCompute<float>(ctx);
        case DT_FLOAT16:
            return DoCompute<Eigen::half>(ctx);
        case DT_BFLOAT16:
            return DoCompute<Eigen::bfloat16>(ctx);
        case DT_INT8:
        case DT_QINT8:
            return DoCompute<int8_t>(ctx);
        case DT_INT16:
            return DoCompute<int16_t>(ctx);
        case DT_INT32:
        case DT_QINT32:
            return DoCompute<int32_t>(ctx);
        case DT_INT64:
            return DoCompute<int64_t>(ctx);
        case DT_UINT8:
        case DT_QUINT8:
            return DoCompute<uint8_t>(ctx);
        case DT_UINT16:
            return DoCompute<uint16_t>(ctx);
        case DT_UINT32:
            return DoCompute<uint32_t>(ctx);
        case DT_UINT64:
            return DoCompute<uint64_t>(ctx);
        case DT_COMPLEX64:
            return DoComplexCompute<float>(ctx);
        case DT_COMPLEX128:
            return DoComplexCompute<double>(ctx);
        default:
            KERNEL_LOG_ERROR(
                "[%s] input dtype=%d (%s) not supported, expect one of "
                "{double,float,float16,bfloat16,int8/16/32/64,uint8/16/32/64,qint8/32,quint8,complex64/128}.",
                ctx.GetOpType().c_str(), static_cast<int>(dtype), DTypeStr(dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

uint32_t ClipByValueV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ClipByValueV2 params failed.");
    if (ctx.Input(0)->NumElements() == 0) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor, skip compute.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }

    KERNEL_HANDLE_ERROR(GetInputAndCheck(ctx), "Check ClipByValueV2 params failed.");

    const DataType input_dtype = ctx.Input(0)->GetDataType();
    const int64_t total_elems = ctx.Input(0)->NumElements();
    const bool is_scalar_bounds = (ctx.Input(1)->NumElements() == 1);
    KERNEL_LOG_INFO(
        "[%s] Compute begin: dtype=%d (%s), total_elems=%ld, is_scalar_bounds=%d.",
        ctx.GetOpType().c_str(), static_cast<int>(input_dtype), DTypeStr(input_dtype).c_str(),
        total_elems, static_cast<int>(is_scalar_bounds));

    return DispatchByDtype(ctx, input_dtype);
}

REGISTER_CPU_KERNEL(kClipByValueV2, ClipByValueV2CpuKernel);
} // namespace aicpu