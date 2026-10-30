/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumsum_aicpu.h"
#include <complex>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCumsumInputNum = 2U;
const uint32_t kCumsumOutputNum = 1U;
const uint32_t kReserveCores = 2U;
const int64_t kParalledDataSize = static_cast<int64_t>(512 * 1024);
const char* const kCumsum = "Cumsum";
#define CUMSUM_COMPUTE_CASE(DTYPE, TYPE, CTX)                                                     \
    case (DTYPE): {                                                                               \
        uint32_t result = CumsumCompute<TYPE>(CTX);                                               \
        if (result != KERNEL_STATUS_OK) {                                                         \
            KERNEL_LOG_ERROR("Cumsum kernel compute failed, dtype=%s.", DTypeStr(DTYPE).c_str()); \
            return result;                                                                        \
        }                                                                                         \
        break;                                                                                    \
    }

struct CumsumLayout {
    int64_t depth;
    int64_t outer;
    int64_t inner;
    int64_t depth_start;
    int64_t depth_step;
    int64_t depth_end;
    bool exclusive;
};

// Half-open index range [begin, end) for partitioning work across threads.
struct Range {
    int64_t begin;
    int64_t end;
};

CumsumLayout MakeLayout(int64_t depth, int64_t outer, int64_t inner, bool reverse, bool exclusive)
{
    CumsumLayout layout = {};
    layout.depth = depth;
    layout.outer = outer;
    layout.inner = inner;
    layout.exclusive = exclusive;
    if (reverse) {
        layout.depth_start = depth - 1;
        layout.depth_step = -1;
        layout.depth_end = -1;
    } else {
        layout.depth_start = 0;
        layout.depth_step = 1;
        layout.depth_end = depth;
    }
    return layout;
}
} // namespace

namespace aicpu {
namespace {
bool IsScalarTensor(const CpuKernelContext& ctx)
{
    return (ctx.Input(0)->GetTensorShape()->GetDims() == 0) && (ctx.Input(0)->NumElements() == 1);
}

int32_t ParseAxis(const CpuKernelContext& ctx)
{
    auto axis_data = PtrToPtr<void, int32_t>(ctx.Input(1)->GetData());
    int32_t axis = 0;
    if (axis_data != nullptr) {
        axis = *axis_data;
    }
    if (axis < 0) {
        axis += ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDims();
    }
    return axis;
}
} // namespace

// Accumulate one row of outer elements in exclusive mode: write accumulator before adding input.
template <typename T>
inline void AccumulateRowExclusive(const T* __restrict__ in, T* __restrict__ out, T* __restrict__ acc, int64_t span)
{
    for (int64_t o = 0; o < span; ++o) {
        out[o] = acc[o];
        acc[o] += in[o];
    }
}

// Accumulate one row of outer elements in inclusive mode: add input before writing accumulator.
template <typename T>
inline void AccumulateRowInclusive(const T* __restrict__ in, T* __restrict__ out, T* __restrict__ acc, int64_t span)
{
    for (int64_t o = 0; o < span; ++o) {
        acc[o] += in[o];
        out[o] = acc[o];
    }
}

// Compute cumsum for sequences where depth elements are contiguous in memory (outer == 1).
// Sequential dependency along depth prevents vectorization, but the contiguous access pattern
// is optimal for the hardware prefetcher. Parallelize over independent sequences (inner).
template <typename T>
uint32_t ComputeCumsumContiguous(
    const T* __restrict__ input, T* __restrict__ output, const CumsumLayout& layout, Range seqs)
{
    for (int64_t seq = seqs.begin; seq < seqs.end; ++seq) {
        const T* __restrict__ in = input + seq * layout.depth;
        T* __restrict__ out = output + seq * layout.depth;
        auto acc = static_cast<T>(0);
        if (layout.exclusive) {
            for (int64_t d = layout.depth_start; d != layout.depth_end; d += layout.depth_step) {
                out[d] = acc;
                acc += in[d];
            }
        } else {
            for (int64_t d = layout.depth_start; d != layout.depth_end; d += layout.depth_step) {
                acc += in[d];
                out[d] = acc;
            }
        }
    }
    return KERNEL_STATUS_OK;
}

// Compute cumsum with loop order optimized for outer > 1: outer loop over depth, inner loop
// over a contiguous slice of the outer dimension. The inner-loop iterations are independent,
// enabling GCC to auto-vectorize with NEON instructions on aarch64.
// Uses __builtin_prefetch to overlap next depth-row fetch with current-row computation.
template <typename T>
uint32_t ComputeCumsumBatched(
    const T* __restrict__ input, T* __restrict__ output, const CumsumLayout& layout, Range inners, Range outers)
{
    const int64_t outer_span = outers.end - outers.begin;
    std::vector<T> acc(static_cast<size_t>(outer_span), static_cast<T>(0));

    for (int64_t inner_idx = inners.begin; inner_idx < inners.end; ++inner_idx) {
        const int64_t base = inner_idx * layout.depth * layout.outer;
        std::fill(acc.begin(), acc.end(), static_cast<T>(0));

        for (int64_t d = layout.depth_start; d != layout.depth_end; d += layout.depth_step) {
            const int64_t row_offset = base + d * layout.outer;
            const T* __restrict__ in_row = input + row_offset + outers.begin;
            T* __restrict__ out_row = output + row_offset + outers.begin;

            if (d + layout.depth_step != layout.depth_end) {
                __builtin_prefetch(input + base + (d + layout.depth_step) * layout.outer + outers.begin, 0, 1);
            }

            if (layout.exclusive) {
                AccumulateRowExclusive<T>(in_row, out_row, acc.data(), outer_span);
            } else {
                AccumulateRowInclusive<T>(in_row, out_row, acc.data(), outer_span);
            }
        }
    }
    return KERNEL_STATUS_OK;
}

// Small-tensor single-threaded dispatch by memory access pattern.
template <typename T>
uint32_t CumsumComputeSmall(const T* __restrict__ input, T* __restrict__ output, const CumsumLayout& layout)
{
    if (layout.outer == 1) {
        return ComputeCumsumContiguous<T>(input, output, layout, {0, layout.inner});
    }
    return ComputeCumsumBatched<T>(input, output, layout, {0, layout.inner}, {0, layout.outer});
}

// Large-tensor multi-threaded dispatch with three parallelization strategies:
// Path 1 (outer==1): parallelize over inner sequences.
// Path 2 (inner>=cores): parallelize inner, each thread processes full outer range.
// Path 3 (inner<cores): chunk outer dimension to create enough parallel work units.
template <typename T>
uint32_t CumsumComputeParallel(
    CpuKernelContext& ctx, const T* __restrict__ input, T* __restrict__ output, const CumsumLayout& layout)
{
    const uint32_t cpu_num = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    const int64_t avail_cores = static_cast<int64_t>(std::max(1U, std::max(cpu_num, kReserveCores) - kReserveCores));

    if (layout.outer == 1) {
        const int64_t max_core_num = std::min(avail_cores, layout.inner);
        const int64_t per_unit_size = layout.inner / max_core_num;
        KERNEL_LOG_INFO("Cumsum parallel: path=contiguous, cores=%ld, per_unit=%ld", avail_cores, per_unit_size);
        auto shard = [input, output, &layout](int64_t begin, int64_t end) {
            (void)ComputeCumsumContiguous<T>(input, output, layout, {begin, end});
        };
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, layout.inner, per_unit_size, shard),
            "Cumsum ParallelFor failed: inner=%ld, depth=%ld", layout.inner, layout.depth)
    } else if (layout.inner >= avail_cores) {
        const int64_t per_unit_size = layout.inner / avail_cores;
        KERNEL_LOG_INFO("Cumsum parallel: path=batched, cores=%ld, per_unit=%ld", avail_cores, per_unit_size);
        auto shard = [input, output, &layout](int64_t begin, int64_t end) {
            (void)ComputeCumsumBatched<T>(input, output, layout, {begin, end}, {0, layout.outer});
        };
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, layout.inner, per_unit_size, shard),
            "Cumsum ParallelFor failed: inner=%ld, outer=%ld, depth=%ld", layout.inner, layout.outer, layout.depth)
    } else {
        const int64_t chunks_per_inner = std::max(1L, (avail_cores + layout.inner - 1) / layout.inner);
        const int64_t chunk_size = std::max(1L, (layout.outer + chunks_per_inner - 1) / chunks_per_inner);
        const int64_t actual_chunks = (layout.outer + chunk_size - 1) / chunk_size;
        const int64_t total_units = layout.inner * actual_chunks;
        const int64_t per_unit_size = std::max(1L, total_units / avail_cores);
        KERNEL_LOG_INFO(
            "Cumsum parallel: path=outer_chunking, cores=%ld, chunks=%ld, chunk_size=%ld", avail_cores, actual_chunks,
            chunk_size);

        auto shard = [input, output, layout, actual_chunks, chunk_size](int64_t begin, int64_t end) {
            for (int64_t unit = begin; unit < end; ++unit) {
                const int64_t inner_idx = unit / actual_chunks;
                const int64_t chunk_idx = unit % actual_chunks;
                const int64_t o_begin = chunk_idx * chunk_size;
                const int64_t o_end = std::min(o_begin + chunk_size, layout.outer);
                (void)ComputeCumsumBatched<T>(input, output, layout, {inner_idx, inner_idx + 1}, {o_begin, o_end});
            }
        };
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, total_units, per_unit_size, shard),
            "Cumsum ParallelFor failed: inner=%ld, outer=%ld, chunks=%ld", layout.inner, layout.outer, actual_chunks)
    }
    return KERNEL_STATUS_OK;
}

void CumsumCpuKernel::AxesCal(
    const CpuKernelContext& ctx, int64_t& inner, int64_t& outer, int64_t& depth, const int32_t& axis) const
{
    auto shape = ctx.Input(kFirstInputIndex)->GetTensorShape();
    const int64_t rank = shape->GetDims();
    for (int32_t i = 0; i < rank; ++i) {
        if (i < axis) {
            inner *= shape->GetDimSize(i);
        } else if (i > axis) {
            outer *= shape->GetDimSize(i);
        } else {
            depth = shape->GetDimSize(i);
        }
    }
}

uint32_t CumsumCpuKernel::Compute(CpuKernelContext& ctx)
{
    // check params
    KERNEL_HANDLE_ERROR(
        NormalCheck(ctx, kCumsumInputNum, kCumsumOutputNum), "[%s] check input and output failed.", kCumsum);
    // parse params
    KERNEL_HANDLE_ERROR(CumsumCheck(ctx), "[%s] check params failed.", kCumsum);
    auto input_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
    switch (input_data_type) {
        CUMSUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        CUMSUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
        CUMSUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
        CUMSUM_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
        default:
            KERNEL_LOG_ERROR("Cumsum kernel data type [%s] not support.", DTypeStr(input_data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t CumsumCpuKernel::CumsumCheck(const CpuKernelContext& ctx)
{
    KERNEL_CHECK_NULLPTR(ctx.Input(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
    KERNEL_CHECK_NULLPTR(
        ctx.Output(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.");

    if (ctx.Input(1)->GetData() != nullptr) {
        KERNEL_CHECK_FALSE(
            (ctx.Input(1)->GetDataType() == DT_INT32 || ctx.Input(1)->GetDataType() == DT_INT64),
            KERNEL_STATUS_PARAM_INVALID, "Axis data type [%u] not supported, expect INT32 or INT64.",
            ctx.Input(1)->GetDataType());
        KERNEL_CHECK_FALSE(
            ctx.Input(1)->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID,
            "Axis tensor must be scalar, got num_elements=%ld.", ctx.Input(1)->NumElements())
    }
    std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();

    KERNEL_CHECK_FALSE(
        (shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
        "Input rank [%zu] must equal output rank [%zu].", shape_input.size(), shape_output.size())
    return KERNEL_STATUS_OK;
}

void CumsumCpuKernel::CumsumGetAttr(const CpuKernelContext& ctx, bool& exclusive, bool& reverse) const
{
    exclusive = false;
    AttrValue* exclusive_attr = ctx.GetAttr("exclusive");
    if (exclusive_attr != nullptr) {
        exclusive = exclusive_attr->GetBool();
    }

    reverse = false;
    AttrValue* reverse_attr = ctx.GetAttr("reverse");
    if (reverse_attr != nullptr) {
        reverse = reverse_attr->GetBool();
    }
}

template <typename T>
uint32_t CumsumCpuKernel::CumsumCompute(CpuKernelContext& ctx) const
{
    auto input_data = PtrToPtr<void, T>(ctx.Input(0)->GetData());
    auto output_data = PtrToPtr<void, T>(ctx.Output(0)->GetData());

    if (IsScalarTensor(ctx)) {
        KERNEL_LOG_INFO("Cumsum: scalar tensor, direct copy.");
        output_data[0] = input_data[0];
        return KERNEL_STATUS_OK;
    }

    int32_t axis = ParseAxis(ctx);
    bool exclusive;
    bool reverse;
    CumsumGetAttr(ctx, exclusive, reverse);

    int64_t inner = 1;
    int64_t outer = 1;
    int64_t depth = 1;
    AxesCal(ctx, inner, outer, depth, axis);

    const int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements();
    const int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
    CumsumLayout layout = MakeLayout(depth, outer, inner, reverse, exclusive);

    KERNEL_LOG_INFO(
        "Cumsum: axis=%d, exclusive=%d, reverse=%d, inner=%ld, outer=%ld, depth=%ld, data_size=%ld", axis, exclusive,
        reverse, inner, outer, depth, data_size);

    if (data_size <= kParalledDataSize) {
        return CumsumComputeSmall<T>(input_data, output_data, layout);
    }
    return CumsumComputeParallel<T>(ctx, input_data, output_data, layout);
}

REGISTER_CPU_KERNEL(kCumsum, CumsumCpuKernel);
} // namespace aicpu
