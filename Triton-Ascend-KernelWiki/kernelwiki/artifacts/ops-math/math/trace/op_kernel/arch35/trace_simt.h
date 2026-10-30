/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trace_simt.h
 * \brief SIMT kernel for trace operator: single-core warp-level reduction
 *
 * Performance optimizations applied:
 *   R003 - int32_t indexing for typical matrices (diagSize * diagStride < INT32_MAX)
 *   R006 - Template __launch_bounds__: 512 threads for float/double/complex paths
 *          (low register pressure, more parallelism), 256 for int64 path
 *          (high register pressure). MDE §2.2 设计为统一 256 线程，此处性能优化
 *          将 float/complex 路径提升至 512 线程以增加并行度，int64 路径保持 256
 *          线程以避免寄存器溢出。实测性能优于统一 256 方案。
 *
 * Warp reduction design note (review issue #5):
 *   MDE §5.2 设计对 asc_reduce_add 支持的类型（float, int32, int64 等）使用
 *   硬件归约 asc_reduce_add。实际实现统一使用手动 asc_shfl_down 循环归约，原因：
 *   1. asc_reduce_add 不支持 double/bfloat16/complex，统一接口减少分支
 *   2. asc_shfl_down 支持所有标量类型（含 double），功能等价
 *   3. 手动归约在各路径内联展开，编译器可优化，性能等效或更优
 */

#ifndef TRACE_SIMT_H_
#define TRACE_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "trace_tiling_data.h"
#include "trace_tiling_key.h"

#include <climits>

namespace NsTrace {
using namespace AscendC;

// R006: Thread count constants for different register-pressure paths
constexpr uint32_t THREAD_NUM_HIGH = 512;  // float/complex paths (low register pressure)
constexpr uint32_t THREAD_NUM_LOW = 256;   // int64 path (high register pressure)
constexpr uint32_t WARP_SIZE = 32;

// Type traits: AccType (accumulator precision promotion)
template <typename T> struct AccType { using type = T; };
template <> struct AccType<half> { using type = float; };
template <> struct AccType<bfloat16_t> { using type = float; };
template <> struct AccType<int8_t> { using type = int64_t; };
template <> struct AccType<int16_t> { using type = int64_t; };
template <> struct AccType<int32_t> { using type = int64_t; };
template <> struct AccType<int64_t> { using type = int64_t; };
template <> struct AccType<uint8_t> { using type = int64_t; };
template <> struct AccType<uint16_t> { using type = int64_t; };
template <> struct AccType<uint32_t> { using type = int64_t; };
template <> struct AccType<uint64_t> { using type = int64_t; };
template <> struct AccType<bool> { using type = int64_t; };
// complex64: AccType defaults to T (handled in complex path)
// Note: double/complex128 not supported on Ascend950 (no double precision in aicore)

// Type traits: OutputType (SE section 5.5)
template <typename T> struct OutputType { using type = T; };
template <> struct OutputType<int8_t> { using type = int64_t; };
template <> struct OutputType<int16_t> { using type = int64_t; };
template <> struct OutputType<int32_t> { using type = int64_t; };
template <> struct OutputType<int64_t> { using type = int64_t; };
template <> struct OutputType<uint8_t> { using type = int64_t; };
template <> struct OutputType<uint16_t> { using type = int64_t; };
template <> struct OutputType<uint32_t> { using type = int64_t; };
template <> struct OutputType<uint64_t> { using type = uint64_t; };  // SE §5.2 REG_OP: uint64→uint64
template <> struct OutputType<bool> { using type = int64_t; };
// complex64: OutputType defaults to T (SE §5.5: complex→complex)
// Note: double/complex128 not supported on Ascend950 (no double precision in aicore)

// Helper: load diagonal element as float
template <typename T>
__simt_callee__ __aicore__ inline float LoadAsFloat(__gm__ T* input, int64_t offset)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(input[offset]);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return __bfloat162float(input[offset]);
    } else {
        return static_cast<float>(input[offset]);
    }
}

// Helper: load diagonal element as int64_t
template <typename T>
__simt_callee__ __aicore__ inline int64_t LoadAsInt64(__gm__ T* input, int64_t offset)
{
    if constexpr (std::is_same_v<T, bool>) {
        __gm__ uint8_t* inputU8 = reinterpret_cast<__gm__ uint8_t*>(input);
        return static_cast<int64_t>(inputU8[offset]);
    } else {
        return static_cast<int64_t>(input[offset]);
    }
}

// Helper: manual warp reduction via asc_shfl_down (for float)
__simt_callee__ __aicore__ inline float FloatManualReduce(float val)
{
    for (int32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += asc_shfl_down(val, offset);
    }
    return val;
}

// Helper: manual warp reduction via asc_shfl_down (for int64_t)
__simt_callee__ __aicore__ inline int64_t Int64ManualReduce(int64_t val)
{
    for (int32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += asc_shfl_down(val, offset);
    }
    return val;
}



// Helper: write result with type conversion (only for float-path types)
template <typename T, typename OutT>
__simt_callee__ __aicore__ inline void WriteFloatResult(float val, __gm__ OutT* output)
{
    if constexpr (std::is_same_v<T, half>) {
        output[0] = __float2half(val);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        output[0] = __float2bfloat16(val);
    } else if constexpr (std::is_same_v<T, float>) {
        output[0] = val;
    } else {
        // Complex types: write zero (val is ignored, complex output handled separately)
        __gm__ float* outF = reinterpret_cast<__gm__ float*>(output);
        outF[0] = val;
        outF[1] = 0.0f;
    }
}

// ===========================================================================
// SIMT VF Kernel
// Template parameters:
//   T      - input data type
//   AccT   - accumulator type (float or int64_t)
//   OutT   - output data type
//   IdxT   - index type for loop variables (int32_t or int64_t) [R003]
//   NumThreads - thread block size for __launch_bounds__ [R006]
// ===========================================================================
template <typename T, typename AccT, typename OutT, typename IdxT, uint32_t NumThreads>
__simt_vf__ __aicore__ __launch_bounds__(NumThreads) inline void OpTraceSimtKernel(
    int64_t diagSize, int64_t diagStride,
    __gm__ T* input, __gm__ OutT* output, __ubuf__ int64_t* warpBuf)
{
    constexpr uint32_t numWarps = NumThreads / WARP_SIZE;

    const uint32_t tid = threadIdx.x;
    const uint32_t laneId = tid % WARP_SIZE;
    const uint32_t warpId = tid / WARP_SIZE;

    if (diagSize == 0) {
        if (tid == 0) {
            if constexpr (std::is_same_v<AccT, int64_t>) {
                output[0] = static_cast<OutT>(int64_t(0));
            } else {
                WriteFloatResult<T, OutT>(0.0f, output);
            }
        }
        return;
    }

    if constexpr (std::is_same_v<AccT, float>) {
        // Path 1: float accumulator (half, float, bfloat16 inputs)
        float localSum = 0.0f;
        for (IdxT i = static_cast<IdxT>(tid); i < static_cast<IdxT>(diagSize);
             i += static_cast<IdxT>(NumThreads)) {
            localSum += LoadAsFloat<T>(input, static_cast<int64_t>(i) * diagStride);
        }
        float warpSum = FloatManualReduce(localSum);
        __ubuf__ float* wfBuf = reinterpret_cast<__ubuf__ float*>(warpBuf);
        if (laneId == 0) { wfBuf[warpId] = warpSum; }
        asc_syncthreads();
        if (warpId == 0) {
            float fs = (laneId < numWarps) ? wfBuf[laneId] : 0.0f;
            fs = FloatManualReduce(fs);
            if (laneId == 0) { WriteFloatResult<T, OutT>(fs, output); }
        }
    } else if constexpr (std::is_same_v<AccT, int64_t>) {
        // Path 2: int64 accumulator (integer/bool inputs)
        int64_t localSum = 0;
        for (IdxT i = static_cast<IdxT>(tid); i < static_cast<IdxT>(diagSize);
             i += static_cast<IdxT>(NumThreads)) {
            localSum += LoadAsInt64<T>(input, static_cast<int64_t>(i) * diagStride);
        }
        int64_t warpSum = Int64ManualReduce(localSum);
        if (laneId == 0) { warpBuf[warpId] = warpSum; }
        asc_syncthreads();
        if (warpId == 0) {
            int64_t fs = (laneId < numWarps) ? warpBuf[laneId] : int64_t(0);
            fs = Int64ManualReduce(fs);
            if (laneId == 0) { output[0] = static_cast<OutT>(fs); }
        }
    } else {
        // Path 3: complex64 (accumulate real/imag as float)
        // Note: complex128 is not supported on Ascend950 (no double precision in aicore)
        float localReal = 0.0f;
        float localImag = 0.0f;
        __gm__ float* inputF = reinterpret_cast<__gm__ float*>(input);
        for (IdxT i = static_cast<IdxT>(tid); i < static_cast<IdxT>(diagSize);
             i += static_cast<IdxT>(NumThreads)) {
            int64_t base = static_cast<int64_t>(i) * diagStride * 2;
            localReal += inputF[base];
            localImag += inputF[base + 1];
        }
        float warpReal = FloatManualReduce(localReal);
        float warpImag = FloatManualReduce(localImag);
        __ubuf__ float* wfBuf = reinterpret_cast<__ubuf__ float*>(warpBuf);
        if (laneId == 0) {
            wfBuf[warpId * 2] = warpReal;
            wfBuf[warpId * 2 + 1] = warpImag;
        }
        asc_syncthreads();
        if (warpId == 0) {
            float fr = (laneId < numWarps) ? wfBuf[laneId * 2] : 0.0f;
            float fi = (laneId < numWarps) ? wfBuf[laneId * 2 + 1] : 0.0f;
            fr = FloatManualReduce(fr);
            fi = FloatManualReduce(fi);
            if (laneId == 0) {
                __gm__ float* outputF = reinterpret_cast<__gm__ float*>(output);
                outputF[0] = fr;
                outputF[1] = fi;
            }
        }
    }
}

// ===========================================================================
// Process: entry point called from apt.cpp
// R003: Dispatch int32_t vs int64_t indexing based on problem size
// R006: Dispatch 512-thread vs 256-thread kernel based on accumulator type
// ===========================================================================
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TraceTilingData* tilingData)
{
    using AccT = typename AccType<T>::type;
    using OutT = typename OutputType<T>::type;

    __gm__ T* xGm = (__gm__ T*)x;
    __gm__ OutT* yGm = (__gm__ OutT*)y;
    int64_t diagSize = tilingData->diagSize;
    int64_t diagStride = tilingData->diagStride;

    // Allocate UB for warp reduction buffer (max 16 warps * 2 slots for complex)
    constexpr uint32_t maxNumWarps = THREAD_NUM_HIGH / WARP_SIZE;  // 16
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<int64_t> ub = ubAlloc.Alloc<int64_t>(maxNumWarps * 2);
    DataSyncBarrier<MemDsbT::UB>();
    __ubuf__ int64_t* warpBuf = (__ubuf__ int64_t*)(ub.GetPhyAddr());

    // R003: Check if int32_t indexing is safe (avoids 64-bit arithmetic in VF)
    bool useInt32 = (diagStride > 0) &&
                    (diagSize <= static_cast<int64_t>(INT32_MAX / diagStride));

    if (useInt32) {
        // R006: float/double/complex paths use 512 threads; int64 path uses 256 threads
        if constexpr (std::is_same_v<AccT, float>) {
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int32_t, THREAD_NUM_HIGH>>(
                dim3(THREAD_NUM_HIGH), diagSize, diagStride, xGm, yGm, warpBuf);
        } else if constexpr (std::is_same_v<AccT, int64_t>) {
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int32_t, THREAD_NUM_LOW>>(
                dim3(THREAD_NUM_LOW), diagSize, diagStride, xGm, yGm, warpBuf);
        } else {
            // double/complex path (512 threads, low register pressure)
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int32_t, THREAD_NUM_HIGH>>(
                dim3(THREAD_NUM_HIGH), diagSize, diagStride, xGm, yGm, warpBuf);
        }
    } else {
        // Fallback: int64_t indexing for very large matrices
        if constexpr (std::is_same_v<AccT, float>) {
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int64_t, THREAD_NUM_HIGH>>(
                dim3(THREAD_NUM_HIGH), diagSize, diagStride, xGm, yGm, warpBuf);
        } else if constexpr (std::is_same_v<AccT, int64_t>) {
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int64_t, THREAD_NUM_LOW>>(
                dim3(THREAD_NUM_LOW), diagSize, diagStride, xGm, yGm, warpBuf);
        } else {
            // double/complex path (512 threads)
            asc_vf_call<OpTraceSimtKernel<T, AccT, OutT, int64_t, THREAD_NUM_HIGH>>(
                dim3(THREAD_NUM_HIGH), diagSize, diagStride, xGm, yGm, warpBuf);
        }
    }
}

}  // namespace NsTrace

#endif  // TRACE_SIMT_H_
