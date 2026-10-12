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
 * \file tabulate_fusion_grad_simt.h
 * \brief SIMT kernel for tabulate_fusion_grad: reverse gradient of 5th-order polynomial
 *
 * Parallelism: nloc 维核间切分 + Grid-Stride on loc 核内, 512 threads.
 * Each thread serially processes all nnei neighbors of one loc (including
 * size-dim serial accumulation and last-neighbor集中累加).
 *
 * Precision strategy (aligned with TBE kernel & golden):
 *   - Horner 多项式求值 (res & grad), 纯 float32 中间计算.
 *   - 串行累加 (无 atomicAdd), 保证确定性.
 *   - floorf (IEEE 754 向下取整), 与 TBE vconv floor 一致.
 *   - 末邻居重复值识别 + 集中累加, 与 TBE vcmpv_eq+countbit1 数学等价.
 */

#ifndef TABULATE_FUSION_GRAD_SIMT_H_
#define TABULATE_FUSION_GRAD_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_fp16.h"       // __half2float / __float2half
#include "simt_api/math_functions.h" // floorf / isfinite
#include "tabulate_fusion_grad_tiling_data.h"
#include "tabulate_fusion_grad_tiling_key.h"

namespace NsTabulateFusionGrad {
using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 512;

// ============================================================================
// dtype conversion helpers (promote half->float at GM boundary)
// ============================================================================

template <typename T>
__simt_callee__ inline float ToFloat(T x)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else {
        return static_cast<float>(x);
    }
}

template <typename T>
__simt_callee__ inline T FromFloat(float x)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(x);
    } else {
        return static_cast<T>(x);
    }
}

// ============================================================================
// SafeFloorToInt: float -> int32_t via floorf, with NaN/Inf/overflow protection
// ============================================================================

__simt_callee__ inline int32_t SafeFloorToInt(float x)
{
    if (!isfinite(x) || x > 2147483647.0f || x < -2147483648.0f) {
        return 0;
    }
    return static_cast<int32_t>(floorf(x));
}

// ============================================================================
// Read table_info 5 params (lower/upper/max/stride0/stride1) from GM
// ============================================================================

template <typename T>
__simt_callee__ inline void ReadTableInfo(__gm__ T* gm, float& lower, float& upper, float& maxVal, float& stride0,
                                          float& stride1)
{
    lower = ToFloat<T>(gm[0]);
    upper = ToFloat<T>(gm[1]);
    maxVal = ToFloat<T>(gm[2]);
    stride0 = ToFloat<T>(gm[3]);
    stride1 = ToFloat<T>(gm[4]);
}

// ============================================================================
// LocateXx: three-segment branch logic to compute tableIdx and xLocal
//   x >= max       -> tableIdx = maxTblIdx, xLocal = 0
//   x >= upper     -> tableIdx = firstStride + floor((x-upper)/stride1)
//   x >= lower     -> tableIdx = floor((x-lower)/stride0)
//   x < lower      -> tableIdx = 0, xLocal = x - lower (negative, extrapolate)
// ============================================================================

__simt_callee__ inline void LocateXx(float x, float lower, float upper, float maxVal, float stride0, float stride1,
                                     int32_t firstStride, int32_t maxTblIdx, int32_t tableDim0, int32_t& tableIdx,
                                     float& xLocal)
{
    tableIdx = 0;
    xLocal = 0.0f;
    if (x >= maxVal) {
        tableIdx = maxTblIdx;
        xLocal = 0.0f;
    } else if (x >= upper) {
        int32_t step = SafeFloorToInt((x - upper) / stride1);
        tableIdx = firstStride + step;
        xLocal = (x - upper) - static_cast<float>(step) * stride1;
    } else if (x >= lower) {
        int32_t step = SafeFloorToInt((x - lower) / stride0);
        tableIdx = step;
        xLocal = (x - lower) - static_cast<float>(step) * stride0;
    } else {
        tableIdx = 0;
        xLocal = x - lower;
    }
    // 边界保护
    if (tableIdx < 0) {
        tableIdx = 0;
    }
    if (tableIdx > tableDim0 - 1) {
        tableIdx = tableDim0 - 1;
    }
}

// ============================================================================
// Read 4 em coefficients c0~c3 from GM
// ============================================================================

template <typename T>
__simt_callee__ inline void ReadEmCoeffs(__gm__ T* gm, int32_t base, float& c0, float& c1, float& c2, float& c3)
{
    c0 = ToFloat<T>(gm[base + 0]);
    c1 = ToFloat<T>(gm[base + 1]);
    c2 = ToFloat<T>(gm[base + 2]);
    c3 = ToFloat<T>(gm[base + 3]);
}

// ============================================================================
// HornerRes: 5th-order polynomial value (forward)
//   res = ((((a5*x + a4)*x + a3)*x + a2)*x + a1)*x + a0
// ============================================================================

__simt_callee__ inline float HornerRes(float x, float a0, float a1, float a2, float a3, float a4, float a5)
{
    float v = a5 * x;
    v = a4 + v;
    v = v * x;
    v = a3 + v;
    v = v * x;
    v = a2 + v;
    v = v * x;
    v = a1 + v;
    v = v * x;
    v = a0 + v;
    return v;
}

// ============================================================================
// HornerGrad: 5th-order polynomial derivative (reverse gradient)
//   grad = ((((5*a5*x + 4*a4)*x + 3*a3)*x + 2*a2)*x + a1)
// ============================================================================

__simt_callee__ inline float HornerGrad(float x, float a1, float a2, float a3, float a4, float a5)
{
    float v = 5.0f * a5 * x;
    v = 4.0f * a4 + v;
    v = v * x;
    v = 3.0f * a3 + v;
    v = v * x;
    v = 2.0f * a2 + v;
    v = v * x;
    v = a1 + v;
    return v;
}

// ============================================================================
// CountLastRepeat: scan from tail to count consecutive repeated last-neighbor values
//   Returns countLast (>=1). If all nnei values equal, returns nnei.
// ============================================================================

template <typename T>
__simt_callee__ inline int32_t CountLastRepeat(__gm__ T* emXGm, int32_t loc, int32_t nnei)
{
    if (nnei <= 1) {
        return 1;
    }
    int32_t base = loc * nnei;
    float lastVal = ToFloat<T>(emXGm[base + nnei - 1]);
    int32_t cnt = 0;
    for (int32_t k = nnei - 1; k >= 0; k--) {
        if (ToFloat<T>(emXGm[base + k]) == lastVal) {
            cnt++;
        } else {
            break;
        }
    }
    return cnt;
}

// ============================================================================
// AccumulateOneNei: size-dim serial accumulation for one neighbor
//   Loads 6 coeffs a0~a5 per size step, Horner eval res & grad,
//   accumulates dy_dem[c] += res * dy[c] and dy_dem_x += grad * (em·dy)
// ============================================================================

template <typename T>
__simt_callee__ inline void AccumulateOneNei(__gm__ T* tableGm, __gm__ T* dyGm, int32_t coefBase, int32_t lastLayerSize,
                                             int32_t sizeAlign64, int32_t loc, float xLocal, float emC0, float emC1,
                                             float emC2, float emC3, float& dyDemC0, float& dyDemC1, float& dyDemC2,
                                             float& dyDemC3, float& dyDemX)
{
    for (int32_t s = 0; s < lastLayerSize; s++) {
        float a0 = ToFloat<T>(tableGm[coefBase + 0 * sizeAlign64 + s]);
        float a1 = ToFloat<T>(tableGm[coefBase + 1 * sizeAlign64 + s]);
        float a2 = ToFloat<T>(tableGm[coefBase + 2 * sizeAlign64 + s]);
        float a3 = ToFloat<T>(tableGm[coefBase + 3 * sizeAlign64 + s]);
        float a4 = ToFloat<T>(tableGm[coefBase + 4 * sizeAlign64 + s]);
        float a5 = ToFloat<T>(tableGm[coefBase + 5 * sizeAlign64 + s]);

        float res = HornerRes(xLocal, a0, a1, a2, a3, a4, a5);
        float grad = HornerGrad(xLocal, a1, a2, a3, a4, a5);

        float dy0 = ToFloat<T>(dyGm[((int64_t)loc * 4 + 0) * lastLayerSize + s]);
        float dy1 = ToFloat<T>(dyGm[((int64_t)loc * 4 + 1) * lastLayerSize + s]);
        float dy2 = ToFloat<T>(dyGm[((int64_t)loc * 4 + 2) * lastLayerSize + s]);
        float dy3 = ToFloat<T>(dyGm[((int64_t)loc * 4 + 3) * lastLayerSize + s]);

        dyDemC0 += res * dy0;
        dyDemC1 += res * dy1;
        dyDemC2 += res * dy2;
        dyDemC3 += res * dy3;

        float emDyDot = emC0 * dy0 + emC1 * dy1 + emC2 * dy2 + emC3 * dy3;
        dyDemX += grad * emDyDot;
    }
}

// ============================================================================
// ApplyRepeatScaling: last-neighbor集中累加 logic
//   If countLast > 1 and current nei is in the repeated tail:
//     - last nei (nei == nnei-1): multiply gradients by countLast
//     - middle repeated nei: zero out (accumulated集中 at last nei)
// ============================================================================

__simt_callee__ inline void ApplyRepeatScaling(int32_t nei, int32_t nnei, int32_t countLast, float& dyDemC0,
                                               float& dyDemC1, float& dyDemC2, float& dyDemC3, float& dyDemX)
{
    bool isRepeatNei = (countLast > 1) && (nei >= nnei - countLast);
    if (!isRepeatNei) {
        return;
    }
    if (nei == nnei - 1) {
        float scale = static_cast<float>(countLast);
        dyDemC0 *= scale;
        dyDemC1 *= scale;
        dyDemC2 *= scale;
        dyDemC3 *= scale;
        dyDemX *= scale;
    } else {
        dyDemC0 = 0.0f;
        dyDemC1 = 0.0f;
        dyDemC2 = 0.0f;
        dyDemC3 = 0.0f;
        dyDemX = 0.0f;
    }
}

// ============================================================================
// WriteBackOneNei: write 4 dy_dem components and 1 dy_dem_x to GM
// ============================================================================

template <typename T>
__simt_callee__ inline void WriteBackOneNei(__gm__ T* dyDemGm, __gm__ T* dyDemXGm, int32_t loc, int32_t nnei,
                                            int32_t nei, float dyDemC0, float dyDemC1, float dyDemC2, float dyDemC3,
                                            float dyDemX)
{
    int32_t base = (loc * nnei + nei) * 4;
    dyDemGm[base + 0] = FromFloat<T>(dyDemC0);
    dyDemGm[base + 1] = FromFloat<T>(dyDemC1);
    dyDemGm[base + 2] = FromFloat<T>(dyDemC2);
    dyDemGm[base + 3] = FromFloat<T>(dyDemC3);
    dyDemXGm[loc * nnei + nei] = FromFloat<T>(dyDemX);
}

// ============================================================================
// ProcessOneLoc: serially process all nnei neighbors for one loc
// ============================================================================

template <typename T>
__simt_callee__ inline void ProcessOneLoc(int32_t loc, int32_t nnei, int32_t lastLayerSize, int32_t sizeAlign64,
                                          int32_t tableDim0, float lower, float upper, float maxVal, float stride0,
                                          float stride1, int32_t firstStride, int32_t maxTblIdx, __gm__ T* tableGm,
                                          __gm__ T* emXGm, __gm__ T* emGm, __gm__ T* dyGm, __gm__ T* dyDemXGm,
                                          __gm__ T* dyDemGm)
{
    int32_t countLast = CountLastRepeat<T>(emXGm, loc, nnei);
    for (int32_t nei = 0; nei < nnei; nei++) {
        float x = ToFloat<T>(emXGm[loc * nnei + nei]);
        int32_t tableIdx = 0;
        float xLocal = 0.0f;
        LocateXx(x, lower, upper, maxVal, stride0, stride1, firstStride, maxTblIdx, tableDim0, tableIdx, xLocal);

        int32_t emBase = (loc * nnei + nei) * 4;
        float emC0 = 0.0f, emC1 = 0.0f, emC2 = 0.0f, emC3 = 0.0f;
        ReadEmCoeffs<T>(emGm, emBase, emC0, emC1, emC2, emC3);

        float dyDemC0 = 0.0f, dyDemC1 = 0.0f, dyDemC2 = 0.0f, dyDemC3 = 0.0f, dyDemX = 0.0f;
        int32_t coefBase = tableIdx * 6 * sizeAlign64;
        AccumulateOneNei<T>(tableGm, dyGm, coefBase, lastLayerSize, sizeAlign64, loc, xLocal, emC0, emC1, emC2, emC3,
                            dyDemC0, dyDemC1, dyDemC2, dyDemC3, dyDemX);

        ApplyRepeatScaling(nei, nnei, countLast, dyDemC0, dyDemC1, dyDemC2, dyDemC3, dyDemX);
        WriteBackOneNei<T>(dyDemGm, dyDemXGm, loc, nnei, nei, dyDemC0, dyDemC1, dyDemC2, dyDemC3, dyDemX);
    }
}

// ============================================================================
// Main SIMT VF kernel: Grid-Stride over loc
// ============================================================================

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpTabulateFusionGradSimtKernel(
    int32_t perCoreNloc, int32_t nloc, int32_t nnei, int32_t lastLayerSize, int32_t sizeAlign64, int32_t tableDim0,
    int32_t locStartOffset, __gm__ T* tableGm, __gm__ T* tableInfoGm, __gm__ T* emXGm, __gm__ T* emGm, __gm__ T* dyGm,
    __gm__ T* dyDemXGm, __gm__ T* dyDemGm)
{
    float lower = 0.0f, upper = 0.0f, maxVal = 0.0f, stride0 = 0.0f, stride1 = 0.0f;
    ReadTableInfo<T>(tableInfoGm, lower, upper, maxVal, stride0, stride1);
    int32_t firstStride = SafeFloorToInt((upper - lower) / stride0);
    int32_t maxTblIdx = firstStride + SafeFloorToInt((maxVal - upper) / stride1) - 1;

    int32_t locStart = locStartOffset + static_cast<int32_t>(blockIdx.x) * perCoreNloc;
    int32_t locEnd = locStart + perCoreNloc;
    if (locEnd > nloc) {
        locEnd = nloc;
    }

    int32_t threadStride = static_cast<int32_t>(blockDim.x);
    for (int32_t loc = locStart + static_cast<int32_t>(threadIdx.x); loc < locEnd; loc += threadStride) {
        ProcessOneLoc<T>(loc, nnei, lastLayerSize, sizeAlign64, tableDim0, lower, upper, maxVal, stride0, stride1,
                         firstStride, maxTblIdx, tableGm, emXGm, emGm, dyGm, dyDemXGm, dyDemGm);
    }
}

// ============================================================================
// Process: free-function dispatcher, launches VF via asc_vf_call
// ============================================================================

template <typename T>
__aicore__ inline void Process(GM_ADDR table, GM_ADDR table_info, GM_ADDR em_x, GM_ADDR em, GM_ADDR dy,
                               GM_ADDR descriptor, GM_ADDR dy_dem_x, GM_ADDR dy_dem, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)descriptor; // descriptor 仅用于 infershape，kernel 内不读取
    (void)workspace;  // 无用户 workspace，仅系统 workspace
    GET_TILING_DATA_WITH_STRUCT(TabulateFusionGradTilingData, tilingData, tiling);
    __gm__ T* tableGm = (__gm__ T*)table;
    __gm__ T* tableInfoGm = (__gm__ T*)table_info;
    __gm__ T* emXGm = (__gm__ T*)em_x;
    __gm__ T* emGm = (__gm__ T*)em;
    __gm__ T* dyGm = (__gm__ T*)dy;
    __gm__ T* dyDemXGm = (__gm__ T*)dy_dem_x;
    __gm__ T* dyDemGm = (__gm__ T*)dy_dem;

    asc_vf_call<OpTabulateFusionGradSimtKernel<T>>(dim3(THREAD_NUM), tilingData.perCoreNloc, tilingData.nloc,
                                                   tilingData.nnei, tilingData.lastLayerSize, tilingData.sizeAlign64,
                                                   tilingData.tableDim0, tilingData.locStartOffset, tableGm,
                                                   tableInfoGm, emXGm, emGm, dyGm, dyDemXGm, dyDemGm);
}

} // namespace NsTabulateFusionGrad
#endif // TABULATE_FUSION_GRAD_SIMT_H_
