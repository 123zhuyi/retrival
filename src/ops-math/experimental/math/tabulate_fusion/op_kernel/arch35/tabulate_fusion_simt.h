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
 * \file tabulate_fusion_simt.h
 * \brief SIMT kernel for tabulate_fusion: table lookup + 5th-order polynomial interpolation
 *
 * Parallelism: (nloc, lastLayerSize) 2D work units, Grid-Stride traversal, 512 threads.
 *
 * Precision strategy (TTK v5, aligned with float64 golden):
 *   - Horner 多项式求值: 纯 float32 中间计算 (输入 half→float, 求值后 float).
 *     float32 精度足以在 1e-3 容忍度下对齐 golden 的 float64 Horner.
 *   - var*ll 乘法与 nnei 累加: 使用符号分离累加 (sign-separated accumulation).
 *     原因: 当 em=boundary (±3.4028235e+38, float32 max) 时, var*ll 可达 ~1e77,
 *     远超 float32 范围 (3.4e38), 导致 float32 累加出现 Inf+(-Inf)=NaN.
 *     golden 用 float64 累加可正确处理; aicore 不支持 double, 故用符号分离策略:
 *     正负项分别累加, 统计 Inf 数量, 最终按 Inf 数量决定结果符号 (对齐 golden
 *     的 float64→float32 截断行为: 当正项占优时 golden 亦输出 +Inf).
 *   - 输出转换: float → T (half/float), 与 golden 的 res.astype(out_dtype) 一致.
 *
 * TTK v3 修复 (保留):
 *   1. tableIdx 越界保护 (对齐 golden 的 if table_idx >= table.shape[0] 钳位);
 *   2. 使用 SafeFloorToInt (floorf + NaN/Inf 保护) 对齐 golden 的 int(np.floor(...));
 *   3. SafeFloorToInt 处理除零/NaN/Inf/溢出, 防止 undefined behavior.
 * TTK v5 修复:
 *   1. 移除 AlignToDtype (float16 每步舍入), 改为纯 float32 中间计算 (golden 已升级为 float64);
 *   2. 累加器从 float32 改为符号分离累加, 修复 em=boundary 时 float32 溢出导致 NaN 的问题
 *      (bb_9/bb_97/bb_84/bb_17 四个极低精度用例的根因).
 */

#ifndef TABULATE_FUSION_SIMT_H_
#define TABULATE_FUSION_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_fp16.h"       // __half2float / __float2half
#include "simt_api/math_functions.h" // floorf / isfinite / isnan / isinf
#include "tabulate_fusion_tiling_data.h"
#include "tabulate_fusion_tiling_key.h"

namespace NsTabulateFusion {
using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 512;
static constexpr float F32_MAX = 3.4028235e38f;
static constexpr int32_t LL_CHANNELS = 4;                // ll0-ll3 descriptor channels
static constexpr float FLOAT_INT32_MAX = 2147483647.0f;  // largest int32 exactly representable in float32
static constexpr float FLOAT_INT32_MIN = -2147483648.0f; // smallest int32 exactly representable in float32

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
    if (!isfinite(x) || x > FLOAT_INT32_MAX || x < FLOAT_INT32_MIN) {
        return 0;
    }
    return static_cast<int32_t>(floorf(x));
}

// ============================================================================
// Sign-separated overflow-safe accumulation helpers
// 背景: aicore 不支持 double, 当 em=boundary (±F32_MAX) 时 var*ll 溢出为 ±Inf,
//   简单 float32 累加会出现 Inf+(-Inf)=NaN. 符号分离策略:
//   - 正负项分别累加 (posSum/negSum), 避免 Inf+(-Inf)=NaN
//   - 统计溢出项数量 (posInfCnt/negInfCnt), 用于最终结果判定
//   - 最终: 若仅正溢出 → +Inf (对齐 golden float64→float32 截断)
//           若仅负溢出 → -Inf
//           若双侧溢出且数量不等 → 按多数侧 ±Inf
//           若双侧溢出且数量相等 → 返回非溢出残差 (best effort)
// ============================================================================

// Produce +Inf via float32 overflow (F32_MAX * F32_MAX = +Inf in IEEE 754)
__simt_callee__ inline float MakePosInf() { return F32_MAX * F32_MAX; }

// Accumulate one product into sign-separated accumulators
__simt_callee__ inline void AccumSign(float product, float& posSum, float& negSum, int32_t& posInfCnt,
                                      int32_t& negInfCnt)
{
    if (isinf(product)) {
        if (product > 0.0f) {
            posInfCnt++;
        } else {
            negInfCnt++;
        }
    } else if (product >= 0.0f) {
        posSum += product;
        if (isinf(posSum)) {
            posInfCnt++;
            posSum = 0.0f;
        }
    } else {
        negSum += product;
        if (isinf(negSum)) {
            negInfCnt++;
            negSum = 0.0f;
        }
    }
}

// Combine sign-separated accumulators into final result
__simt_callee__ inline float CombineSign(float posSum, float negSum, int32_t posInfCnt, int32_t negInfCnt)
{
    if (posInfCnt > 0 && negInfCnt > 0) {
        if (posInfCnt > negInfCnt) {
            return MakePosInf();
        }
        if (negInfCnt > posInfCnt) {
            return -MakePosInf();
        }
        return posSum + negSum;
    }
    if (posInfCnt > 0) {
        return MakePosInf();
    }
    if (negInfCnt > 0) {
        return -MakePosInf();
    }
    return posSum + negSum;
}

// ============================================================================
// Read table_info 5 params (lower/upper/max/stride0/stride1) from GM, promote to float
// ============================================================================

template <typename T>
__simt_callee__ inline void ReadTableInfo(__gm__ T* tableInfoGm, float& lower, float& upper, float& maxVal,
                                          float& stride0, float& stride1)
{
    lower = ToFloat<T>(tableInfoGm[0]);
    upper = ToFloat<T>(tableInfoGm[1]);
    maxVal = ToFloat<T>(tableInfoGm[2]);
    stride0 = ToFloat<T>(tableInfoGm[3]);
    stride1 = ToFloat<T>(tableInfoGm[4]);
}

// ============================================================================
// Locate_xx: branch logic to compute table_idx and xx_new from xx
// ============================================================================

__simt_callee__ inline void LocateXx(float xx, float lower, float upper, float maxVal, float stride0, float stride1,
                                     int32_t firstStride, int32_t secondStride, int32_t& tableIdx, float& xxNew)
{
    tableIdx = 0;
    xxNew = 0.0f;
    if (xx >= lower && xx < upper) {
        tableIdx = SafeFloorToInt((xx - lower) / stride0);
        xxNew = xx - (static_cast<float>(tableIdx) * stride0 + lower);
    } else if (xx >= upper && xx < maxVal) {
        int32_t localIdx = SafeFloorToInt((xx - upper) / stride1);
        tableIdx = firstStride + localIdx;
        xxNew = xx - (static_cast<float>(localIdx) * stride1 + upper);
    } else if (xx >= maxVal) {
        tableIdx = firstStride + secondStride - 1;
        xxNew = 0.0f;
    }
}

// ============================================================================
// Read 6 polynomial coefficients a0~a5 from table row (gather by tableIdx)
// ============================================================================

template <typename T>
__simt_callee__ inline void ReadTableCoeffs(__gm__ T* tableGm, int32_t base, int32_t lastSizeAlign, float& a0,
                                            float& a1, float& a2, float& a3, float& a4, float& a5)
{
    a0 = ToFloat<T>(tableGm[base + 0 * lastSizeAlign]);
    a1 = ToFloat<T>(tableGm[base + 1 * lastSizeAlign]);
    a2 = ToFloat<T>(tableGm[base + 2 * lastSizeAlign]);
    a3 = ToFloat<T>(tableGm[base + 3 * lastSizeAlign]);
    a4 = ToFloat<T>(tableGm[base + 4 * lastSizeAlign]);
    a5 = ToFloat<T>(tableGm[base + 5 * lastSizeAlign]);
}

// ============================================================================
// 5th-order polynomial Horner evaluation (pure float32, no per-step rounding)
// TTK v5: 移除 AlignToDtype, 改为纯 float32. golden 已升级为 float64 中间计算.
// ============================================================================

template <typename T>
__simt_callee__ inline float HornerEval(float xxNew, float a0, float a1, float a2, float a3, float a4, float a5)
{
    float var = a5 * xxNew;
    var = a4 + var;
    var = var * xxNew;
    var = a3 + var;
    var = var * xxNew;
    var = a2 + var;
    var = var * xxNew;
    var = a1 + var;
    var = var * xxNew;
    var = a0 + var;
    return var;
}

// ============================================================================
// Read 4 em coefficients ll0~ll3 from GM, promote to float
// ============================================================================

template <typename T>
__simt_callee__ inline void ReadEmCoeffs(__gm__ T* emGm, int32_t emBase, float& ll0, float& ll1, float& ll2, float& ll3)
{
    ll0 = ToFloat<T>(emGm[emBase + 0]);
    ll1 = ToFloat<T>(emGm[emBase + 1]);
    ll2 = ToFloat<T>(emGm[emBase + 2]);
    ll3 = ToFloat<T>(emGm[emBase + 3]);
}

// ============================================================================
// Main SIMT VF kernel: Grid-Stride over (nloc, lastLayerSize) work units
// ============================================================================

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpTabulateFusionSimt(
    int32_t nloc, int32_t nnei, int32_t lastLayerSize, int32_t lastSizeAlign, int32_t tableRowSize, int32_t tableRows,
    __gm__ T* tableInfoGm, __gm__ T* tableGm, __gm__ T* emXGm, __gm__ T* emGm, __gm__ T* descriptorGm)
{
    float lower = 0.0f, upper = 0.0f, maxVal = 0.0f, stride0 = 0.0f, stride1 = 0.0f;
    ReadTableInfo<T>(tableInfoGm, lower, upper, maxVal, stride0, stride1);
    int32_t firstStride = SafeFloorToInt((upper - lower) / stride0);
    int32_t secondStride = SafeFloorToInt((maxVal - upper) / stride1);

    int32_t totalWork = nloc * lastLayerSize;
    int32_t stride = static_cast<int32_t>(blockDim.x * gridDim.x);
    for (int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x); idx < totalWork; idx += stride) {
        int32_t nlocI = idx / lastLayerSize;
        int32_t j = idx % lastLayerSize;
        // TTK v5: 符号分离累加器, 防止 em=boundary 时 float32 溢出导致 NaN
        float pS0 = 0.0f, nS0 = 0.0f, pS1 = 0.0f, nS1 = 0.0f;
        float pS2 = 0.0f, nS2 = 0.0f, pS3 = 0.0f, nS3 = 0.0f;
        int32_t pI0 = 0, nI0 = 0, pI1 = 0, nI1 = 0, pI2 = 0, nI2 = 0, pI3 = 0, nI3 = 0;

        for (int32_t nneiJ = 0; nneiJ < nnei; nneiJ++) {
            float xx = ToFloat<T>(emXGm[nlocI * nnei + nneiJ]);
            int32_t tableIdx = 0;
            float xxNew = 0.0f;
            LocateXx(xx, lower, upper, maxVal, stride0, stride1, firstStride, secondStride, tableIdx, xxNew);
            if (tableIdx < 0) {
                tableIdx = 0;
            } else if (tableIdx >= tableRows) {
                tableIdx = tableRows - 1;
            }
            int32_t base = tableIdx * tableRowSize + j;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f;
            ReadTableCoeffs<T>(tableGm, base, lastSizeAlign, a0, a1, a2, a3, a4, a5);
            float var = HornerEval<T>(xxNew, a0, a1, a2, a3, a4, a5);
            int32_t emBase = (nlocI * nnei + nneiJ) * LL_CHANNELS;
            float ll0 = 0.0f, ll1 = 0.0f, ll2 = 0.0f, ll3 = 0.0f;
            ReadEmCoeffs<T>(emGm, emBase, ll0, ll1, ll2, ll3);
            AccumSign(var * ll0, pS0, nS0, pI0, nI0);
            AccumSign(var * ll1, pS1, nS1, pI1, nI1);
            AccumSign(var * ll2, pS2, nS2, pI2, nI2);
            AccumSign(var * ll3, pS3, nS3, pI3, nI3);
        }

        int32_t outBase = nlocI * LL_CHANNELS * lastLayerSize + j;
        descriptorGm[outBase + 0 * lastLayerSize] = FromFloat<T>(CombineSign(pS0, nS0, pI0, nI0));
        descriptorGm[outBase + 1 * lastLayerSize] = FromFloat<T>(CombineSign(pS1, nS1, pI1, nI1));
        descriptorGm[outBase + 2 * lastLayerSize] = FromFloat<T>(CombineSign(pS2, nS2, pI2, nI2));
        descriptorGm[outBase + 3 * lastLayerSize] = FromFloat<T>(CombineSign(pS3, nS3, pI3, nI3));
    }
}

// ============================================================================
// Process: free-function dispatcher, launches VF via asc_vf_call
// ============================================================================

template <typename T>
__aicore__ inline void Process(GM_ADDR table, GM_ADDR tableInfo, GM_ADDR emX, GM_ADDR em, GM_ADDR descriptor,
                               const TabulateFusionTilingData* tilingData)
{
    __gm__ T* tableGm = (__gm__ T*)table;
    __gm__ T* tableInfoGm = (__gm__ T*)tableInfo;
    __gm__ T* emXGm = (__gm__ T*)emX;
    __gm__ T* emGm = (__gm__ T*)em;
    __gm__ T* descriptorGm = (__gm__ T*)descriptor;

    asc_vf_call<OpTabulateFusionSimt<T>>(dim3(THREAD_NUM), tilingData->nloc, tilingData->nnei,
                                         tilingData->lastLayerSize, tilingData->lastSizeAlign, tilingData->tableRowSize,
                                         tilingData->tableRows, tableInfoGm, tableGm, emXGm, emGm, descriptorGm);
}

} // namespace NsTabulateFusion
#endif // TABULATE_FUSION_SIMT_H_
