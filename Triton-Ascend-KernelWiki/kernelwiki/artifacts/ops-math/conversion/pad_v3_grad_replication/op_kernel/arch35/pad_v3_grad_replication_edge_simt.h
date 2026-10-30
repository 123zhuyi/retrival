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
 * \file pad_v3_grad_replication_edge_simt.h
 * \brief pad_v3_grad_replication tail-axis SIMT kernel (adapted from pad_v3_grad edge_simt)
 *
 * 切尾轴（SPLIT_AXIS == DIM_NUM-1）时使用纯 SIMT 方式：每个 thread 独立遍历
 * grad_input 位置，计算其对应的 grad_output 范围，直接从 GM 累加后写回。
 * 不依赖 UB buffer 做矩形区域搬移，适合非尾轴有 padding 的多维场景。
 */

#ifndef PAD_V3_GRAD_REPLICATION_EDGE_SIMT_H
#define PAD_V3_GRAD_REPLICATION_EDGE_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "pad_v3_grad_replication_struct.h"

constexpr int32_t REP_EDGE_THREAD_DIM = 2048;
constexpr int32_t REP_EDGE_HALF_THREAD_DIM = 1024;
constexpr int32_t REP_EDGE_QUARTER_THREAD_DIM = 512;
constexpr int32_t REP_EDGE_EIGHTH_THREAD_DIM = 256;
constexpr int32_t REP_EDGE_SIXTEENTH_THREAD_DIM = 128;
constexpr int32_t REP_EDGE_THIRTY2ND_THREAD_DIM = 64;

namespace PadV3GradReplication {
using namespace AscendC;

// ---- helper: flat index → multi-dim input position ----
template <uint8_t DIM_NUM, typename U, typename GmOffsetType>
__simt_callee__ __aicore__ void CalPos(GmOffsetType yIdx, U* outIndex, __ubuf__ U* origStrides,
                                       __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (uint8_t i = 0; i < DIM_NUM - 1; i++) {
        outIndex[i] = static_cast<U>(Simt::UintDiv(yIdx, magics[i], shifts[i]));
        yIdx -= outIndex[i] * origStrides[i];
    }
    outIndex[DIM_NUM - 1] = static_cast<U>(yIdx);
}

// ---- helper: per-dimension padded-output scope ----
template <uint8_t DIM, typename U>
__simt_callee__ __aicore__ void CalScope(U (*scope)[2], U* outIndex, __ubuf__ U* leftPads, __ubuf__ U* rightPads,
                                         __ubuf__ U* inShapes, __ubuf__ U* outShapes)
{
    int8_t flag = 0;
    for (uint8_t i = 0; i < DIM; i++) {
        flag = 0;
        U paddedPos = outIndex[i] + leftPads[i];
        if (paddedPos >= inShapes[i]) {
            flag = 1;
        }
        // ---- scope start (padded coordinates) ----
        if (outIndex[i] == 0) {
            scope[i][0] = 0;
        } else {
            scope[i][0] = paddedPos + static_cast<U>(flag);
        }
        // ---- scope end (padded coordinates) ----
        if (outIndex[i] == outShapes[i] - 1) {
            scope[i][1] = paddedPos + rightPads[i];
        } else {
            scope[i][1] = paddedPos - static_cast<U>(flag);
        }
    }
}

// ---- helper: write accumulated result to grad_input GM ----
template <typename T, typename CastType, typename GmOffsetType>
__simt_callee__ __aicore__ void CopyOut(GmOffsetType idx, __gm__ volatile T* outputGM, CastType total)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        outputGM[idx] = __float2bfloat16_rn_sat(total);
    } else if constexpr (std::is_same_v<T, float16_t>) {
        outputGM[idx] = __float2half_rn_sat(total);
    } else {
        outputGM[idx] = static_cast<T>(total);
    }
}

// =====================================================================
// Per-dimension SIMT vector functions
// =====================================================================

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimOne(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                               __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            GmOffsetType paddedOff = static_cast<GmOffsetType>(a0);
            CastType tmpVal;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                tmpVal = __bfloat162float(gradOutGM[paddedOff]);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                tmpVal = __half2float(gradOutGM[paddedOff]);
            } else {
                tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
            }
            total += tmpVal;
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimTwo(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                               __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                GmOffsetType paddedOff = static_cast<GmOffsetType>(static_cast<GmOffsetType>(a0) *
                                                                       static_cast<GmOffsetType>(paddedStrides[0]) +
                                                                   static_cast<GmOffsetType>(a1));
                CastType tmpVal;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    tmpVal = __half2float(gradOutGM[paddedOff]);
                } else {
                    tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                }
                total += tmpVal;
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_QUARTER_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimThree(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                                 uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                                 __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                                 __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    GmOffsetType paddedOff = static_cast<GmOffsetType>(
                        static_cast<GmOffsetType>(a0) * static_cast<GmOffsetType>(paddedStrides[0]) +
                        static_cast<GmOffsetType>(a1) * static_cast<GmOffsetType>(paddedStrides[1]) +
                        static_cast<GmOffsetType>(a2));
                    CastType tmpVal;
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        tmpVal = __half2float(gradOutGM[paddedOff]);
                    } else {
                        tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                    }
                    total += tmpVal;
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimFour(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                                __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    for (U a3 = scope[3][0]; a3 <= scope[3][1]; ++a3) {
                        GmOffsetType paddedOff = static_cast<GmOffsetType>(
                            static_cast<GmOffsetType>(a0) * static_cast<GmOffsetType>(paddedStrides[0]) +
                            static_cast<GmOffsetType>(a1) * static_cast<GmOffsetType>(paddedStrides[1]) +
                            static_cast<GmOffsetType>(a2) * static_cast<GmOffsetType>(paddedStrides[2]) +
                            static_cast<GmOffsetType>(a3));
                        CastType tmpVal;
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                        } else if constexpr (std::is_same_v<T, float16_t>) {
                            tmpVal = __half2float(gradOutGM[paddedOff]);
                        } else {
                            tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                        }
                        total += tmpVal;
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimFive(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                                __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    for (U a3 = scope[3][0]; a3 <= scope[3][1]; ++a3) {
                        for (U a4 = scope[4][0]; a4 <= scope[4][1]; ++a4) {
                            GmOffsetType paddedOff = static_cast<GmOffsetType>(
                                static_cast<GmOffsetType>(a0) * static_cast<GmOffsetType>(paddedStrides[0]) +
                                static_cast<GmOffsetType>(a1) * static_cast<GmOffsetType>(paddedStrides[1]) +
                                static_cast<GmOffsetType>(a2) * static_cast<GmOffsetType>(paddedStrides[2]) +
                                static_cast<GmOffsetType>(a3) * static_cast<GmOffsetType>(paddedStrides[3]) +
                                static_cast<GmOffsetType>(a4));
                            CastType tmpVal;
                            if constexpr (std::is_same_v<T, bfloat16_t>) {
                                tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                            } else if constexpr (std::is_same_v<T, float16_t>) {
                                tmpVal = __half2float(gradOutGM[paddedOff]);
                            } else {
                                tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                            }
                            total += tmpVal;
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

// ---- dims 6-8: 线程数进一步缩减以控制寄存器压力 ----
// 前端轴无 padding，scope 恒为单点，仅是多了外层循环。

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_SIXTEENTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimSix(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                               __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    for (U a3 = scope[3][0]; a3 <= scope[3][1]; ++a3) {
                        for (U a4 = scope[4][0]; a4 <= scope[4][1]; ++a4) {
                            for (U a5 = scope[5][0]; a5 <= scope[5][1]; ++a5) {
                                GmOffsetType paddedOff = static_cast<GmOffsetType>(
                                    static_cast<GmOffsetType>(a0) * static_cast<GmOffsetType>(paddedStrides[0]) +
                                    static_cast<GmOffsetType>(a1) * static_cast<GmOffsetType>(paddedStrides[1]) +
                                    static_cast<GmOffsetType>(a2) * static_cast<GmOffsetType>(paddedStrides[2]) +
                                    static_cast<GmOffsetType>(a3) * static_cast<GmOffsetType>(paddedStrides[3]) +
                                    static_cast<GmOffsetType>(a4) * static_cast<GmOffsetType>(paddedStrides[4]) +
                                    static_cast<GmOffsetType>(a5));
                                CastType tmpVal;
                                if constexpr (std::is_same_v<T, bfloat16_t>) {
                                    tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                                } else if constexpr (std::is_same_v<T, float16_t>) {
                                    tmpVal = __half2float(gradOutGM[paddedOff]);
                                } else {
                                    tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                                }
                                total += tmpVal;
                            }
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_SIXTEENTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimSeven(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                                 uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                                 __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                                 __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    for (U a3 = scope[3][0]; a3 <= scope[3][1]; ++a3) {
                        for (U a4 = scope[4][0]; a4 <= scope[4][1]; ++a4) {
                            for (U a5 = scope[5][0]; a5 <= scope[5][1]; ++a5) {
                                for (U a6 = scope[6][0]; a6 <= scope[6][1]; ++a6) {
                                    GmOffsetType paddedOff = static_cast<GmOffsetType>(
                                        static_cast<GmOffsetType>(a0) * static_cast<GmOffsetType>(paddedStrides[0]) +
                                        static_cast<GmOffsetType>(a1) * static_cast<GmOffsetType>(paddedStrides[1]) +
                                        static_cast<GmOffsetType>(a2) * static_cast<GmOffsetType>(paddedStrides[2]) +
                                        static_cast<GmOffsetType>(a3) * static_cast<GmOffsetType>(paddedStrides[3]) +
                                        static_cast<GmOffsetType>(a4) * static_cast<GmOffsetType>(paddedStrides[4]) +
                                        static_cast<GmOffsetType>(a5) * static_cast<GmOffsetType>(paddedStrides[5]) +
                                        static_cast<GmOffsetType>(a6));
                                    CastType tmpVal;
                                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                                        tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                                    } else if constexpr (std::is_same_v<T, float16_t>) {
                                        tmpVal = __half2float(gradOutGM[paddedOff]);
                                    } else {
                                        tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                                    }
                                    total += tmpVal;
                                }
                            }
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

template <typename T, uint8_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(REP_EDGE_THIRTY2ND_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimEight(__gm__ T* gradOutGM, __gm__ volatile T* gradInGM, GmOffsetType inputSize,
                                 uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* paddedShapes, __ubuf__ U* origShapes,
                                 __ubuf__ U* paddedStrides, __ubuf__ U* origStrides, __ubuf__ U* leftPads,
                                 __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < inputSize;
         idx += blockNum * Simt::GetThreadNum()) {
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;
        CalPos<DIM, U, GmOffsetType>(yIdx, outIndex, origStrides, magics, shifts);
        U scope[DIM][2];
        CalScope<DIM, U>(scope, outIndex, leftPads, rightPads, paddedShapes, origShapes);

        CastType total = 0;
        for (U a0 = scope[0][0]; a0 <= scope[0][1]; ++a0) {
            for (U a1 = scope[1][0]; a1 <= scope[1][1]; ++a1) {
                for (U a2 = scope[2][0]; a2 <= scope[2][1]; ++a2) {
                    for (U a3 = scope[3][0]; a3 <= scope[3][1]; ++a3) {
                        for (U a4 = scope[4][0]; a4 <= scope[4][1]; ++a4) {
                            for (U a5 = scope[5][0]; a5 <= scope[5][1]; ++a5) {
                                for (U a6 = scope[6][0]; a6 <= scope[6][1]; ++a6) {
                                    for (U a7 = scope[7][0]; a7 <= scope[7][1]; ++a7) {
                                        GmOffsetType paddedOff = static_cast<GmOffsetType>(
                                            static_cast<GmOffsetType>(a0) *
                                                static_cast<GmOffsetType>(paddedStrides[0]) +
                                            static_cast<GmOffsetType>(a1) *
                                                static_cast<GmOffsetType>(paddedStrides[1]) +
                                            static_cast<GmOffsetType>(a2) *
                                                static_cast<GmOffsetType>(paddedStrides[2]) +
                                            static_cast<GmOffsetType>(a3) *
                                                static_cast<GmOffsetType>(paddedStrides[3]) +
                                            static_cast<GmOffsetType>(a4) *
                                                static_cast<GmOffsetType>(paddedStrides[4]) +
                                            static_cast<GmOffsetType>(a5) *
                                                static_cast<GmOffsetType>(paddedStrides[5]) +
                                            static_cast<GmOffsetType>(a6) *
                                                static_cast<GmOffsetType>(paddedStrides[6]) +
                                            static_cast<GmOffsetType>(a7));
                                        CastType tmpVal;
                                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                                            tmpVal = __bfloat162float(gradOutGM[paddedOff]);
                                        } else if constexpr (std::is_same_v<T, float16_t>) {
                                            tmpVal = __half2float(gradOutGM[paddedOff]);
                                        } else {
                                            tmpVal = static_cast<CastType>(gradOutGM[paddedOff]);
                                        }
                                        total += tmpVal;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, gradInGM, total);
    }
}

// =====================================================================
// Kernel class: tail-axis SIMT 入口
// =====================================================================

template <typename T>
class PadV3GradReplicationEdgeSimt {
public:
    __aicore__ inline PadV3GradReplicationEdgeSimt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradReplicationTilingData* tilingData);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> gradOutGM_; // GM x: grad_output (padded)
    GlobalTensor<T> gradInGM_;  // GM y: grad_input  (original)
    uint32_t mBlockIdx_;
    const PadV3GradReplicationTilingData* mTD_;
};

template <typename T>
__aicore__ inline void PadV3GradReplicationEdgeSimt<T>::Init(GM_ADDR x, GM_ADDR y,
                                                             const PadV3GradReplicationTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    gradOutGM_.SetGlobalBuffer((__gm__ T*)x);
    gradInGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
template <typename U>
__aicore__ inline void PadV3GradReplicationEdgeSimt<T>::Process()
{
    using CastType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                        std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;
    using GmOffsetType = std::conditional_t<std::is_same_v<U, int64_t>, uint64_t, uint32_t>;

    uint32_t blockNum = GetBlockNum();
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t dimNum = mTD_->dimNum;

    // total input elements = ∏ inputShape[k] (grad_input size)
    GmOffsetType inputSize = 1;
    for (uint32_t i = 0; i < dimNum; i++) {
        inputSize *= static_cast<GmOffsetType>(mTD_->inputShape[i]);
    }
    if (inputSize == 0) {
        return;
    }

    // 紧凑 stride：paddedStride[dim-1]=1, paddedStride[k] = ∏_{j>k} outputShape[j]
    __ubuf__ U paddedStrides[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ U origStrides[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    {
        U ps = 1, os = 1;
        for (int i = static_cast<int>(dimNum) - 1; i >= 0; i--) {
            paddedStrides[i] = ps;
            origStrides[i] = os;
            ps = static_cast<U>(static_cast<GmOffsetType>(ps) * static_cast<GmOffsetType>(mTD_->outputShape[i]));
            os = static_cast<U>(static_cast<GmOffsetType>(os) * static_cast<GmOffsetType>(mTD_->inputShape[i]));
        }
    }

    // 快速除 magic/shift
    __ubuf__ GmOffsetType magics[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ GmOffsetType shifts[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ U paddedShapes[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ U origShapes[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ U leftPads[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];
    __ubuf__ U rightPads[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];

    for (uint32_t i = 0; i < dimNum; i++) {
        paddedShapes[i] = static_cast<U>(mTD_->outputShape[i]);
        origShapes[i] = static_cast<U>(mTD_->inputShape[i]);
        leftPads[i] = static_cast<U>(mTD_->leftPad[i]);
        rightPads[i] = static_cast<U>(mTD_->rightPad[i]);
        GmOffsetType m = 0, s = 0;
        GetUintDivMagicAndShift(m, s, static_cast<GmOffsetType>(origStrides[i]));
        magics[i] = m;
        shifts[i] = s;
    }
    DataSyncBarrier<MemDsbT::UB>();

    // dim-based dispatch (与 pad_v3_grad edge_simt 对齐的线程分配策略)
    if (dimNum == 1) {
        Simt::VF_CALL<SimtComputeEdgeDimOne<T, 1, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 2) {
        Simt::VF_CALL<SimtComputeEdgeDimTwo<T, 2, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_HALF_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 3) {
        Simt::VF_CALL<SimtComputeEdgeDimThree<T, 3, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_QUARTER_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 4) {
        Simt::VF_CALL<SimtComputeEdgeDimFour<T, 4, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_EIGHTH_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 5) {
        Simt::VF_CALL<SimtComputeEdgeDimFive<T, 5, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_EIGHTH_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 6) {
        Simt::VF_CALL<SimtComputeEdgeDimSix<T, 6, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_SIXTEENTH_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 7) {
        Simt::VF_CALL<SimtComputeEdgeDimSeven<T, 7, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_SIXTEENTH_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    } else if (dimNum == 8) {
        Simt::VF_CALL<SimtComputeEdgeDimEight<T, 8, U, GmOffsetType, CastType>>(
            Simt::Dim3(REP_EDGE_THIRTY2ND_THREAD_DIM), (__gm__ T*)(gradOutGM_.GetPhyAddr()),
            (__gm__ volatile T*)(gradInGM_.GetPhyAddr()), inputSize, mBlockIdx_, blockNum, paddedShapes, origShapes,
            paddedStrides, origStrides, leftPads, rightPads, magics, shifts);
    }
}

} // namespace PadV3GradReplication

#endif // PAD_V3_GRAD_REPLICATION_EDGE_SIMT_H
