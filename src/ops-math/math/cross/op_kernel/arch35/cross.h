/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CROSS_H
#define CROSS_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cross_struct.h"
#include "cross_tiling_key.h"

using namespace AscendC;

constexpr int32_t CROSS_THREAD_DIM = 512;
constexpr int32_t CROSS_THREAD_DIM_FOR_INT64 = 512;

struct CrossStridePara {
    int32_t ms[8];
    int32_t x1s[8];
    int32_t x2s[8];
    int32_t ys[8];
};

namespace CrossKernel {

template <typename T>
class Cross {
public:
    __aicore__ inline Cross() {}
    __aicore__ inline ~Cross() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const CrossRegbaseTilingData& tilingData)
    {
        totalVectors_ = tilingData.totalVectors;
        coreNum_ = tilingData.coreNum;
        dim_ = tilingData.dim;
        dimNum_ = tilingData.dimNum;
        dimStride_ = tilingData.dimStride;
        vectorsPerBlock_ = tilingData.vectorsPerBlock;
        formerBlock_ = tilingData.formerBlock;
        blocksPerCore_ = tilingData.blocksPerCore;
        usedInt64_ = tilingData.usedInt64;
        activeDimCount_ = tilingData.activeDimCount;
        for (int i = 0; i < MAX_DIM; i++) {
            activeDimIndices_[i] = (int32_t)tilingData.activeDimIndices[i];
        }

        for (int i = 0; i < 8; i++) {
            para_.ms[i] = tilingData.mergedStride[i];
            para_.x1s[i] = tilingData.x1Stride[i];
            para_.x2s[i] = tilingData.x2Stride[i];
            para_.ys[i] = tilingData.yStride[i];
        }

        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
    }

    __aicore__ inline void Process()
    {
        int32_t blockIdx = GetBlockIdx();
        int32_t startIdx = (blockIdx < formerBlock_) ? (vectorsPerBlock_ + 1) * blockIdx :
                                                       formerBlock_ + vectorsPerBlock_ * blockIdx;
        int32_t count = (blockIdx < formerBlock_) ? (vectorsPerBlock_ + 1) : vectorsPerBlock_;

        if (count <= 0)
            return;

        constexpr int32_t CROSS_VEC_PATH_MAX_ELEM = 6 * 1024 * 1024;
        if (!usedInt64_) {
            if (para_.x1s[dim_] == 1 && para_.x2s[dim_] == 1 && para_.ys[dim_] == 1 && activeDimCount_ >= 3 &&
                totalVectors_ <= CROSS_VEC_PATH_MAX_ELEM) {
                asc_vf_call<SimtCrossComputeBatchedVec<T>>(
                    dim3(CROSS_THREAD_DIM), (__gm__ T*)x1Gm_.GetPhyAddr(), (__gm__ T*)x2Gm_.GetPhyAddr(),
                    (__gm__ T*)yGm_.GetPhyAddr(), startIdx, count, para_, activeDimCount_, activeDimIndices_[0],
                    activeDimIndices_[1], activeDimIndices_[2], activeDimIndices_[3], activeDimIndices_[4],
                    activeDimIndices_[5], activeDimIndices_[6]);
            } else if (activeDimCount_ == 2) {
                asc_vf_call<SimtCrossCompute2<T>>(dim3(CROSS_THREAD_DIM), (__gm__ T*)x1Gm_.GetPhyAddr(),
                                                  (__gm__ T*)x2Gm_.GetPhyAddr(), (__gm__ T*)yGm_.GetPhyAddr(), startIdx,
                                                  count, dim_, para_, activeDimIndices_[0], activeDimIndices_[1]);
            } else if (activeDimCount_ == 1) {
                asc_vf_call<SimtCrossCompute1<T>>(dim3(CROSS_THREAD_DIM), (__gm__ T*)x1Gm_.GetPhyAddr(),
                                                  (__gm__ T*)x2Gm_.GetPhyAddr(), (__gm__ T*)yGm_.GetPhyAddr(), startIdx,
                                                  count, dim_, para_, activeDimIndices_[0]);
            } else {
                asc_vf_call<SimtCrossComputeBatched<T>>(
                    dim3(CROSS_THREAD_DIM), (__gm__ T*)x1Gm_.GetPhyAddr(), (__gm__ T*)x2Gm_.GetPhyAddr(),
                    (__gm__ T*)yGm_.GetPhyAddr(), startIdx, count, dim_, para_, activeDimCount_, activeDimIndices_[0],
                    activeDimIndices_[1], activeDimIndices_[2], activeDimIndices_[3], activeDimIndices_[4],
                    activeDimIndices_[5], activeDimIndices_[6]);
            }
        } else {
            asc_vf_call<SimtCrossComputeInt64<T>>(
                dim3(CROSS_THREAD_DIM_FOR_INT64), (__gm__ T*)x1Gm_.GetPhyAddr(), (__gm__ T*)x2Gm_.GetPhyAddr(),
                (__gm__ T*)yGm_.GetPhyAddr(), startIdx, count, dim_, dimNum_, dimStride_, para_, activeDimCount_,
                activeDimIndices_[0], activeDimIndices_[1], activeDimIndices_[2], activeDimIndices_[3],
                activeDimIndices_[4], activeDimIndices_[5], activeDimIndices_[6], activeDimIndices_[7]);
        }
    }

private:
    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM_FOR_INT64) static void SimtCrossComputeInt64(
        __gm__ U* x1Gm, __gm__ U* x2Gm, __gm__ U* yGm, int32_t startIdx, int32_t count, int32_t dim, int32_t dimNum,
        int32_t dimStride, CrossStridePara para, int32_t activeDimCount, int32_t ai0, int32_t ai1, int32_t ai2,
        int32_t ai3, int32_t ai4, int32_t ai5, int32_t ai6, int32_t ai7)
    {
        const int32_t idx = threadIdx.x;
        const int32_t step = blockDim.x;

        const int64_t sx1 = (int64_t)para.x1s[dim];
        const int64_t sx2 = (int64_t)para.x2s[dim];
        const int64_t sy = (int64_t)para.ys[dim];
        const int32_t aiArr[8] = {ai0, ai1, ai2, ai3, ai4, ai5, ai6, ai7};

        int32_t i = idx;
        while (i < count) {
            int32_t v = startIdx + i;
            int64_t x1b = 0;
            int64_t x2b = 0;
            int64_t yb = 0;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int64_t m = (int64_t)para.ms[ai];
                int64_t x1s = (int64_t)para.x1s[ai];
                int64_t x2s = (int64_t)para.x2s[ai];
                int64_t ys = (int64_t)para.ys[ai];

                int64_t c = (int64_t)v / m;
                v = (int32_t)((int64_t)v - c * m);

                x1b += c * x1s;
                x2b += c * x2s;
                yb += c * ys;
            }

            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2 * sx1];

            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2 * sx2];

            yGm[yb] = (U)(a1 * b2 - a2 * b1);
            yGm[yb + sy] = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2 * sy] = (U)(a0 * b1 - a1 * b0);

            i += step;
        }
    }

    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM) static void SimtCrossCompute1(__gm__ U* x1Gm, __gm__ U* x2Gm,
                                                                             __gm__ U* yGm, int32_t startIdx,
                                                                             int32_t count, int32_t dim,
                                                                             CrossStridePara para, int32_t ai0)
    {
        const int32_t step = blockDim.x;
        const int32_t sx1 = para.x1s[dim];
        const int32_t sx2 = para.x2s[dim];
        const int32_t sy = para.ys[dim];
        const int32_t m0 = para.ms[ai0];
        const int32_t x1s0 = para.x1s[ai0];
        const int32_t x2s0 = para.x2s[ai0];
        const int32_t ys0 = para.ys[ai0];

        for (int32_t i = threadIdx.x; i < count; i += step) {
            int32_t v = startIdx + i;
            int32_t c0 = v / m0;
            v -= c0 * m0;
            int32_t x1b = c0 * x1s0;
            int32_t x2b = c0 * x2s0;
            int32_t yb = c0 * ys0;

            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2 * sx1];
            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2 * sx2];

            yGm[yb] = (U)(a1 * b2 - a2 * b1);
            yGm[yb + sy] = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2 * sy] = (U)(a0 * b1 - a1 * b0);
        }
    }

    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM) static void SimtCrossCompute2(__gm__ U* x1Gm, __gm__ U* x2Gm,
                                                                             __gm__ U* yGm, int32_t startIdx,
                                                                             int32_t count, int32_t dim,
                                                                             CrossStridePara para, int32_t ai0,
                                                                             int32_t ai1)
    {
        const int32_t step = blockDim.x;
        const int32_t sx1 = para.x1s[dim];
        const int32_t sx2 = para.x2s[dim];
        const int32_t sy = para.ys[dim];
        const int32_t m0 = para.ms[ai0];
        const int32_t x1s0 = para.x1s[ai0];
        const int32_t x2s0 = para.x2s[ai0];
        const int32_t ys0 = para.ys[ai0];
        const int32_t x1s1 = para.x1s[ai1];
        const int32_t x2s1 = para.x2s[ai1];
        const int32_t ys1 = para.ys[ai1];

        for (int32_t i = threadIdx.x; i < count; i += step) {
            int32_t v = startIdx + i;
            int32_t c0 = v / m0;
            v -= c0 * m0;
            int32_t x1b = c0 * x1s0 + v * x1s1;
            int32_t x2b = c0 * x2s0 + v * x2s1;
            int32_t yb = c0 * ys0 + v * ys1;

            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2 * sx1];
            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2 * sx2];

            yGm[yb] = (U)(a1 * b2 - a2 * b1);
            yGm[yb + sy] = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2 * sy] = (U)(a0 * b1 - a1 * b0);
        }
    }

    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM) static void SimtCrossComputeBatched(
        __gm__ U* x1Gm, __gm__ U* x2Gm, __gm__ U* yGm, int32_t startIdx, int32_t count, int32_t dim,
        CrossStridePara para, int32_t activeDimCount, int32_t ai0, int32_t ai1, int32_t ai2, int32_t ai3, int32_t ai4,
        int32_t ai5, int32_t ai6)
    {
        const int32_t step = blockDim.x;
        const int32_t sx1 = para.x1s[dim];
        const int32_t sx2 = para.x2s[dim];
        const int32_t sy = para.ys[dim];
        const int32_t aiArr[CrossConst::MAX_ACTIVE_DIMS] = {ai0, ai1, ai2, ai3, ai4, ai5, ai6};

        int32_t i = threadIdx.x;

        while (i + step < count) {
            int32_t v0 = startIdx + i;
            int32_t v1 = startIdx + i + step;

            int32_t x1b0 = 0, x2b0 = 0, yb0 = 0;
            int32_t vt = v0;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b0 += c * para.x1s[ai];
                x2b0 += c * para.x2s[ai];
                yb0 += c * para.ys[ai];
            }

            int32_t x1b1 = 0, x2b1 = 0, yb1 = 0;
            vt = v1;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b1 += c * para.x1s[ai];
                x2b1 += c * para.x2s[ai];
                yb1 += c * para.ys[ai];
            }

            float a0_0 = (float)x1Gm[x1b0];
            float a1_0 = (float)x1Gm[x1b0 + sx1];
            float a2_0 = (float)x1Gm[x1b0 + 2 * sx1];
            float b0_0 = (float)x2Gm[x2b0];
            float b1_0 = (float)x2Gm[x2b0 + sx2];
            float b2_0 = (float)x2Gm[x2b0 + 2 * sx2];

            float a0_1 = (float)x1Gm[x1b1];
            float a1_1 = (float)x1Gm[x1b1 + sx1];
            float a2_1 = (float)x1Gm[x1b1 + 2 * sx1];
            float b0_1 = (float)x2Gm[x2b1];
            float b1_1 = (float)x2Gm[x2b1 + sx2];
            float b2_1 = (float)x2Gm[x2b1 + 2 * sx2];

            yGm[yb0] = (U)(a1_0 * b2_0 - a2_0 * b1_0);
            yGm[yb0 + sy] = (U)(a2_0 * b0_0 - a0_0 * b2_0);
            yGm[yb0 + 2 * sy] = (U)(a0_0 * b1_0 - a1_0 * b0_0);

            yGm[yb1] = (U)(a1_1 * b2_1 - a2_1 * b1_1);
            yGm[yb1 + sy] = (U)(a2_1 * b0_1 - a0_1 * b2_1);
            yGm[yb1 + 2 * sy] = (U)(a0_1 * b1_1 - a1_1 * b0_1);

            i += 2 * step;
        }

        while (i < count) {
            int32_t v = startIdx + i;
            int32_t x1b = 0, x2b = 0, yb = 0;
            int32_t vt = v;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b += c * para.x1s[ai];
                x2b += c * para.x2s[ai];
                yb += c * para.ys[ai];
            }
            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2 * sx1];
            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2 * sx2];
            yGm[yb] = (U)(a1 * b2 - a2 * b1);
            yGm[yb + sy] = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2 * sy] = (U)(a0 * b1 - a1 * b0);
            i += step;
        }
    }

    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM) static void SimtCrossComputeBatchedVec(
        __gm__ U* x1Gm, __gm__ U* x2Gm, __gm__ U* yGm, int32_t startIdx, int32_t count, CrossStridePara para,
        int32_t activeDimCount, int32_t ai0, int32_t ai1, int32_t ai2, int32_t ai3, int32_t ai4, int32_t ai5,
        int32_t ai6)
    {
        const int32_t step = blockDim.x;
        const int32_t aiArr[CrossConst::MAX_ACTIVE_DIMS] = {ai0, ai1, ai2, ai3, ai4, ai5, ai6};

        int32_t i = threadIdx.x;
        while (i + step < count) {
            int32_t v0 = startIdx + i;
            int32_t v1 = startIdx + i + step;

            int32_t x1b0 = 0, x2b0 = 0, yb0 = 0;
            int32_t vt = v0;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b0 += c * para.x1s[ai];
                x2b0 += c * para.x2s[ai];
                yb0 += c * para.ys[ai];
            }

            int32_t x1b1 = 0, x2b1 = 0, yb1 = 0;
            vt = v1;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b1 += c * para.x1s[ai];
                x2b1 += c * para.x2s[ai];
                yb1 += c * para.ys[ai];
            }

            int32_t x1b0_a = x1b0 & ~3;
            int32_t x1b0_off = x1b0 & 3;
            float4 a0_lo = asc_ldcg(reinterpret_cast<__gm__ float4*>(x1Gm + x1b0_a));
            float4 a0_hi = asc_ldcg(reinterpret_cast<__gm__ float4*>(x1Gm + x1b0_a + 4));

            int32_t x1b1_a = x1b1 & ~3;
            int32_t x1b1_off = x1b1 & 3;
            float4 a1_lo = asc_ldcg(reinterpret_cast<__gm__ float4*>(x1Gm + x1b1_a));
            float4 a1_hi = asc_ldcg(reinterpret_cast<__gm__ float4*>(x1Gm + x1b1_a + 4));

            int32_t x2b0_a = x2b0 & ~3;
            int32_t x2b0_off = x2b0 & 3;
            float4 b0_lo = asc_ldcg(reinterpret_cast<__gm__ float4*>(x2Gm + x2b0_a));
            float4 b0_hi = asc_ldcg(reinterpret_cast<__gm__ float4*>(x2Gm + x2b0_a + 4));

            int32_t x2b1_a = x2b1 & ~3;
            int32_t x2b1_off = x2b1 & 3;
            float4 b1_lo = asc_ldcg(reinterpret_cast<__gm__ float4*>(x2Gm + x2b1_a));
            float4 b1_hi = asc_ldcg(reinterpret_cast<__gm__ float4*>(x2Gm + x2b1_a + 4));

            float a0_0 = (x1b0_off == 0) ? a0_lo.x : (x1b0_off == 1) ? a0_lo.y : (x1b0_off == 2) ? a0_lo.z : a0_lo.w;
            float a1_0 = (x1b0_off == 0) ? a0_lo.y : (x1b0_off == 1) ? a0_lo.z : (x1b0_off == 2) ? a0_lo.w : a0_hi.x;
            float a2_0 = (x1b0_off == 0) ? a0_lo.z : (x1b0_off == 1) ? a0_lo.w : (x1b0_off == 2) ? a0_hi.x : a0_hi.y;
            float a0_1 = (x1b1_off == 0) ? a1_lo.x : (x1b1_off == 1) ? a1_lo.y : (x1b1_off == 2) ? a1_lo.z : a1_lo.w;
            float a1_1 = (x1b1_off == 0) ? a1_lo.y : (x1b1_off == 1) ? a1_lo.z : (x1b1_off == 2) ? a1_lo.w : a1_hi.x;
            float a2_1 = (x1b1_off == 0) ? a1_lo.z : (x1b1_off == 1) ? a1_lo.w : (x1b1_off == 2) ? a1_hi.x : a1_hi.y;
            float b0_0 = (x2b0_off == 0) ? b0_lo.x : (x2b0_off == 1) ? b0_lo.y : (x2b0_off == 2) ? b0_lo.z : b0_lo.w;
            float b1_0 = (x2b0_off == 0) ? b0_lo.y : (x2b0_off == 1) ? b0_lo.z : (x2b0_off == 2) ? b0_lo.w : b0_hi.x;
            float b2_0 = (x2b0_off == 0) ? b0_lo.z : (x2b0_off == 1) ? b0_lo.w : (x2b0_off == 2) ? b0_hi.x : b0_hi.y;
            float b0_1 = (x2b1_off == 0) ? b1_lo.x : (x2b1_off == 1) ? b1_lo.y : (x2b1_off == 2) ? b1_lo.z : b1_lo.w;
            float b1_1 = (x2b1_off == 0) ? b1_lo.y : (x2b1_off == 1) ? b1_lo.z : (x2b1_off == 2) ? b1_lo.w : b1_hi.x;
            float b2_1 = (x2b1_off == 0) ? b1_lo.z : (x2b1_off == 1) ? b1_lo.w : (x2b1_off == 2) ? b1_hi.x : b1_hi.y;

            yGm[yb0] = (U)(a1_0 * b2_0 - a2_0 * b1_0);
            yGm[yb0 + 1] = (U)(a2_0 * b0_0 - a0_0 * b2_0);
            yGm[yb0 + 2] = (U)(a0_0 * b1_0 - a1_0 * b0_0);

            yGm[yb1] = (U)(a1_1 * b2_1 - a2_1 * b1_1);
            yGm[yb1 + 1] = (U)(a2_1 * b0_1 - a0_1 * b2_1);
            yGm[yb1 + 2] = (U)(a0_1 * b1_1 - a1_1 * b0_1);

            i += 2 * step;
        }

        while (i < count) {
            int32_t v = startIdx + i;
            int32_t x1b = 0, x2b = 0, yb = 0;
            int32_t vt = v;
            for (int d = 0; d < activeDimCount; d++) {
                int32_t ai = aiArr[d];
                int32_t m = para.ms[ai];
                int32_t c = vt / m;
                vt -= c * m;
                x1b += c * para.x1s[ai];
                x2b += c * para.x2s[ai];
                yb += c * para.ys[ai];
            }
            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + 1];
            float a2 = (float)x1Gm[x1b + 2];
            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + 1];
            float b2 = (float)x2Gm[x2b + 2];
            yGm[yb] = (U)(a1 * b2 - a2 * b1);
            yGm[yb + 1] = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2] = (U)(a0 * b1 - a1 * b0);
            i += step;
        }
    }

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    int32_t totalVectors_{0};
    int32_t coreNum_{0};
    int32_t dim_{0};
    int32_t dimNum_{0};
    int32_t dimStride_{1};
    int32_t vectorsPerBlock_{0};
    int32_t formerBlock_{0};
    int32_t blocksPerCore_{0};
    int32_t activeDimCount_{0};
    int32_t activeDimIndices_[MAX_DIM];
    bool usedInt64_{false};
    CrossStridePara para_;
};

} // namespace CrossKernel
#endif
