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

constexpr int32_t CROSS_THREAD_DIM = 1024;
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

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                const CrossRegbaseTilingData& tilingData)
    {
        totalVectors_ = tilingData.totalVectors;
        vectorsPerCore_ = tilingData.vectorsPerCore;
        coreNum_ = tilingData.coreNum;
        dim_ = tilingData.dim;
        dimNum_ = tilingData.dimNum;
        dimStride_ = tilingData.dimStride;
        formerCore_ = tilingData.formerCore;
        usedInt64_ = tilingData.usedInt64;

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
        int32_t startIdx = (blockIdx < formerCore_) ? (vectorsPerCore_+1)*blockIdx : formerCore_ + vectorsPerCore_*blockIdx;
        int32_t count    = (blockIdx < formerCore_) ? (vectorsPerCore_+1) : vectorsPerCore_;

        if (count <= 0) return;

        if (!usedInt64_) {
            asc_vf_call<SimtCrossCompute<T>>(
                dim3(CROSS_THREAD_DIM),
                (__gm__ T*)x1Gm_.GetPhyAddr(),
                (__gm__ T*)x2Gm_.GetPhyAddr(),
                (__gm__ T*)yGm_.GetPhyAddr(),
                startIdx, count, dim_, dimNum_, dimStride_, para_
            );
        } else {
            asc_vf_call<SimtCrossComputeInt64<T>>(
                dim3(CROSS_THREAD_DIM_FOR_INT64),
                (__gm__ T*)x1Gm_.GetPhyAddr(),
                (__gm__ T*)x2Gm_.GetPhyAddr(),
                (__gm__ T*)yGm_.GetPhyAddr(),
                startIdx, count, dim_, dimNum_, dimStride_, para_
            );
        }
    }

private:
    // 普通 int32 偏移版本
    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM)
    static void SimtCrossCompute(
        __gm__ U* x1Gm, __gm__ U* x2Gm, __gm__ U* yGm,
        int32_t startIdx, int32_t count, int32_t dim, int32_t dimNum, int32_t dimStride,
        CrossStridePara para)
    {
        const int32_t idx   = threadIdx.x;
        const int32_t step  = blockDim.x;
        const int32_t sx1   = para.x1s[dim];
        const int32_t sx2   = para.x2s[dim];
        const int32_t sy    = para.ys[dim];

        int32_t i = idx;
        while (i < count) {
            int32_t v = startIdx + i;
            int32_t x1b = 0, x2b = 0, yb = 0;

            for (int d = 0; d < dimNum; d++) {
                int32_t c = v / para.ms[d];
                v -= c * para.ms[d];
                x1b += c * para.x1s[d];
                x2b += c * para.x2s[d];
                yb += c * para.ys[d];
            }

            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2*sx1];
            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2*sx2];

            yGm[yb]         = (U)(a1*b2 - a2*b1);
            yGm[yb + sy]    = (U)(a2*b0 - a0*b2);
            yGm[yb + 2*sy]  = (U)(a0*b1 - a1*b0);

            i += step;
        }
    }

    template <typename U>
    __simt_vf__ LAUNCH_BOUND(CROSS_THREAD_DIM_FOR_INT64)
    static void SimtCrossComputeInt64(
        __gm__ U* x1Gm, __gm__ U* x2Gm, __gm__ U* yGm,
        int32_t startIdx, int32_t count, int32_t dim, int32_t dimNum, int32_t dimStride,
        CrossStridePara para)
    {
        const int32_t idx   = threadIdx.x;
        const int32_t step  = blockDim.x;

        const int64_t sx1   = (int64_t)para.x1s[dim];
        const int64_t sx2   = (int64_t)para.x2s[dim];
        const int64_t sy    = (int64_t)para.ys[dim];

        int32_t i = idx;
        while (i < count) {
            int32_t v = startIdx + i;
            int64_t x1b = 0;
            int64_t x2b = 0;
            int64_t yb  = 0;

            for (int d = 0; d < dimNum; d++) {
                int64_t m    = (int64_t)para.ms[d];
                int64_t x1s  = (int64_t)para.x1s[d];
                int64_t x2s  = (int64_t)para.x2s[d];
                int64_t ys   = (int64_t)para.ys[d];

                int64_t c = (int64_t)v / m;
                v = (int32_t)((int64_t)v - c * m);

                x1b += c * x1s;
                x2b += c * x2s;
                yb  += c * ys;
            }

            float a0 = (float)x1Gm[x1b];
            float a1 = (float)x1Gm[x1b + sx1];
            float a2 = (float)x1Gm[x1b + 2 * sx1];

            float b0 = (float)x2Gm[x2b];
            float b1 = (float)x2Gm[x2b + sx2];
            float b2 = (float)x2Gm[x2b + 2 * sx2];

            yGm[yb]          = (U)(a1 * b2 - a2 * b1);
            yGm[yb + sy]     = (U)(a2 * b0 - a0 * b2);
            yGm[yb + 2 * sy] = (U)(a0 * b1 - a1 * b0);

            i += step;
        }
    }

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    int32_t totalVectors_{0};
    int32_t vectorsPerCore_{0};
    int32_t coreNum_{0};
    int32_t dim_{0};
    int32_t dimNum_{0};
    int32_t dimStride_{1};
    int32_t formerCore_{0};
    bool usedInt64_{false};
    CrossStridePara para_;
};

} // namespace CrossKernel
#endif