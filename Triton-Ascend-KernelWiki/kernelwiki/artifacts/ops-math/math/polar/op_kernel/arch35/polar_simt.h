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
 * \file polar_simt.h
 * \brief polar simt kernel
 */
#ifndef POLAR_SIMT_H
#define POLAR_SIMT_H

#include "kernel_operator.h"
#include "polar_struct.h"
#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

namespace PolarOp {
using namespace AscendC;

constexpr int32_t POLAR_THREAD_DIM = 1024;

struct PolarStridePara {
    int64_t ms[POLAR_MAX_DIM];
    int64_t absS[POLAR_MAX_DIM];
    int64_t angleS[POLAR_MAX_DIM];
    int64_t yS[POLAR_MAX_DIM];
};

template <typename T>
class PolarSimt {
public:
    __aicore__ inline PolarSimt() {}
    __aicore__ inline ~PolarSimt() {}

    __aicore__ inline void Init(GM_ADDR abs, GM_ADDR angle, GM_ADDR y, const PolarTilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        coreNum_ = tilingData.coreNum;
        formerCore_ = tilingData.formerCore;
        dimNum_ = tilingData.dimNum;

        for (int i = 0; i < POLAR_MAX_DIM; i++) {
            para_.ms[i] = tilingData.mergedStride[i];
            para_.absS[i] = tilingData.absStride[i];
            para_.angleS[i] = tilingData.angleStride[i];
            para_.yS[i] = tilingData.yStride[i];
        }

        absGm_.SetGlobalBuffer((__gm__ T*)abs);
        angleGm_.SetGlobalBuffer((__gm__ T*)angle);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
    }

    __aicore__ inline void Process()
    {
        int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        int64_t startIdx = (blockIdx < formerCore_) ? (elementsPerCore_ + 1) * blockIdx :
                                                      formerCore_ + elementsPerCore_ * blockIdx;
        int64_t count = (blockIdx < formerCore_) ? (elementsPerCore_ + 1) : elementsPerCore_;

        if (count <= 0)
            return;

        asc_vf_call<SimtPolarCompute<T>>(dim3(POLAR_THREAD_DIM), (__gm__ T*)(absGm_.GetPhyAddr()),
                                         (__gm__ T*)(angleGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), startIdx,
                                         count, dimNum_, para_);
    }

private:
    template <typename U>
    __simt_vf__ LAUNCH_BOUND(POLAR_THREAD_DIM) static void SimtPolarCompute(__gm__ U* absGm, __gm__ U* angleGm,
                                                                            __gm__ U* yGm, int64_t startIdx,
                                                                            int64_t count, int64_t dimNum,
                                                                            PolarStridePara para)
    {
        const int64_t idx = threadIdx.x;
        const int64_t step = blockDim.x;
        int64_t i = idx;
        while (i < count) {
            int64_t v = startIdx + i;
            int64_t absBase = 0;
            int64_t angleBase = 0;
            int64_t yBase = 0;
            int64_t vv = v;
            for (int64_t d = 0; d < dimNum; d++) {
                int64_t c = vv / para.ms[d];
                vv -= c * para.ms[d];
                absBase += c * para.absS[d];
                angleBase += c * para.angleS[d];
                yBase += c * para.yS[d];
            }
            U absVal = absGm[absBase];
            U cosVal = Simt::Cos(angleGm[angleBase]);
            U sinVal = Simt::Sin(angleGm[angleBase]);
            yGm[2 * yBase] = absVal * cosVal;
            yGm[2 * yBase + 1] = absVal * sinVal;
            i += step;
        }
    }

    GlobalTensor<T> absGm_;
    GlobalTensor<T> angleGm_;
    GlobalTensor<T> yGm_;
    int64_t totalElements_{0};
    int64_t elementsPerCore_{0};
    int64_t coreNum_{0};
    int64_t formerCore_{0};
    int64_t dimNum_{0};
    PolarStridePara para_;
};

} // namespace PolarOp

#endif // POLAR_SIMT_H
