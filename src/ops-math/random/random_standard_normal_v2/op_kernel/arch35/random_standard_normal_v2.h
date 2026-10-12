/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace RandomStandardNormalV2 {
using namespace AscendC;

constexpr uint16_t BUFFER_NUM = 2;
constexpr uint16_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint16_t RESULT_ELEMENT_CNT = 4;
constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
constexpr float DOUBLE_MULTIPLE = 2.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr uint64_t K_Reserved_Per_Output = 256;
constexpr uint64_t GROUP_SIZE = 2;
constexpr uint64_t USED_THREAD = 1024;
constexpr uint64_t THREAD_LAUNCH = 1024;

__simt_callee__ __aicore__ inline void BoxMullerFloat(const float x0, const float x1, float* f0, float* f1)
{
    const float eps = 1.0e-7f;
    float u1 = x0;
    if (u1 < eps) {
        u1 = eps;
    }
    float v1 = static_cast<float>(DOUBLE_MULTIPLE * PI * x1);
    float u2 = sqrtf(-DOUBLE_MULTIPLE * logf(u1));
    sincosf(v1, f0, f1);
    *f0 *= u2;
    *f1 *= u2;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void SimtBoxMuller(__ubuf__ float* yOutputTmp,
                                                                             const uint32_t calCount)
{
    int64_t groupOffset = threadIdx.x * GROUP_SIZE;

    while (groupOffset < calCount) {
        float uniformRes[GROUP_SIZE];
        BoxMullerFloat(yOutputTmp[groupOffset], yOutputTmp[groupOffset + 1], &uniformRes[0], &uniformRes[1]);

        yOutputTmp[groupOffset] = uniformRes[0];
        yOutputTmp[groupOffset + 1] = uniformRes[1];

        groupOffset += blockDim.x * GROUP_SIZE;
    }
}

template <typename T, typename OFFSET_T>
class RandomStandardNormalV2Op : public RandomKernelBase::RandomKernelBaseOp {
public:
    __aicore__ inline RandomStandardNormalV2Op(TPipe* pipe, const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR offset, GM_ADDR y, GM_ADDR count);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;

    GlobalTensor<OFFSET_T> offsetGm_;
    GlobalTensor<T> outputGm_;
    TBuf<QuePosition::VECCALC> uniformResult_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    static constexpr Reg::CastTrait castTraitTf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                   Reg::MaskMergeMode::ZEROING};
};

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::Init(GM_ADDR offset, GM_ADDR y, GM_ADDR count)
{
    RandomKernelBaseOp::VarsInit();
    offsetGm_.SetGlobalBuffer((__gm__ OFFSET_T*)offset);
    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(uniformResult_, tiling_->singleBufferSize * sizeof(float));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, tiling_->singleBufferSize * sizeof(float));
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::Process()
{
    if (offsetGm_(0) > 0) {
        RandomKernelBaseOp::Skip(offsetGm_(0));
    }

    auto blockOffSet = tiling_->normalCoreProNum * blockIdx_;
    int64_t groupCnt = blockOffSet / RESULT_ELEMENT_CNT;
    RandomKernelBaseOp::Skip(groupCnt);
    for (auto idx = 0; idx < ubRepeatimes_; idx++) {
        int64_t currUbTilingSize = idx == (ubRepeatimes_ - 1) ?
                                       curCoreProNum_ - (ubRepeatimes_ - 1) * tiling_->singleBufferSize :
                                       tiling_->singleBufferSize;
        int64_t curOffSet = blockOffSet + idx * tiling_->singleBufferSize;
        LocalTensor<float> yOutputTmp = uniformResult_.Get<float>();
        uint16_t uniformResCount = Ops::Base::CeilAlign(currUbTilingSize, static_cast<OFFSET_T>(DOUBLE_UNIFORM_RESULT));
        PhiloxRandom<10>(yOutputTmp, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         uniformResCount);
        asc_vf_call<SimtBoxMuller>(dim3{USED_THREAD}, (__ubuf__ float*)(yOutputTmp.GetPhyAddr()), uniformResCount);
        LocalTensor<T> Output = outQue_.AllocTensor<T>();
        RandomKernelBase::Float32Conversion(Output, yOutputTmp, currUbTilingSize);
        outQue_.EnQue(Output);
        LocalTensor<T> yOutput = outQue_.DeQue<T>();
        RandomKernelBase::CopyOut(yOutput, outputGm_, 1, static_cast<uint32_t>(currUbTilingSize * sizeof(T)),
                                  curOffSet);
        outQue_.FreeTensor(yOutput);
        groupCnt = currUbTilingSize / RESULT_ELEMENT_CNT;
        RandomKernelBaseOp::Skip(groupCnt);
    }
    SyncAll();

    if (blockIdx_ == tiling_->usedCoreNum - 1) {
        offsetGm_(0) = offsetGm_(0) + tiling_->outputSize * K_Reserved_Per_Output;
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(offsetGm_);
    }
}

} // namespace RandomStandardNormalV2
