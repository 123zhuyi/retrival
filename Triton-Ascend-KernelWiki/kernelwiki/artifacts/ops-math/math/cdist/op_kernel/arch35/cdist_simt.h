/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_simt.h
 * \brief cdist_simt
 */
 #ifndef ASCENDC_AS_STRIDED_SMIT_H_
 #define ASCENDC_AS_STRIDED_SMIT_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include "../cdist_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "kernel_operator.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace NsCdist {
using namespace AscendC;

constexpr int64_t THREAD_NUM = 512;
constexpr float LN_MIN_FLOAT = -87.0f;

template <typename T>
class CdistSimt
{
public:
    __aicore__ inline CdistSimt(TPipe* pipe, const CdistTilingData* tiling) : pipe_(pipe), tilingData_(tiling) {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y);
    __aicore__ inline void Process();

private:
    const CdistTilingData* tilingData_;
    TPipe* pipe_;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    int64_t blockIdx_ = 0; //核号
    int64_t blockCount_ = 0; //偏移
    int64_t curCoreBaseIndex_ = 0;
    int64_t curCoreEmelents_ = 0;
    int64_t perCoreEmelents_ = 0;
    int64_t B_ = 0;
    int64_t P_ = 0;
    int64_t R_ = 0;
    int64_t M_ = 0;
    float p_ = 2.0f;
};

template <typename T>
__aicore__ inline void CdistSimt<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
{
    blockCount_ = tilingData_->realCoreNum;
    B_ = tilingData_->B;
    P_ = tilingData_->P;
    R_ = tilingData_->R;
    M_ = tilingData_->M;
    p_ = tilingData_->p;

    blockIdx_ = GetBlockIdx();
    perCoreEmelents_ = tilingData_->blockFactor;

    if(blockIdx_ == blockCount_ - 1) {
        curCoreEmelents_ = tilingData_->blockTailFactor;
    } else {
        curCoreEmelents_ = tilingData_->blockFactor;
    }
    curCoreBaseIndex_ = perCoreEmelents_ * blockIdx_;


    x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedZero(
    __gm__ T* x1GmAddr,  __gm__ T* x2GmAddr, __gm__ T* yGmAddr, int64_t outputBasicIndex, int64_t count,
    uint64_t m0, uint64_t s0, uint64_t m1, uint64_t s1, int64_t B, int64_t P, int64_t R, int64_t M)
{  //输入BPM、BRM  输出BPR
    for(uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint64_t outputIdx = outputBasicIndex + index;
        uint64_t inputCurIdx = outputIdx;
        uint64_t Bidx = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= Bidx * P * R;
        uint64_t Pidx = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= Pidx * R;
        uint64_t Ridx = inputCurIdx;

        uint64_t inputIdx1 = Bidx*P*M + Pidx*M;
        uint64_t inputIdx2 = Bidx*R*M + Ridx*M; 

        yGmAddr[outputIdx] = 0;
        for(int64_t i = 0; i < M; i++) {
            float absNum = abs(static_cast<float>(x1GmAddr[inputIdx1+i]) - static_cast<float>(x2GmAddr[inputIdx2+i]));
            float ceilNum = ceilf(absNum);
            float minNum;
            if(isnan(ceilNum)) {
                minNum = ceilNum;
            } else {
                minNum = fminf(ceilNum, static_cast<float>(1));
            }
            asc_atomic_add(yGmAddr + outputIdx, static_cast<T>(minNum));
        }
    }
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedInf(
    __gm__ T* x1GmAddr,  __gm__ T* x2GmAddr, __gm__ T* yGmAddr, int64_t outputBasicIndex, int64_t count,
    uint64_t m0, uint64_t s0, uint64_t m1, uint64_t s1, int64_t B, int64_t P, int64_t R, int64_t M)
{  //输入BPM、BRM  输出BPR
    for(uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint64_t outputIdx = outputBasicIndex + index;
        uint64_t inputCurIdx = outputIdx;
        uint64_t Bidx = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= Bidx * P * R;
        uint64_t Pidx = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= Pidx * R;
        uint64_t Ridx = inputCurIdx;

        uint64_t inputIdx1 = Bidx*P*M + Pidx*M;
        uint64_t inputIdx2 = Bidx*R*M + Ridx*M; 

        yGmAddr[outputIdx] = 0;
        for(int64_t i = 0; i < M; i++) {
            float absNum = abs(static_cast<float>(x1GmAddr[inputIdx1+i]) - static_cast<float>(x2GmAddr[inputIdx2+i]));
            asc_atomic_max(yGmAddr + outputIdx, static_cast<T>(absNum));
        }
    }
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedOne(
    __gm__ T* x1GmAddr,  __gm__ T* x2GmAddr, __gm__ T* yGmAddr, int64_t outputBasicIndex, int64_t count,
    uint64_t m0, uint64_t s0, uint64_t m1, uint64_t s1, int64_t B, int64_t P, int64_t R, int64_t M)
{  //输入BPM、BRM  输出BPR
    for(uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint64_t outputIdx = outputBasicIndex + index;
        uint64_t inputCurIdx = outputIdx;
        uint64_t Bidx = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= Bidx * P * R;
        uint64_t Pidx = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= Pidx * R;
        uint64_t Ridx = inputCurIdx;

        uint64_t inputIdx1 = Bidx*P*M + Pidx*M;
        uint64_t inputIdx2 = Bidx*R*M + Ridx*M; 

        yGmAddr[outputIdx] = 0;
        float sumNum = 0;
        for(int64_t i = 0; i < M; i++) {
            sumNum += abs(static_cast<float>(x1GmAddr[inputIdx1+i]) - static_cast<float>(x2GmAddr[inputIdx2+i]));
        }
        yGmAddr[outputIdx] = sumNum;
    }
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedTwo(
    __gm__ T* x1GmAddr,  __gm__ T* x2GmAddr, __gm__ T* yGmAddr, int64_t outputBasicIndex, int64_t count,
    uint64_t m0, uint64_t s0, uint64_t m1, uint64_t s1, int64_t B, int64_t P, int64_t R, int64_t M)
{  //输入BPM、BRM  输出BPR
    for(uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint64_t outputIdx = outputBasicIndex + index;
        uint64_t inputCurIdx = outputIdx;
        uint64_t Bidx = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= Bidx * P * R;
        uint64_t Pidx = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= Pidx * R;
        uint64_t Ridx = inputCurIdx;

        uint64_t inputIdx1 = Bidx*P*M + Pidx*M;
        uint64_t inputIdx2 = Bidx*R*M + Ridx*M; 

        float sumNum = 0;
        for(int64_t i = 0; i < M; i++) {
            float absNum = static_cast<float>(x1GmAddr[inputIdx1+i]) - static_cast<float>(x2GmAddr[inputIdx2+i]);
            sumNum += absNum * absNum;
        }
        yGmAddr[outputIdx] = sqrtf(sumNum);
    }
}


template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedOther(
    __gm__ T* x1GmAddr,  __gm__ T* x2GmAddr, __gm__ T* yGmAddr, int64_t outputBasicIndex, int64_t count,
    uint64_t m0, uint64_t s0, uint64_t m1, uint64_t s1, int64_t B, int64_t P, int64_t R, int64_t M, float p)
{  //输入BPM、BRM  输出BPR
    for(uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint64_t outputIdx = outputBasicIndex + index;
        uint64_t inputCurIdx = outputIdx;
        uint64_t Bidx = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= Bidx * P * R;
        uint64_t Pidx = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= Pidx * R;
        uint64_t Ridx = inputCurIdx;

        uint64_t inputIdx1 = Bidx*P*M + Pidx*M;
        uint64_t inputIdx2 = Bidx*R*M + Ridx*M; 

        float sumNum = 0;
        for(int64_t i = 0; i < M; i++) {
            float absNum = abs(static_cast<float>(x1GmAddr[inputIdx1+i]) - static_cast<float>(x2GmAddr[inputIdx2+i]));
            float powNum = logf(absNum) * p;
            if(powNum < LN_MIN_FLOAT) {
                powNum *= 0.5f;
                float temp = expf(powNum);
                sumNum += temp * temp;
            }
            else {
                sumNum += expf(powNum);
            }
        }
        yGmAddr[outputIdx] = expf(logf(sumNum)/p);
    }
}

template <typename T>
__aicore__ inline void CdistSimt<T>::Process()
{
    __gm__ T* x1GmAddr = (__gm__ T*)x1Gm_.GetPhyAddr();
    __gm__ T* x2GmAddr = (__gm__ T*)x2Gm_.GetPhyAddr();
    __gm__ T* yGmAddr = (__gm__ T*)yGm_.GetPhyAddr();

    uint64_t magicPR = 0;
    uint64_t shiftPR = 0;
    uint64_t magicR = 0;
    uint64_t shiftR = 0;
    GetUintDivMagicAndShift(magicPR, shiftPR, static_cast<uint64_t>(R_ * P_));
    GetUintDivMagicAndShift(magicR, shiftR, static_cast<uint64_t>(R_));


    if(p_ == 0.0f) {
        asc_vf_call<SimtStridedZero<T>>(
            dim3{THREAD_NUM, 1, 1}, x1GmAddr, x2GmAddr, yGmAddr, curCoreBaseIndex_, curCoreEmelents_,
            magicPR, shiftPR, magicR, shiftR, B_, P_, R_, M_);
    } else if(p_ == 1.0f) {
        asc_vf_call<SimtStridedOne<T>>(
            dim3{THREAD_NUM, 1, 1}, x1GmAddr, x2GmAddr, yGmAddr, curCoreBaseIndex_, curCoreEmelents_,
            magicPR, shiftPR, magicR, shiftR, B_, P_, R_, M_);
    } else if(p_ == static_cast<float>(INFINITY)) {
        asc_vf_call<SimtStridedInf<T>>(
            dim3{THREAD_NUM, 1, 1}, x1GmAddr, x2GmAddr, yGmAddr, curCoreBaseIndex_, curCoreEmelents_,
            magicPR, shiftPR, magicR, shiftR, B_, P_, R_, M_);
    } else if(p_ == 2.0f) {
        asc_vf_call<SimtStridedTwo<T>>(
            dim3{THREAD_NUM, 1, 1}, x1GmAddr, x2GmAddr, yGmAddr, curCoreBaseIndex_, curCoreEmelents_,
            magicPR, shiftPR, magicR, shiftR, B_, P_, R_, M_);
    } else {
        asc_vf_call<SimtStridedOther<T>>(
            dim3{THREAD_NUM, 1, 1}, x1GmAddr, x2GmAddr, yGmAddr, curCoreBaseIndex_, curCoreEmelents_,
            magicPR, shiftPR, magicR, shiftR, B_, P_, R_, M_, p_);
    }
}

}

#endif