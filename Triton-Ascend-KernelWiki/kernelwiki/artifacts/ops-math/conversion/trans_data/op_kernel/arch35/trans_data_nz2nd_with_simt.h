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
 * \file trans_data_nz2nd_with_simt.h
 * \brief kernel of trans_data nz2nd with simt
 */

#ifndef TRANS_DATA_NZ2ND_WITH_SIMT_IMP_H_
#define TRANS_DATA_NZ2ND_WITH_SIMT_IMP_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"

namespace TRSD
{
using namespace AscendC;

constexpr int32_t THREAD_DIM = 512;

template <typename T>
class TransNZ2NDWithSIMT
{
public:
    __aicore__ inline TransNZ2NDWithSIMT(){};
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, const TransDataNzToNdTilingData* tilingDataPtr);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inGM_;
    GlobalTensor<T> outGM_;
    const TransDataNzToNdTilingData* tdPtr_ = nullptr;
};

template <typename T>
__aicore__ inline void TransNZ2NDWithSIMT<T>::Init(GM_ADDR src, GM_ADDR dst, const TransDataNzToNdTilingData* tilingDataPtr)
{
    inGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
    outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
    tdPtr_ = tilingDataPtr;
}

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SIMTNZ2NDTrans(__gm__ T* dst, __gm__ T* src, uint64_t shapeSize, U n, U c, U c1, U n1, U n0, U c0,
                                   U hS, U hM, U nS, U nM, U c1S, U c1M, U n1S, U n1M)
{
    uint64_t tNum = uint64_t(blockDim.x);
    uint64_t blockID = uint64_t(blockIdx.x);
    uint64_t bNum = uint64_t(gridDim.x);
    U hIdx = 0;
    U nIdx = 0;
    U cIdx = 0;
    U c1Idx = 0;
    U n1Idx = 0;
    U n0Idx = 0;
    U c0Idx = 0;
    U inputProdN0 = c0;
    U inputProdN1 = n0 * inputProdN0;
    U inputProdC1 = n1 * inputProdN1;
    U inputProdH = c1 * inputProdC1;

    for (uint64_t idx = threadIdx.x + blockID * tNum; idx < shapeSize; idx += bNum * tNum) {
        U idxU = U(idx);
        hIdx = Simt::UintDiv(idxU, hM, hS);
        idxU -= hIdx * n * c;
        nIdx = Simt::UintDiv(idxU, nM, nS);
        idxU -= nIdx * c;
        cIdx = idxU;

        n1Idx = Simt::UintDiv(nIdx, n1M, n1S);
        n0Idx = nIdx - n1Idx * n0;
        c1Idx = Simt::UintDiv(cIdx, c1M, c1S);
        c0Idx = cIdx - c1Idx * c0;

        U inputIndex = hIdx * inputProdH + c1Idx * inputProdC1 + n1Idx * inputProdN1 + n0Idx * inputProdN0 + c0Idx;
        dst[idx] = src[inputIndex];    
    }
}


template <typename T>
template <typename U>
__aicore__ inline void TransNZ2NDWithSIMT<T>::Process()
{
    __gm__ T* srcAddr = (__gm__ T*)inGM_.GetPhyAddr();
    __gm__ T* dstAddr = (__gm__ T*)outGM_.GetPhyAddr();

    auto h = U(tdPtr_->h);
    auto n = U(tdPtr_->n);
    auto c = U(tdPtr_->c);
    auto c0 = U(tdPtr_->c0);
    auto c1 = U(tdPtr_->c1);
    auto n0 = U(tdPtr_->n0);
    auto n1 = U(tdPtr_->n1);

    U hS = 0;
    U hM = 0;
    U nS = 0;
    U nM = 0;
    GetUintDivMagicAndShift(hM, hS, n * c);
    GetUintDivMagicAndShift(nM, nS, c);

    U c1S = 0;
    U c1M = 0;
    U n1S = 0;
    U n1M = 0;
    GetUintDivMagicAndShift(c1M, c1S, c0);
    GetUintDivMagicAndShift(n1M, n1S, n0);

    uint64_t shapeSize = h * c * n;
    asc_vf_call<SIMTNZ2NDTrans<T, U>>(dim3(THREAD_DIM), dstAddr, srcAddr, shapeSize, n, c, c1, n1, n0, c0,
                                   hS, hM, nS, nM, c1S, c1M, n1S, n1M);
}

}  // namespace TRSD

#endif  // TRANS_DATA_NZ2ND_WITH_SIMT_IMP_H_
