/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_data_with_simt.h
 * \brief kernel of trans_data with simt
 */

#ifndef TRANS_DATA_WITH_SIMT_IMP_H_
#define TRANS_DATA_WITH_SIMT_IMP_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"

namespace TRSD
{
using namespace AscendC;

constexpr size_t THREAD_BOUND = 2048;

template <typename T>
class TransWithSIMT
{
public:
    __aicore__ inline TransWithSIMT(){};
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, const TransDataASCTilingData* tilingDataPtr);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    const TransDataASCTilingData* tdPtr = nullptr;
};

template <typename T>
__aicore__ inline void TransWithSIMT<T>::Init(GM_ADDR src, GM_ADDR dst, const TransDataASCTilingData* tilingDataPtr)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
    tdPtr = tilingDataPtr;
}

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_BOUND / sizeof(U)) __aicore__
    void SIMTTrans(__gm__ T* dst, __gm__ T* src, uint64_t shapeSize, U c1, U padN, U c0, U oriN, U oriC, U mPNC, U sPNC,
                   U mPNC0, U sPNC0, U mC1, U sC1, U mC0, U sC0, U mPN, U sPN)
{
    uint64_t tNum = uint64_t(blockDim.x);
    uint64_t blockID = uint64_t(blockIdx.x);
    uint64_t bNum = uint64_t(gridDim.x);
    U hIdx = 0;
    U c1Idx = 0;
    U nIdx = 0;
    U cIdx = 0;
    auto oriNC = oriN * oriC;
    for (uint64_t idx = threadIdx.x + blockID * tNum; idx < shapeSize; idx += bNum * tNum) {
        U idxU = U(idx);
        hIdx = Simt::UintDiv(idxU, mPNC, sPNC);
        U c1Cnt = Simt::UintDiv(idxU, mPNC0, sPNC0);
        c1Idx = c1Cnt - Simt::UintDiv(c1Cnt, mC1, sC1) * c1;
        U nCnt = Simt::UintDiv(idxU, mC0, sC0);
        nIdx = nCnt - Simt::UintDiv(nCnt, mPN, sPN) * padN;
        cIdx = idxU - nCnt * c0 + c1Idx * c0;
        if (nIdx >= oriN || cIdx >= oriC) {
            dst[idx] = T(0);
        } else {
            dst[idx] = src[hIdx * oriNC + nIdx * oriC + cIdx];
        }
    }
}

template <typename T>
template <typename U>
__aicore__ inline void TransWithSIMT<T>::Process()
{
    __gm__ T* srcAddr = (__gm__ T*)inGM.GetPhyAddr();
    __gm__ T* dstAddr = (__gm__ T*)outGM.GetPhyAddr();

    auto c0 = U(tdPtr->c0);
    auto oriN = U(tdPtr->n);
    auto oriC = U(tdPtr->c);
    auto c1 = U(Ops::Base::CeilDiv(oriC, c0));
    auto NI = U(16);
    auto padN = U(Ops::Base::CeilAlign(oriN, NI));
    uint64_t shapeSize = uint64_t(tdPtr->h) * padN * c1 * c0;
    int32_t tNum = int32_t(tdPtr->tNum);
    U mPNC = 0;
    U sPNC = 0;
    U mPNC0 = 0;
    U sPNC0 = 0;
    U mC1 = 0;
    U sC1 = 0;
    U mC0 = 0;
    U sC0 = 0;
    U mPN = 0;
    U sPN = 0;
    GetUintDivMagicAndShift(mPNC, sPNC, c1 * padN * c0);
    GetUintDivMagicAndShift(mPNC0, sPNC0, padN * c0);
    GetUintDivMagicAndShift(mC1, sC1, c1);
    GetUintDivMagicAndShift(mC0, sC0, c0);
    GetUintDivMagicAndShift(mPN, sPN, padN);
    asc_vf_call<SIMTTrans<T, U>>(dim3(tNum), dstAddr, srcAddr, shapeSize, c1, padN, c0, oriN, oriC, mPNC, sPNC,
                                   mPNC0, sPNC0, mC1, sC1, mC0, sC0, mPN, sPN);
}

}  // namespace TRSD

#endif  // TRANS_DATA_WITH_SIMT_IMP_H_
