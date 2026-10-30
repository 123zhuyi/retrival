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
 * \file roll_hsplit.h
 * \brief
 */

#ifndef __ROLL_HSPLIT_SIMD_H__
#define __ROLL_HSPLIT_SIMD_H__
#include "kernel_operator.h"
#include "roll_struct.h"
#include "op_kernel/platform_util.h"

namespace Roll {
using namespace AscendC;
constexpr int32_t BTYEALIGNSIZE = 32;
constexpr int32_t REG_SIZE = Ops::Base::GetVRegSize();
constexpr int32_t INC_LEN = 96;
template <typename T>
class RollHSplitSimd {
public:
    __aicore__ inline RollHSplitSimd(TPipe* pipe, const RollTilingData* tiling) : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InsertSync(const HardEvent& event);
    __aicore__ inline void CopyIn(int64_t index, int64_t block_count, int64_t block_len);
    __aicore__ inline void Gather(int32_t Addr, uint16_t repeatnum, int64_t stride);
    __aicore__ inline void CopyOut(LocalTensor<T>& Tensor, int64_t inputIndex, int64_t outIndex, int64_t block_count,
                                   int64_t block_len, int64_t src_stride, int64_t dst_stride);
    __aicore__ inline void CopyOut_Align(LocalTensor<T> xTensor, int64_t xTensorAddr, int64_t index, int64_t offsetOut,
                                         int64_t blockCount);
    __aicore__ inline void CopyOut_UnAlign(LocalTensor<T> xTensor, LocalTensor<T> alienTensor, int64_t xTensorAddr,
                                           int64_t alienTensorAddr, int64_t index, int64_t offsetOut,
                                           int64_t blockCount, int64_t maskNum, bool outKey);
    __aicore__ inline void ComputeAllParam(int64_t inputIndex, int64_t index, int64_t hRe, bool isCount);
    __aicore__ inline void ComputeAllParamForFour(int64_t inputIndex, int64_t hRe, bool isCount);
    __aicore__ inline void CopyInAndOut(int64_t inputIndex, int64_t& offsetIn, int64_t hRe, int64_t maskNum,
                                        bool outKey, bool isAlign);
    __aicore__ inline void Process_UnAlignAndAlign(int64_t inputIndex, bool isAlign);
    __aicore__ inline void Process();

private:
    const RollTilingData* tilingData_;
    TPipe* pipe_;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xInQue_;
    TQueBind<TPosition::VECOUT, TPosition::GM, 1> AlienBuf;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t curCoreBaseIndex_ = 0;
    int64_t perCoreElements_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t blockIdx_ = 0;
    int64_t inputIndices_[MAX_DIM_NUM] = {0};

    int64_t hShapes = 0;
    int64_t wShapes = 0;
    int64_t hLen_ = 0;
    int64_t blockFactor = 0;
    int64_t wAlienLen_ = 0;
    int64_t WAddrShift = 0;
    int64_t dstStride = 0;
    int64_t outIndex[4] = {0};
    int64_t countSum[4] = {0};
    int64_t lenSum[4] = {0};
    UbParam coreUbParam;
};

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    coreUbParam = (blockIdx_ == tilingData_->blockCount - 1) ? tilingData_->tailCoreUbParam :
                                                               tilingData_->mainCoreUbParam;
    blockFactor = (blockIdx_ == tilingData_->blockCount - 1) ? tilingData_->blockTailFactor : tilingData_->blockFactor;
    hShapes = tilingData_->shapes[tilingData_->dimNum - 2];
    wShapes = tilingData_->shapes[tilingData_->dimNum - 1];
    hLen_ = (tilingData_->blockSplitAxis == coreUbParam.UbSplitAxis) ? blockFactor :
                                                                       tilingData_->shapes[tilingData_->dimNum - 2];
    wAlienLen_ = (wShapes * sizeof(T) + BTYEALIGNSIZE - 1) / BTYEALIGNSIZE * BTYEALIGNSIZE / sizeof(T);
    dstStride = REG_SIZE / sizeof(T);
    WAddrShift = wShapes - tilingData_->shifts[tilingData_->dimNum - 1];

    curCoreElements_ = blockFactor * tilingData_->strides[tilingData_->blockSplitAxis];
    perCoreElements_ = tilingData_->blockFactor * tilingData_->strides[tilingData_->blockSplitAxis];
    curCoreBaseIndex_ = perCoreElements_ * blockIdx_;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(xInQue_, BUF_NUM, coreUbParam.UbFactor * wAlienLen_ * sizeof(T));
    if ((tilingData_->shifts[tilingData_->dimNum - 1] != 0) && (WAddrShift * sizeof(T) % BTYEALIGNSIZE != 0)) {
        pipe_->InitBuffer(AlienBuf, BUF_NUM, coreUbParam.UbFactor * REG_SIZE);
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE3_V:
            SetFlag<HardEvent::MTE3_V>(eventID);
            WaitFlag<HardEvent::MTE3_V>(eventID);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::CopyIn(int64_t index, int64_t block_count, int64_t block_len)
{
    LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_count;
    dataCopyParams.blockLen = block_len * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xTensor, xGm_[index], dataCopyParams, dataCopyPadParams);
    xInQue_.EnQue<T>(xTensor);
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::Gather(int32_t Addr, uint16_t repeatnum, int64_t stride)
{
    LocalTensor<T> xTensor = xInQue_.DeQue<T>();
    LocalTensor<T> AlienTensor = AlienBuf.AllocTensor<T>();
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> dst0;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        auto dstPtr = (__ubuf__ T*)AlienTensor.GetPhyAddr();
        auto srcPtr = (__ubuf__ T*)xTensor.GetPhyAddr();
        for (uint16_t i = 0; i < repeatnum; i++) {
            auto srcUbT = srcPtr + Addr + i * stride;
            Reg::DataCopyUnAlignPre(u0, srcUbT);
            Reg::DataCopyUnAlign(dst0, u0, srcUbT);
            Reg::DataCopyUnAlign(dstPtr, dst0, u1, dstStride);
        }
        Reg::DataCopyUnAlignPost(dstPtr, u1, 0);
    }
    xInQue_.EnQue<T>(xTensor);
    AlienBuf.EnQue<T>(AlienTensor);
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::CopyOut(LocalTensor<T>& Tensor, int64_t inputIndex, int64_t outIndex,
                                                  int64_t block_count, int64_t block_len, int64_t src_stride,
                                                  int64_t dst_stride)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_count;
    dataCopyParams.blockLen = block_len * sizeof(T);
    dataCopyParams.srcStride = src_stride * sizeof(T) / BTYEALIGNSIZE;
    dataCopyParams.dstStride = dst_stride * sizeof(T);
    DataCopyPad(yGm_[outIndex], Tensor[inputIndex], dataCopyParams);
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::ComputeAllParam(int64_t inputIndex, int64_t index, int64_t hRe, bool isCount)
{
    outIndex[index] = 0;
    for (int64_t dim = 0; dim < tilingData_->dimNum; dim++) {
        inputIndices_[dim] = inputIndex / tilingData_->strides[dim];
        inputIndex = inputIndex % tilingData_->strides[dim];
        outIndex[index] += (inputIndices_[dim] + tilingData_->shifts[dim]) % tilingData_->shapes[dim] *
                           tilingData_->strides[dim];
    }
    if (!isCount) {
        return;
    }
    if (tilingData_->shifts[tilingData_->dimNum - 1] != 0 && tilingData_->shifts[tilingData_->dimNum - 2] != 0) {
        countSum[index] = tilingData_->shapes[tilingData_->dimNum - 2] - inputIndices_[tilingData_->dimNum - 2];
        if (countSum[index] > tilingData_->shifts[tilingData_->dimNum - 2]) {
            countSum[index] -= tilingData_->shifts[tilingData_->dimNum - 2];
        }
        if (countSum[index] > hRe) {
            countSum[index] = hRe;
        }
        lenSum[index] = tilingData_->shapes[tilingData_->dimNum - 1] - inputIndices_[tilingData_->dimNum - 1];
        if (lenSum[index] > tilingData_->shifts[tilingData_->dimNum - 1]) {
            lenSum[index] -= tilingData_->shifts[tilingData_->dimNum - 1];
        }
    } else {
        if (tilingData_->shifts[tilingData_->dimNum - 2] == 0) {
            countSum[index] = hRe;
            lenSum[index] = tilingData_->shapes[tilingData_->dimNum - 1] - inputIndices_[tilingData_->dimNum - 1];
            if (lenSum[index] > tilingData_->shifts[tilingData_->dimNum - 1]) {
                lenSum[index] -= tilingData_->shifts[tilingData_->dimNum - 1];
            }
        } else if (tilingData_->shifts[tilingData_->dimNum - 1] == 0) {
            lenSum[index] = wShapes;
            countSum[index] = tilingData_->shapes[tilingData_->dimNum - 2] - inputIndices_[tilingData_->dimNum - 2];
            if (countSum[index] > tilingData_->shifts[tilingData_->dimNum - 2]) {
                countSum[index] -= tilingData_->shifts[tilingData_->dimNum - 2];
            }
            if (countSum[index] > hRe) {
                countSum[index] = hRe;
            }
        }
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::ComputeAllParamForFour(int64_t inputIndex, int64_t hRe, bool isCount)
{
    ComputeAllParam(inputIndex, 0, hRe, isCount);
    if (tilingData_->shifts[tilingData_->dimNum - 1] != 0) {
        ComputeAllParam(inputIndex + lenSum[0], 1, hRe, isCount);
    }
    hRe -= countSum[0];
    if (isCount) {
        for (int32_t i = 0; i < 2; i++) {
            countSum[i + 2] = 0;
            lenSum[i + 2] = 0;
        }
    }
    if (hRe != 0 && tilingData_->shifts[tilingData_->dimNum - 2] != 0) {
        ComputeAllParam(inputIndex + countSum[0] * wShapes, 2, hRe, isCount);
        if (tilingData_->shifts[tilingData_->dimNum - 1] != 0) {
            ComputeAllParam(inputIndex + lenSum[0] + countSum[0] * wShapes, 3, hRe, isCount);
        }
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::CopyOut_Align(LocalTensor<T> xTensor, int64_t xTensorAddr, int64_t index,
                                                        int64_t offsetOut, int64_t blockCount)
{
    CopyOut(xTensor, xTensorAddr, outIndex[2 * index] + offsetOut, blockCount, lenSum[2 * index],
            wAlienLen_ - lenSum[2 * index], lenSum[2 * index + 1]);
    if (lenSum[2 * index] < wShapes) {
        CopyOut(xTensor, xTensorAddr + lenSum[2 * index], outIndex[2 * index + 1] + offsetOut, blockCount,
                lenSum[2 * index + 1], wAlienLen_ - lenSum[2 * index + 1], lenSum[2 * index]);
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::CopyOut_UnAlign(LocalTensor<T> xTensor, LocalTensor<T> alienTensor,
                                                          int64_t xTensorAddr, int64_t alienTensorAddr, int64_t index,
                                                          int64_t offsetOut, int64_t blockCount, int64_t maskNum,
                                                          bool outKey)
{
    CopyOut(xTensor, xTensorAddr, outIndex[2 * index] + offsetOut, blockCount, lenSum[2 * index],
            wAlienLen_ - lenSum[2 * index], lenSum[2 * index + 1]);
    CopyOut(alienTensor, alienTensorAddr, outIndex[2 * index + 1] + offsetOut, blockCount, maskNum, dstStride - maskNum,
            wShapes - maskNum);
    if (outKey) {
        CopyOut(xTensor, xTensorAddr + lenSum[2 * index] + maskNum, outIndex[2 * index + 1] + offsetOut + maskNum,
                blockCount, lenSum[2 * index + 1] - maskNum, wAlienLen_ - lenSum[2 * index + 1] + maskNum,
                lenSum[2 * index] + maskNum);
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::CopyInAndOut(int64_t inputIndex, int64_t& offsetIn, int64_t hRe,
                                                       int64_t maskNum, bool outKey, bool isAlign)
{
    int64_t hLenRe = hRe;
    int64_t offsetOut = 0;
    for (int32_t i = 0; i < 2; i++) {
        int64_t factor = (hLenRe == coreUbParam.UbTailFactor) ? coreUbParam.UbTailFactor : coreUbParam.UbFactor;
        if (hLenRe < coreUbParam.UbFactor) {
            factor = 0;
        }
        int64_t countloops = countSum[2 * i] / factor;
        int64_t perLoopCount = factor;
        int64_t lastLoopCount = countSum[2 * i] - countloops * perLoopCount;
        for (int64_t countloop = 0; countloop < countloops; countloop++) {
            CopyIn(inputIndex + offsetIn, perLoopCount, wShapes);
            if (isAlign) {
                LocalTensor<T> xTensor = xInQue_.DeQue<T>();
                CopyOut_Align(xTensor, 0, i, offsetOut, perLoopCount);
                xInQue_.FreeTensor<T>(xTensor);
            } else {
                InsertSync(HardEvent::MTE2_V);
                Gather(lenSum[2 * i], perLoopCount, wAlienLen_);
                InsertSync(HardEvent::V_MTE3);
                LocalTensor<T> xTensor = xInQue_.DeQue<T>();
                LocalTensor<T> alienTensor = AlienBuf.DeQue<T>();
                CopyOut_UnAlign(xTensor, alienTensor, 0, 0, i, offsetOut, perLoopCount, maskNum, outKey);
                xInQue_.FreeTensor<T>(xTensor);
                AlienBuf.FreeTensor<T>(alienTensor);
            }
            offsetIn += perLoopCount * wShapes;
            offsetOut += perLoopCount * wShapes;
        }
        hLenRe -= countloops * perLoopCount;
        factor = (hLenRe == coreUbParam.UbTailFactor) ? coreUbParam.UbTailFactor : coreUbParam.UbFactor;
        if (hLenRe < coreUbParam.UbFactor) {
            factor = hLenRe;
        }
        if (lastLoopCount > 0) {
            CopyIn(inputIndex + offsetIn, factor, wShapes);
            if (isAlign) {
                LocalTensor<T> xTensor = xInQue_.DeQue<T>();
                CopyOut_Align(xTensor, 0, i, offsetOut, lastLoopCount);
                if (lastLoopCount != factor) {
                    CopyOut_Align(xTensor, lastLoopCount * wAlienLen_, i + 1, 0, factor - lastLoopCount);
                    countSum[2 * i + 2] -= factor - lastLoopCount;
                    countSum[2 * i + 3] -= factor - lastLoopCount;
                }
                xInQue_.FreeTensor<T>(xTensor);
            } else {
                InsertSync(HardEvent::MTE2_V);
                Gather(lenSum[2 * i], factor, wAlienLen_);
                InsertSync(HardEvent::V_MTE3);
                LocalTensor<T> xTensor = xInQue_.DeQue<T>();
                LocalTensor<T> alienTensor = AlienBuf.DeQue<T>();
                CopyOut_UnAlign(xTensor, alienTensor, 0, 0, i, offsetOut, lastLoopCount, maskNum, outKey);
                if (lastLoopCount != factor) {
                    InsertSync(HardEvent::V_MTE3);
                    CopyOut_UnAlign(xTensor, alienTensor, lastLoopCount * wAlienLen_, lastLoopCount * dstStride, i + 1,
                                    0, factor - lastLoopCount, maskNum, outKey);
                    countSum[2 * i + 2] -= factor - lastLoopCount;
                    countSum[2 * i + 3] -= factor - lastLoopCount;
                }
                xInQue_.FreeTensor<T>(xTensor);
                AlienBuf.FreeTensor<T>(alienTensor);
            }
            offsetIn += factor * wShapes;
            hLenRe -= factor;
            offsetOut = (factor - lastLoopCount) * wShapes;
            continue;
        }
        offsetOut = 0;
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::Process_UnAlignAndAlign(int64_t inputIndex, bool isAlign)
{
    int64_t maskNum = 0;
    bool outKey = true;
    if (!isAlign) {
        if (tilingData_->shifts[tilingData_->dimNum - 1] <= REG_SIZE / sizeof(T)) {
            maskNum = tilingData_->shifts[tilingData_->dimNum - 1];
            outKey = false;
        } else {
            int32_t line = (WAddrShift * sizeof(T) + BTYEALIGNSIZE - 1) / BTYEALIGNSIZE;
            maskNum = (line * BTYEALIGNSIZE - WAddrShift * sizeof(T) + INC_LEN) / sizeof(T);
        }
    }
    int64_t HStartlen = 0;
    if (tilingData_->blockSplitAxis == coreUbParam.UbSplitAxis && inputIndex != 0) {
        HStartlen = (inputIndex / wShapes + hShapes - 1) / hShapes * hShapes - inputIndex / wShapes;
        if (HStartlen > blockFactor) {
            HStartlen = blockFactor;
        }
    }
    int64_t offsetIn = 0;
    if (HStartlen > 0) {
        ComputeAllParamForFour(inputIndex, HStartlen, true);
        CopyInAndOut(inputIndex, offsetIn, HStartlen, maskNum, outKey, isAlign);
    }

    int64_t blockloopCount = (curCoreElements_ / wShapes - HStartlen) / hShapes;
    int64_t lastLoopCount = curCoreElements_ / wShapes - HStartlen - blockloopCount * hShapes;
    if (blockloopCount > 0) {
        ComputeAllParamForFour(inputIndex + offsetIn, hShapes, true);
    }
    int64_t countSum2 = countSum[2];
    int64_t countSum3 = countSum[3];
    for (int64_t countloop = 0; countloop < blockloopCount; countloop++) {
        CopyInAndOut(inputIndex, offsetIn, hShapes, maskNum, outKey, isAlign);
        ComputeAllParamForFour(inputIndex + offsetIn, hShapes, false);
        countSum[2] = countSum2;
        countSum[3] = countSum3;
    }
    if (lastLoopCount > 0) {
        ComputeAllParamForFour(inputIndex + offsetIn, lastLoopCount, true);
        CopyInAndOut(inputIndex, offsetIn, lastLoopCount, maskNum, outKey, isAlign);
    }
}

template <typename T>
__aicore__ inline void RollHSplitSimd<T>::Process()
{
    if ((tilingData_->shifts[tilingData_->dimNum - 1] != 0) && (WAddrShift * sizeof(T) % BTYEALIGNSIZE != 0)) {
        Process_UnAlignAndAlign(curCoreBaseIndex_, false);
    } else {
        Process_UnAlignAndAlign(curCoreBaseIndex_, true);
    }
}
} // namespace Roll
#endif // __ROLL_SIMD_H__
