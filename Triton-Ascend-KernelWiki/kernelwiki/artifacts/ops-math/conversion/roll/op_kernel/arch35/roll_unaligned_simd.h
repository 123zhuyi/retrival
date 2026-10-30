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
 * \file roll_unaligned_simd.h
 * \brief
 */

#ifndef __ROLL_UNALIEGNED_SIMD_H__
#define __ROLL_UNALIEGNED_SIMD_H__
#include "kernel_operator.h"
#include "roll_struct.h"
#include "op_kernel/platform_util.h"

namespace Roll {
using namespace AscendC;
constexpr int32_t UNALIGN_BYTESIZE = 32;
constexpr int32_t UNALIGN_REG_SIZE = Ops::Base::GetVRegSize();
constexpr int32_t UNALIGN_INC_LEN = 96;
template <typename T>
class RollUnaliegnedSimd {
public:
    __aicore__ inline RollUnaliegnedSimd(TPipe* pipe, const RollTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIn(int64_t index, int32_t blockCount, int32_t blockLen);
    __aicore__ inline void CopyOutForGather(int64_t srcindex, int64_t outindex, int32_t addr_offset, int32_t alignIndex,
                                            bool outKey, LocalTensor<T>& xTensor, LocalTensor<T>& alignTensor);
    __aicore__ inline void CopyOut(int64_t srcindex, int64_t outindex, LocalTensor<T>& xTensor);
    __aicore__ inline void ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GaTher(int32_t Addr, uint16_t repeatnum, int32_t strides, int32_t dstStride,
                                  int32_t eventIdToMTE2);

private:
    const RollTilingData* tilingData_;
    TPipe* pipe_;
    UbParam curCoreUbParam;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xInQue_;
    TQueBind<TPosition::VECOUT, TPosition::GM, 1> AlienBuf;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int32_t AlignWLen;
    int32_t AlignHWLen;
    int32_t UbForHW;
    int32_t WAddrShift;
    int32_t HWLen;
    int64_t curCoreBaseIndex_ = 0;
    int64_t blockIdx_ = 0;
    int64_t inputIndices_[MAX_DIM_NUM] = {0};
    int32_t gatherKey = 10001;
};

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreUbParam = tilingData_->tailCoreUbParam;
        if (curCoreUbParam.UbFactor == 0) {
            curCoreUbParam.UbFactor = curCoreUbParam.UbTailFactor;
        }
        curCoreUbParam.UbFactor *= tilingData_->strides[curCoreUbParam.UbSplitAxis];
        curCoreUbParam.UbCount = (tilingData_->blockTailFactor * tilingData_->strides[tilingData_->blockSplitAxis] +
                                  curCoreUbParam.UbFactor - 1) /
                                 curCoreUbParam.UbFactor;
        curCoreUbParam.UbTailFactor = tilingData_->blockTailFactor * tilingData_->strides[tilingData_->blockSplitAxis] -
                                      (curCoreUbParam.UbCount - 1) * curCoreUbParam.UbFactor;
    } else {
        curCoreUbParam = tilingData_->mainCoreUbParam;
        if (curCoreUbParam.UbFactor == 0) {
            curCoreUbParam.UbFactor = curCoreUbParam.UbTailFactor;
        }
        curCoreUbParam.UbFactor *= tilingData_->strides[curCoreUbParam.UbSplitAxis];
        curCoreUbParam.UbCount = (tilingData_->blockFactor * tilingData_->strides[tilingData_->blockSplitAxis] +
                                  curCoreUbParam.UbFactor - 1) /
                                 curCoreUbParam.UbFactor;
        curCoreUbParam.UbTailFactor = tilingData_->blockFactor * tilingData_->strides[tilingData_->blockSplitAxis] -
                                      (curCoreUbParam.UbCount - 1) * curCoreUbParam.UbFactor;
    }
    curCoreBaseIndex_ = tilingData_->blockFactor * blockIdx_ * tilingData_->strides[tilingData_->blockSplitAxis];
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    AlignWLen = (tilingData_->shapes[tilingData_->dimNum - 1] * sizeof(T) + UNALIGN_BYTESIZE - 1) / UNALIGN_BYTESIZE *
                UNALIGN_BYTESIZE / sizeof(T);
    AlignHWLen = AlignWLen * tilingData_->shapes[tilingData_->dimNum - 2];
    HWLen = tilingData_->shapes[tilingData_->dimNum - 2] * tilingData_->shapes[tilingData_->dimNum - 1];
    UbForHW = curCoreUbParam.UbFactor / HWLen; // 多少个ub
    WAddrShift = tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1];
    pipe_->InitBuffer(xInQue_, BUF_NUM, UbForHW * AlignHWLen * sizeof(T));
    if (tilingData_->shifts[tilingData_->dimNum - 1] != 0) {
        if (WAddrShift * sizeof(T) % UNALIGN_BYTESIZE != 0) {
            gatherKey = 11000;
            pipe_->InitBuffer(AlienBuf, BUF_NUM,
                              UNALIGN_REG_SIZE * UbForHW * tilingData_->shapes[tilingData_->dimNum - 2]);
        } else {
            gatherKey = 10100;
        }
    } else if (tilingData_->shapes[tilingData_->dimNum - 1] * sizeof(T) % UNALIGN_BYTESIZE != 0) {
        gatherKey = 10010;
    } else {
        gatherKey = 10001;
    }
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::CopyIn(int64_t index, int32_t blockCount, int32_t blockLen)
{
    LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = blockLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xTensor, xGm_[index], dataCopyParams, dataCopyPadParams);
    xInQue_.EnQue<T>(xTensor);
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::GaTher(int32_t Addr, uint16_t repeatnum, int32_t strides,
                                                     int32_t dstStride, int32_t eventIdToMTE2)
{
    WaitFlag<HardEvent::MTE2_V>(eventIdToMTE2);
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
            auto srcUbT = srcPtr + Addr + i * strides;
            Reg::DataCopyUnAlignPre(u0, srcUbT);
            Reg::DataCopyUnAlign(dst0, u0, srcUbT);
            Reg::DataCopyUnAlign(dstPtr, dst0, u1, dstStride);
        }
        Reg::DataCopyUnAlignPost(dstPtr, u1, 0);
    }
    AlienBuf.EnQue<T>(AlienTensor);
    xInQue_.EnQue<T>(xTensor);
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::CopyOutForGather(int64_t srcIndex, int64_t outIndex, int32_t addr_offset,
                                                               int32_t alignIndex, bool outKey, LocalTensor<T>& xTensor,
                                                               LocalTensor<T>& alignTensor)
{
    DataCopyExtParams dataCopyParams;
    for (int32_t i = 0; i < tilingData_->moveparam.mte3Count; i += 2) {
        dataCopyParams.blockCount = tilingData_->moveparam.blockCount[i];
        dataCopyParams.blockLen = tilingData_->moveparam.blockLen[i] * sizeof(T);
        dataCopyParams.srcStride = (tilingData_->moveparam.srcStride[i] - tilingData_->moveparam.blockLen[i]) *
                                   sizeof(T) / UNALIGN_BYTESIZE;
        dataCopyParams.dstStride = (tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->moveparam.blockLen[i]) *
                                   sizeof(T);
        DataCopyPad(yGm_[outIndex + tilingData_->moveparam.dstOffset[i]],
                    xTensor[srcIndex + tilingData_->moveparam.srcOffset[i]], dataCopyParams);
    }
    if (outKey) {
        for (int32_t i = 1; i < tilingData_->moveparam.mte3Count; i += 2) {
            dataCopyParams.blockCount = tilingData_->moveparam.blockCount[i];
            dataCopyParams.blockLen = (tilingData_->moveparam.blockLen[i] - addr_offset) * sizeof(T);
            dataCopyParams.srcStride = (tilingData_->moveparam.srcStride[i] - tilingData_->moveparam.blockLen[i] +
                                        addr_offset) *
                                       sizeof(T) / UNALIGN_BYTESIZE;
            dataCopyParams.dstStride = (tilingData_->shapes[tilingData_->dimNum - 1] -
                                        tilingData_->moveparam.blockLen[i] + addr_offset) *
                                       sizeof(T);
            DataCopyPad(yGm_[outIndex + tilingData_->moveparam.dstOffset[i] + addr_offset],
                        xTensor[srcIndex + tilingData_->moveparam.srcOffset[i] + addr_offset], dataCopyParams);
        }
    }
    for (int32_t i = 1; i < tilingData_->moveparam.mte3Count; i += 2) { // aligntensor
        dataCopyParams.blockCount = tilingData_->moveparam.blockCount[i];
        dataCopyParams.blockLen = addr_offset * sizeof(T);
        dataCopyParams.srcStride = (UNALIGN_REG_SIZE - addr_offset * sizeof(T)) / UNALIGN_BYTESIZE;
        dataCopyParams.dstStride = (tilingData_->shapes[tilingData_->dimNum - 1] - addr_offset) * sizeof(T);
        DataCopyPad(
            yGm_[outIndex + tilingData_->moveparam.dstOffset[i]],
            alignTensor[alignIndex + (i / 2) * tilingData_->moveparam.blockCount[1] * UNALIGN_REG_SIZE / sizeof(T)],
            dataCopyParams);
    }
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::CopyOut(int64_t srcindex, int64_t outindex, LocalTensor<T>& xTensor)
{
    DataCopyExtParams dataCopyParams;
    for (int32_t i = 0; i < tilingData_->moveparam.mte3Count; i++) {
        dataCopyParams.blockCount = tilingData_->moveparam.blockCount[i];
        dataCopyParams.blockLen = tilingData_->moveparam.blockLen[i] * sizeof(T);
        dataCopyParams.srcStride = (tilingData_->moveparam.srcStride[i] - tilingData_->moveparam.blockLen[i]) *
                                   sizeof(T) / UNALIGN_BYTESIZE;
        dataCopyParams.dstStride = (tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->moveparam.blockLen[i]) *
                                   sizeof(T);
        DataCopyPad(yGm_[outindex + tilingData_->moveparam.dstOffset[i]],
                    xTensor[srcindex + tilingData_->moveparam.srcOffset[i]], dataCopyParams);
    }
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex)
{
    for (int64_t dim = 0; dim < tilingData_->dimNum; dim++) {
        inputIndices_[dim] = inputIndex / tilingData_->strides[dim];
        inputIndex = inputIndex % tilingData_->strides[dim];
        if (dim < tilingData_->dimNum - 2) {
            outputIndex += (inputIndices_[dim] + tilingData_->shifts[dim]) % tilingData_->shapes[dim] *
                           tilingData_->strides[dim];
        }
    }
}

template <typename T>
__aicore__ inline void RollUnaliegnedSimd<T>::Process()
{
    if (gatherKey == 11000) {
        int32_t dstStride = UNALIGN_REG_SIZE / sizeof(T);
        int32_t alignSize = dstStride * tilingData_->shapes[tilingData_->dimNum - 2];
        int32_t maskNum;
        bool outKey = true;
        if (tilingData_->shifts[tilingData_->dimNum - 1] <= UNALIGN_REG_SIZE / sizeof(T)) {
            maskNum = tilingData_->shifts[tilingData_->dimNum - 1];
            outKey = false;
        } else {
            int32_t rowstartidx = WAddrShift * sizeof(T);
            int32_t tmpaline = (rowstartidx + UNALIGN_BYTESIZE - 1) / UNALIGN_BYTESIZE * UNALIGN_BYTESIZE;
            maskNum = (tmpaline - rowstartidx + UNALIGN_INC_LEN) / sizeof(T);
        }
        for (int32_t loopNum = 0; loopNum < curCoreUbParam.UbCount; loopNum++) {
            int32_t CopyLoop;
            if (loopNum == curCoreUbParam.UbCount - 1) {
                CopyLoop = curCoreUbParam.UbTailFactor / HWLen;
            } else {
                CopyLoop = UbForHW;
            }
            CopyIn(curCoreBaseIndex_, CopyLoop * tilingData_->shapes[tilingData_->dimNum - 2],
                   tilingData_->shapes[tilingData_->dimNum - 1]);
            int32_t eventIdToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdToMTE2);
            GaTher(WAddrShift, CopyLoop * tilingData_->shapes[tilingData_->dimNum - 2], AlignWLen, dstStride,
                   eventIdToMTE2);
            LocalTensor<T> xTensor = xInQue_.DeQue<T>();
            LocalTensor<T> alienTensor = AlienBuf.DeQue<T>();
            for (int32_t i = 0; i < CopyLoop; i++) {
                int64_t outIndex = 0;
                ComputeOutIndex(curCoreBaseIndex_ + i * HWLen, outIndex);
                CopyOutForGather(i * AlignHWLen, outIndex, maskNum, i * alignSize, outKey, xTensor, alienTensor);
            }
            xInQue_.FreeTensor<T>(xTensor);
            AlienBuf.FreeTensor<T>(alienTensor);
            curCoreBaseIndex_ += curCoreUbParam.UbFactor;
        }
    } else if (gatherKey == 10100 || gatherKey == 10010) {
        for (int32_t loopNum = 0; loopNum < curCoreUbParam.UbCount; loopNum++) {
            int32_t CopyLoop;
            if (loopNum == curCoreUbParam.UbCount - 1) {
                CopyLoop = curCoreUbParam.UbTailFactor / HWLen;
            } else {
                CopyLoop = UbForHW;
            }
            CopyIn(curCoreBaseIndex_, CopyLoop * tilingData_->shapes[tilingData_->dimNum - 2],
                   tilingData_->shapes[tilingData_->dimNum - 1]);
            LocalTensor<T> xTensor = xInQue_.DeQue<T>();
            for (int32_t i = 0; i < CopyLoop; i++) {
                int64_t outIndex = 0;
                ComputeOutIndex(curCoreBaseIndex_ + i * HWLen, outIndex);
                CopyOut(i * AlignHWLen, outIndex, xTensor);
            }
            xInQue_.FreeTensor<T>(xTensor);
            curCoreBaseIndex_ += curCoreUbParam.UbFactor;
        }
    } else {
        for (int32_t loopNum = 0; loopNum < curCoreUbParam.UbCount; loopNum++) {
            int32_t CopyLoop;
            if (loopNum == curCoreUbParam.UbCount - 1) {
                CopyLoop = curCoreUbParam.UbTailFactor / HWLen;
            } else {
                CopyLoop = UbForHW;
            }
            CopyIn(curCoreBaseIndex_, 1, CopyLoop * HWLen);
            LocalTensor<T> xTensor = xInQue_.DeQue<T>();
            for (int32_t i = 0; i < CopyLoop; i++) {
                int64_t outIndex = 0;
                ComputeOutIndex(curCoreBaseIndex_ + i * HWLen, outIndex);
                CopyOut(i * AlignHWLen, outIndex, xTensor);
            }
            xInQue_.FreeTensor<T>(xTensor);
            curCoreBaseIndex_ += curCoreUbParam.UbFactor;
        }
    }
}
} // namespace Roll
#endif // __ROLL_SIMD_H__
