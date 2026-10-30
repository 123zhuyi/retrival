/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_UB_SPLIT_SAME_LEN_SMALL_G_H
#define SPLIT_V_UB_SPLIT_SAME_LEN_SMALL_G_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;
template <typename T, typename U, typename Y> // T原始数据类型  U做datacopygather的数据类型 Y是做vci的数据类型
class SplitVUbSplitSameLenSmallG {
public:
    __aicore__ inline SplitVUbSplitSameLenSmallG(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t srcOffset, int64_t srcStride,
                                  int64_t dstStride, LocalTensor<T>& xUb);
    __aicore__ inline void ComputeIdx(int64_t processGNum, int64_t processMNNum, uint32_t gnAlignSize,
                                      uint32_t mnAlignSize);
    __aicore__ inline void Compute(uint32_t g, int64_t processMNNum, uint32_t gnAlignSize, uint32_t mnAlignSize,
                                   LocalTensor<T>& srcUb);
    __aicore__ inline void CopyOut(int64_t localOffset, int64_t blockCount, int64_t blockLen, int64_t dstOffset,
                                   int64_t srcStride, int64_t dstStride);

private:
    TPipe& pipe_;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static int64_t VL_LEN = Ops::Base::GetVRegSize();
    const SplitVTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TQue<QuePosition::VECCALC, 1> idxQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<T> yLocal_;
    LocalTensor<U> idxLocal_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int64_t dtypeSize_ = sizeof(T);
    int64_t gSize_ = 0;
    int64_t nSize_ = 0;
    int64_t mUBCount_ = 0;
    int64_t mUBFactor_ = 0;
    int64_t mUBFactorTail_ = 0;
    int64_t gUBCount_ = 0;
    int64_t gUBFactor_ = 0;
    int64_t gUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;

    // Record global loopTime
    int64_t curIdx_ = 0;
    int64_t mIdx_ = 0;
    int64_t gIdx_ = 0;
    int64_t gmOffset_ = 0;
    int64_t tmpLoop_ = 0;
    int64_t nRegSize_ = 0;
    int64_t blockCount_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubSize_ = 0;

    int32_t uVL_ = Ops::Base::GetVRegSize() / sizeof(U);
    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};
};

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Init(GM_ADDR x, GM_ADDR y,
                                                                 const SplitVTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    gSize_ = tilingData_->gSize;
    ubSize_ = tilingData_->ubSize;
    nSize_ = tilingData_->nBlockFactorNum;
    mUBFactor_ = tilingData_->mBlockFactor;
    mUBFactorTail_ = tilingData_->mBlockFactorTail;
    mUBCount_ = tilingData_->mBlockCount;
    gUBFactor_ = tilingData_->gUBFactor;
    gUBFactorTail_ = tilingData_->gUBFactorTail;
    gUBCount_ = tilingData_->gUBCount;
    blockFactor_ = tilingData_->blockFactor;
    blockFactorTail_ = tilingData_->blockFactorTail;

    uint32_t initInputSpace = (ubSize_ / BUFFER_NUM - VL_LEN) / BUFFER_NUM;
    uint32_t initIdxSpace = Ops::Base::GetVRegSize();
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, initInputSpace);
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, initInputSpace);
    pipe_.InitBuffer(idxQueue_, BUFFER_NUM, initIdxSpace);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

    // Calc start idx per core
    blkProcessNum_ = blockFactor_;
    curIdx_ = blockIdx_ * blockFactor_;
    if (blockIdx_ < blockFactorTail_) {
        blkProcessNum_ += 1;
        curIdx_ += blockIdx_;
    } else {
        curIdx_ += blockFactorTail_;
    }
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        tmpLoop_ = uVL_ / (nSize_ * BUFFER_NUM);
    } else {
        tmpLoop_ = uVL_ / nSize_;
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    int64_t processMNum = 0;
    int64_t processGNum = 0;
    int64_t processNum = 0;
    int64_t srcStride = 0;
    int64_t yGMIdx = 0;
    uint32_t gnAlignSize = 0;
    uint32_t mnAlignSize = 0;
    uint32_t blockTail = 0;
    int64_t localOffset = 0;
    int64_t ubOffset = 0;
    for (uint64_t i = 0; i < blkProcessNum_; i++) {
        mIdx_ = curIdx_ / gUBCount_;
        gIdx_ = curIdx_ % gUBCount_;
        // Calc 4 types processNum
        if (mUBCount_ > 1 && gUBCount_ > 1 && mIdx_ == mUBCount_ - 1 && gIdx_ == gUBCount_ - 1) {
            // Process last tail
            processMNum = mUBFactorTail_;
            processGNum = gUBFactorTail_;
        } else if (mUBCount_ > 1 && mIdx_ == mUBCount_ - 1) {
            // Process Mtail
            processMNum = mUBFactorTail_;
            processGNum = gUBFactor_;
        } else if (gUBCount_ > 1 && gIdx_ == gUBCount_ - 1) {
            // Process Gtail
            processMNum = mUBFactor_;
            processGNum = gUBFactorTail_;
        } else {
            // Process mainFactor
            processMNum = mUBFactor_;
            processGNum = gUBFactor_;
        }

        // Copy in
        gmOffset_ = gIdx_ * gUBFactor_ * nSize_ + mIdx_ * mUBFactor_ * gSize_ * nSize_;
        srcStride = gSize_ * nSize_ - processGNum * nSize_;
        if (processMNum < tmpLoop_) {
            tmpLoop_ = processMNum;
        }

        // Comput and Copy out
        processNum = processMNum * nSize_;
        gnAlignSize = processGNum * nSize_;
        mnAlignSize = CeilDivision(processMNum * nSize_, BLOCK_ELENUM) * BLOCK_ELENUM;
        nRegSize_ = nSize_ * tmpLoop_;

        ComputeIdx(processGNum, processNum, gnAlignSize, mnAlignSize);
        idxLocal_ = idxQueue_.DeQue<U>();
        yLocal_ = outQueueY_.AllocTensor<T>();
        LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
        CopyIn(1, processMNum * processGNum * nSize_, gmOffset_, srcStride, 0, xUb);
        LocalTensor<T> srcUb = inQueueX_.DeQue<T>();
        for (uint32_t g = 0; g < processGNum; g++) {
            Compute(g, processNum, gnAlignSize, mnAlignSize, srcUb);
            gmOffset_ = mIdx_ * mUBFactor_ * nSize_;
            yGMIdx = gIdx_ * gUBFactor_ + g;
            yGm_.SetGlobalBuffer(GetTensorAddr(yGMIdx));
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            ubOffset = g * mnAlignSize;
            CopyOut(ubOffset, 1, processMNum * nSize_, gmOffset_, 0, 0);
        }
        outQueueY_.FreeTensor(yLocal_);
        inQueueX_.FreeTensor(xUb);
        idxQueue_.FreeTensor(idxLocal_);
        curIdx_++;
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::CopyIn(int64_t blockCount, int64_t blockLen,
                                                                   int64_t srcOffset, int64_t srcStride,
                                                                   int64_t dstStride, LocalTensor<T>& xUb)
{
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = dstStride / BLOCK_ELENUM;
    DataCopyPad<T, PaddingMode::Compact>(xUb, xGm_[srcOffset], copyInParam_, padParam_);
    inQueueX_.EnQue(xUb);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::CopyOut(int64_t localOffset, int64_t blockCount,
                                                                    int64_t blockLen, int64_t dstOffset,
                                                                    int64_t srcStride, int64_t dstStride)
{
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = srcStride / BLOCK_ELENUM;
    copyOutParam_.dstStride = dstStride * dtypeSize_;
    DataCopyPad(yGm_[dstOffset], yLocal_[localOffset], copyOutParam_);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::ComputeIdx(int64_t processGNum, int64_t processMNNum,
                                                                       uint32_t gnAlignSize, uint32_t mnAlignSize)
{
    LocalTensor<U> idxUb = idxQueue_.AllocTensor<U>();
    uint32_t processNum = processMNNum;
    uint16_t loopNum = CeilDivision(processNum, nRegSize_);
    uint32_t processLastNum = nRegSize_ * loopNum == processNum ? nRegSize_ : processNum % nRegSize_;
    uint32_t mnSize = tmpLoop_ * nSize_;
    uint32_t mnSizeLast = processLastNum;
    uint32_t nRegSize = nRegSize_;
    int32_t uVL = uVL_;
    int64_t nSize = nSize_;
    int64_t tmpLoop = tmpLoop_;
    uint16_t gLoopNum = processGNum;

    if constexpr (sizeof(T) == sizeof(int64_t)) {
        uint32_t nSizeInt64 = nSize * BUFFER_NUM;
        uint32_t tmpLoop = uVL / nSizeInt64;
        nRegSize = nRegSize_ * BUFFER_NUM;
        gnAlignSize = gnAlignSize * BUFFER_NUM;
        mnAlignSize = mnAlignSize * BUFFER_NUM;
        processNum = processNum * BUFFER_NUM;
        processLastNum = processNum % nRegSize == 0 ? nRegSize : processNum % nRegSize;
        loopNum = CeilDivision(processNum, tmpLoop * nSizeInt64);
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxUb.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexRegB64;
            AscendC::Reg::RegTensor<U> tmpB64;
            AscendC::Reg::RegTensor<U> tmp1B64;
            AscendC::Reg::RegTensor<U> tmp2B64;
            AscendC::Reg::RegTensor<U> addRegB64;
            AscendC::Reg::RegTensor<U> niRegB64;
            AscendC::Reg::RegTensor<U> subRegB64;
            AscendC::Reg::MaskReg maskB64;

            maskB64 = AscendC::Reg::UpdateMask<U>(processNum);

            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexRegB64, startIdx);
            AscendC::Reg::Duplicate(niRegB64, (U)nSizeInt64, maskB64);
            AscendC::Reg::Div(tmpB64, (AscendC::Reg::RegTensor<U>&)indexRegB64, niRegB64, maskB64);
            AscendC::Reg::Muls(tmp1B64, tmpB64, (U)gnAlignSize, maskB64);
            AscendC::Reg::Mul(subRegB64, tmpB64, niRegB64, maskB64);
            AscendC::Reg::Sub(tmp2B64, (AscendC::Reg::RegTensor<U>&)indexRegB64, subRegB64, maskB64);
            AscendC::Reg::Add(addRegB64, tmp1B64, tmp2B64, maskB64);

            AscendC::Reg::DataCopy(idxPtr, addRegB64, maskB64);
        }
    } else {
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxUb.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexReg;
            AscendC::Reg::RegTensor<U> tmp;
            AscendC::Reg::RegTensor<U> tmp1;
            AscendC::Reg::RegTensor<U> tmp2;
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> niReg;
            AscendC::Reg::RegTensor<U> subReg;
            AscendC::Reg::MaskReg mask;

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexReg, startIdx);
            AscendC::Reg::Duplicate(niReg, (U)nSize, mask);
            AscendC::Reg::Div(tmp, (AscendC::Reg::RegTensor<U>&)indexReg, niReg, mask);
            AscendC::Reg::Muls(tmp1, tmp, (U)gnAlignSize, mask);
            AscendC::Reg::Mul(subReg, tmp, niReg, mask);
            AscendC::Reg::Sub(tmp2, (AscendC::Reg::RegTensor<U>&)indexReg, subReg, mask);
            AscendC::Reg::Add(addReg, tmp1, tmp2, mask);

            AscendC::Reg::DataCopy(idxPtr, addReg, mask);
        }
    }
    idxQueue_.EnQue(idxUb);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Compute(uint32_t g, int64_t processMNNum,
                                                                    uint32_t gnAlignSize, uint32_t mnAlignSize,
                                                                    LocalTensor<T>& srcUb)
{
    uint32_t processNum = processMNNum;
    uint16_t loopNum = CeilDivision(processNum, nRegSize_);
    uint32_t gnSize = g * nSize_;
    uint32_t processLastNum = nRegSize_ * loopNum == processNum ? nRegSize_ : processNum % nRegSize_;
    uint32_t mnSize = tmpLoop_ * nSize_;
    uint32_t mnSizeLast = processLastNum;
    uint32_t nRegSize = nRegSize_;
    int32_t uVL = uVL_;
    int64_t nSize = nSize_;
    int64_t tmpLoop = tmpLoop_;

    if constexpr (sizeof(T) == sizeof(int64_t)) {
        LocalTensor<U> srcUbU32 = srcUb.template ReinterpretCast<U>();
        // As 2 uint32 data concatenated to process
        uint32_t nSizeInt64 = nSize_ * BUFFER_NUM;
        gnSize = g * nSizeInt64;
        mnSize = mnSize * BUFFER_NUM;
        mnSizeLast = mnSizeLast * BUFFER_NUM;
        tmpLoop = uVL / nSizeInt64;
        nRegSize = nRegSize_ * BUFFER_NUM;
        gnAlignSize = gnAlignSize * BUFFER_NUM;
        mnAlignSize = mnAlignSize * BUFFER_NUM;
        loopNum = loopNum * BUFFER_NUM;
        processNum = processNum * BUFFER_NUM;
        processLastNum = processNum % nRegSize == 0 ? nRegSize : processNum % nRegSize;
        loopNum = CeilDivision(processNum, tmpLoop * nSizeInt64);
        __ubuf__ U* srcPtr = (__ubuf__ U*)srcUbU32.GetPhyAddr();
        __ubuf__ U* dstPtr = (__ubuf__ U*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxLocal_.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::UnalignReg uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskTmp;

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            maskTmp = AscendC::Reg::UpdateMask<U>(nRegSize);
            AscendC::Reg::DataCopy(addReg, idxPtr);
            AscendC::Reg::Adds(addReg, addReg, (U)gnSize, mask);

            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::Reg::DataCopyGather(dstReg, srcPtr, addReg, maskTmp);
                // Copy out
                AscendC::Reg::DataCopyUnAlign(dstPtr, dstReg, uDst, mnSize);
                AscendC::Reg::Adds(addReg, addReg, (U)(tmpLoop * gnAlignSize), mask);
            }
            maskTmp = AscendC::Reg::UpdateMask<U>(processLastNum);
            AscendC::Reg::DataCopyGather(dstReg, srcPtr, addReg, maskTmp);
            AscendC::Reg::DataCopyUnAlign(dstPtr, dstReg, uDst, mnSizeLast);
            AscendC::Reg::DataCopyUnAlignPost(dstPtr, uDst, 0);
        }
    } else {
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcUb.GetPhyAddr();
        __ubuf__ U* dstPtrU = (__ubuf__ U*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ T* dstPtrT = (__ubuf__ T*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxLocal_.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::RegTensor<T> dstRegT;
            AscendC::Reg::UnalignReg uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskTmp;

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            maskTmp = AscendC::Reg::UpdateMask<U>(nRegSize);
            AscendC::Reg::DataCopy(addReg, idxPtr);
            AscendC::Reg::Adds(addReg, addReg, (U)gnSize, mask);

            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::Reg::DataCopyGather(dstReg, srcPtr, addReg, maskTmp);
                // Copy out
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    // Convert B16 to B8
                    AscendC::Reg::Pack(dstRegT, dstReg);
                    AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, mnSize);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
                } else {
                    AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, mnSize);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
                }
                AscendC::Reg::Adds(addReg, addReg, (U)(tmpLoop * gnAlignSize), mask);
            }
            maskTmp = AscendC::Reg::UpdateMask<U>(processLastNum);
            AscendC::Reg::DataCopyGather(dstReg, srcPtr, addReg, maskTmp);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                // Convert B16 to B8
                AscendC::Reg::Pack(dstRegT, dstReg);
                AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, mnSizeLast);
                AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
            } else {
                AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, mnSizeLast);
                AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
            }
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline __gm__ T* SplitVUbSplitSameLenSmallG<T, U, Y>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // namespace SplitV
