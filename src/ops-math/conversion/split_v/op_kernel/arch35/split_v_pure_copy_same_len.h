/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_PURE_COPY_SAME_LEN_H
#define SPLIT_V_PURE_COPY_SAME_LEN_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;
template <typename T> class SplitVPureCopyModeSameLen {
public:
    __aicore__ inline SplitVPureCopyModeSameLen(TPipe &pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index);
    __aicore__ inline void ProcessWithCutN();
    __aicore__ inline void ProcessWithCutMG();
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t srcOffset,
                                  int64_t ubOffset, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void CopyOut(int64_t yGMIdx, int64_t blockCount, int64_t blockLen, int64_t dstOffset,
                                   int64_t ubOffset, int64_t srcStride, int64_t dstStride);
    __aicore__ inline int64_t CalcGMOffset();

private:
    TPipe &pipe_;
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    const SplitVTilingData *tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueueX_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<T> xLocal_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int64_t dtypeSize_ = sizeof(T);
    int64_t mSize_ = 0;
    int64_t gSize_ = 0;
    int64_t nSize_ = 0;
    int64_t mUBCount_ = 0;
    int64_t mUBFactor_ = 0;
    int64_t mUBFactorTail_ = 0;
    int64_t gUBCount_ = 0;
    int64_t gUBFactor_ = 0;
    int64_t gUBFactorTail_ = 0;
    int64_t nUBCount_ = 0;
    int64_t nUBFactor_ = 0;
    int64_t nUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;

    // Record global loopTime
    int64_t curIdx_ = 0;
    int64_t mIdx_ = 0;
    int64_t gIdx_ = 0;
    int64_t nIdx_ = 0;
    int64_t gmOffset_ = 0;

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};
    LoopModeParams loopMode_;
};

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData *tilingData) {
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    mSize_ = tilingData_->mBlockFactorNum;
    gSize_ = tilingData_->gSize;
    nSize_ = tilingData_->nBlockFactorNum;
    mUBFactor_ = tilingData_->mBlockFactor;
    mUBFactorTail_ = tilingData_->mBlockFactorTail;
    mUBCount_ = tilingData_->mBlockCount;
    gUBFactor_ = tilingData_->gUBFactor;
    gUBFactorTail_ = tilingData_->gUBFactorTail;
    gUBCount_ = tilingData_->gUBCount;
    nUBFactor_ = tilingData_->nBlockFactor;
    nUBFactorTail_ = tilingData_->nBlockFactorTail;
    nUBCount_ = tilingData_->nBlockCount;
    blockFactor_ = tilingData_->blockFactor;
    blockFactorTail_ = tilingData_->blockFactorTail;

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tilingData_->ubSize / BUFFER_NUM);
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void *>(y));

    // Calc start idx per core
    blkProcessNum_ = blockFactor_;
    curIdx_ = blockIdx_ * blockFactor_;
    if (blockIdx_ < blockFactorTail_) {
        blkProcessNum_ += 1;
        curIdx_ += blockIdx_;
    } else {
        curIdx_ += blockFactorTail_;
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::Process() {
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    if (nUBCount_ > 1) {
        ProcessWithCutN();   // UB only cut axisN
    } else {
        ProcessWithCutMG();  // UB cut axisM and axisG
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::ProcessWithCutN() {
    int64_t processNNum = 0;
    int64_t yGMIdx = 0;
    int64_t mgIdx = 0;
    loopMode_.loop1Size = 1;
    loopMode_.loop2Size = 1;
    for (uint64_t i = 0; i < blkProcessNum_; i++) {
        // Process curent nFactor
        nIdx_ = curIdx_ % nUBCount_;
        processNNum = (nIdx_ == nUBCount_ - 1) ? nUBFactorTail_ : nUBFactor_;
        gmOffset_ = CalcGMOffset();
        CopyIn(1, processNNum, gmOffset_, 0, 0, 0);
        xLocal_ = inQueueX_.DeQue<T>();
        mgIdx = curIdx_ / nUBCount_;
        yGMIdx = mgIdx % gSize_;
        // Calc copyOut offset: mDirectionOffset + nDirectionOffset
        gmOffset_ = mgIdx / gUBCount_ * nSize_ + nIdx_ * nUBFactor_;
        CopyOut(yGMIdx, 1, processNNum, gmOffset_, 0, 0, 0);
        inQueueX_.FreeTensor(xLocal_);
        curIdx_++;
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::ProcessWithCutMG() {
    int64_t processMNum = 0;
    int64_t processGNum = 0;
    int64_t yGMIdx = 0;
    int64_t srcStride = 0;
    int64_t dstStride = 0;
    int64_t blockCount = 0;
    int64_t nAlignSize = Ops::Base::CeilAlign(nSize_, BLOCK_NUM);
    int64_t ubOffset = 0;
    int64_t blockLen = 0;
    for (uint64_t i = 0; i < blkProcessNum_; i++) {
        // Used for cutting M/G, when No is 1
        mIdx_ = curIdx_ / gUBCount_;
        gIdx_ = curIdx_ - mIdx_ * gUBCount_;
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

        gmOffset_ = gIdx_ * gUBFactor_ * nSize_ + mIdx_ * mUBFactor_ * gSize_ * nSize_;
        srcStride = 0;
        dstStride = nAlignSize * (processMNum - 1);
        blockCount = processGNum;
        blockLen = nSize_;
        loopMode_.loop1Size = processMNum;
        loopMode_.loop2Size = 1;
        loopMode_.loop1SrcStride = gSize_ * nSize_ * dtypeSize_;
        loopMode_.loop1DstStride = nAlignSize * dtypeSize_;
        CopyIn(blockCount, blockLen, gmOffset_, 0, srcStride, dstStride);
        xLocal_ = inQueueX_.DeQue<T>();
        for (uint64_t g = 0; g < processGNum; g++) {
            // CopyOut offset —— axisM
            gmOffset_ = mIdx_ * mUBFactor_ * nSize_;
            yGMIdx = gIdx_ * gUBFactor_ + g;
            ubOffset = nAlignSize * processMNum * g;
            CopyOut(yGMIdx, processMNum, nSize_, gmOffset_, ubOffset, 0, 0);
        }
        inQueueX_.FreeTensor(xLocal_);
        curIdx_++;
    }
}

template <typename T>
__aicore__ inline int64_t SplitVPureCopyModeSameLen<T>::CalcGMOffset() {
    int64_t offset = 0;
    // UBouter: [Mo, Go, No], UBinner: [Mi, Gi, Ni]
    int64_t gRSize = nUBCount_;
    int64_t mRSize = gUBCount_ * nUBCount_;
    // Src:[M, G, N] ==> Dst:[G, M, N]
    int64_t gASize = nSize_;
    int64_t mASize = gSize_ * nSize_;
    offset = curIdx_ % nUBCount_ * nUBFactor_ + 
             curIdx_ / gRSize % gUBCount_ * gUBFactor_ * gASize + 
             curIdx_ / mRSize * mUBFactor_ * mASize;
    return offset;
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::CopyIn(int64_t blockCount, int64_t blockLen, int64_t srcOffset,
                                                            int64_t ubOffset, int64_t srcStride, int64_t dstStride) {
    LocalTensor<T> srcLocal = inQueueX_.AllocTensor<T>();
    SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = dstStride * dtypeSize_ / BLOCK_SIZE;
    DataCopyPad(srcLocal[ubOffset], xGm_[srcOffset], copyInParam_, padParam_);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inQueueX_.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::CopyOut(int64_t yGMIdx, int64_t blockCount, int64_t blockLen,
                                                             int64_t dstOffset, int64_t ubOffset, int64_t srcStride,
                                                             int64_t dstStride) {
    yGm_.SetGlobalBuffer(GetTensorAddr(yGMIdx));
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = srcStride * dtypeSize_ / BLOCK_SIZE;
    copyOutParam_.dstStride = dstStride * dtypeSize_;
    DataCopyPad(yGm_[dstOffset], xLocal_[ubOffset], copyOutParam_);
}

template <typename T>
__aicore__ inline __gm__ T *SplitVPureCopyModeSameLen<T>::GetTensorAddr(int64_t index) {
    return inputList_.GetDataPtr<T>(index);
}

} // namespace Split V
#endif