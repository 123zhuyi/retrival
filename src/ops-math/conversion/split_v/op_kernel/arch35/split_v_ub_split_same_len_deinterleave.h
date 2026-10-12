/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
#define SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;
template <typename T>
class SplitVUbSplitSameLenDeinterleave {
public:
    __aicore__ inline SplitVUbSplitSameLenDeinterleave(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void CopyIn(int64_t blockLen, int64_t srcOffset);
    __aicore__ inline void CopyOut(LocalTensor<T>& yLocal, int64_t blockLen, int64_t dstOffset);

private:
    TPipe& pipe_;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    const SplitVTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueY0_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int64_t dtypeSize_ = sizeof(T);
    int64_t gSize_ = 0;
    int64_t nSize_ = 0;
    int64_t mUBCount_ = 0;
    int64_t mUBFactor_ = 0;
    int64_t mUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;
    int64_t curIdx_ = 0;
    int64_t mIdx_ = 0;
};

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::Init(GM_ADDR x, GM_ADDR y,
                                                                 const SplitVTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    gSize_ = tilingData_->gSize;
    nSize_ = tilingData_->nBlockFactorNum;
    mUBFactor_ = tilingData_->mBlockFactor;
    mUBFactorTail_ = tilingData_->mBlockFactorTail;
    mUBCount_ = tilingData_->mBlockCount;
    blockFactor_ = tilingData_->blockFactor;
    blockFactorTail_ = tilingData_->blockFactorTail;

    uint32_t inSpace = gSize_ * mUBFactor_ * dtypeSize_;
    uint32_t outSpace = CeilDivision(mUBFactor_, BLOCK_ELENUM) * BLOCK_ELENUM * dtypeSize_;
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, inSpace);
    pipe_.InitBuffer(outQueueY0_, BUFFER_NUM, outSpace);
    pipe_.InitBuffer(outQueueY1_, BUFFER_NUM, outSpace);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

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
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    int64_t processMNum = 0;
    int64_t srcCount = 0;
    for (uint64_t i = 0; i < blkProcessNum_; i++) {
        mIdx_ = curIdx_;
        processMNum = (mIdx_ == mUBCount_ - 1) ? mUBFactorTail_ : mUBFactor_;
        srcCount = processMNum * gSize_;

        int64_t gmOffset = mIdx_ * mUBFactor_ * gSize_ * nSize_;
        CopyIn(srcCount, gmOffset);
        LocalTensor<T> srcLocal = inQueueX_.DeQue<T>();

        LocalTensor<T> yLocal0 = outQueueY0_.AllocTensor<T>();
        LocalTensor<T> yLocal1 = outQueueY1_.AllocTensor<T>();
        AscendC::DeInterleave(yLocal0, yLocal1, srcLocal, srcCount);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        int64_t dstOffset = mIdx_ * mUBFactor_ * nSize_;
        yGm_.SetGlobalBuffer(GetTensorAddr(0));
        CopyOut(yLocal0, processMNum, dstOffset);
        yGm_.SetGlobalBuffer(GetTensorAddr(1));
        CopyOut(yLocal1, processMNum, dstOffset);

        outQueueY0_.FreeTensor(yLocal0);
        outQueueY1_.FreeTensor(yLocal1);
        inQueueX_.FreeTensor(srcLocal);
        curIdx_++;
    }
}

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::CopyIn(int64_t blockLen, int64_t srcOffset)
{
    LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyInParam{1, static_cast<uint32_t>(blockLen * dtypeSize_), 0, 0, 0};
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[srcOffset], copyInParam, padParam);
    inQueueX_.EnQue(xUb);
}

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::CopyOut(LocalTensor<T>& yLocal, int64_t blockLen,
                                                                    int64_t dstOffset)
{
    DataCopyExtParams copyOutParam{1, static_cast<uint32_t>(blockLen * dtypeSize_), 0, 0, 0};
    DataCopyPad(yGm_[dstOffset], yLocal, copyOutParam);
}

template <typename T>
__aicore__ inline __gm__ T* SplitVUbSplitSameLenDeinterleave<T>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
