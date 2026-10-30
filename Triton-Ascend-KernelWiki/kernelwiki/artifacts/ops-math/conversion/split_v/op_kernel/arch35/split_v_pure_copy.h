/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_PURE_COPY_H
#define SPLIT_V_PURE_COPY_H
#include "op_kernel/platform_util.h"
namespace SplitV {
using namespace AscendC;
const int32_t SPLIT_LIST_MAX_LEN = 72;
template <typename T, typename S>
class SplitVPureCopyMode {
public:
    __aicore__ inline SplitVPureCopyMode(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR sizeSplits, const SplitVTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline int64_t SplitPrefix(int64_t index);
    __aicore__ inline int64_t CurSplitSize(int64_t index);
    __aicore__ inline void CalcUbAndProcessCopy(int32_t i);
    __aicore__ inline void CopyArray(const int64_t* src, int64_t* dst, int32_t size);
    __aicore__ inline void CopyOutToGm(int64_t blockCount, int64_t blockLen, int64_t dstOffset, int64_t srcStride,
                                       int64_t dstStride);
    __aicore__ inline void CopyInToUb(int64_t blockCount, int64_t blockLen, int64_t srcOffset_, int64_t srcStride,
                                      int64_t dstStride);
    __aicore__ inline void CopyOfM(int64_t factor, int32_t i);
    __aicore__ inline void CopyOneTile(int64_t blockCount, int64_t blockLen, int64_t inSrcStride, int64_t outDstStride);
    __aicore__ inline int32_t min(int32_t a, int32_t b);
    __aicore__ inline void CalcCurDstOffset(int32_t i);
    __aicore__ inline void CalcNSplitSize(int32_t i);

private:
    TPipe& pipe_;
    const SplitVTilingData* tilingData_;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize(); // 一个 block 的字节数
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    TBuf<> inXBuf_;
    TBuf<> splitBuf_;
    LocalTensor<S> splitOffsetLocal_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<S> sizeSplitsGm_;
    ListTensorDesc inputList_;
    LocalTensor<T> inTensorX_;

    int64_t pingPong_ = 0; // 贯穿 Process->CalcUbAndProcessCopy->CopyOfM 三层的 tile 计数, 驱动 ping-pong

    int32_t blockIdx_ = 0;
    int64_t ubSize_ = 0;
    int64_t realCoreNum_ = 0;       // 真实使用的核数
    int64_t dtypeSize_ = sizeof(T); // 输入dtype所占的字节数
    int64_t nSplitSize_ = 0;        // 当前分块n轴的大小
    int64_t nBlockOffset_ = 0;
    int64_t mBlockOffset_ = 0;
    int64_t blockOffset_ = 0;
    int32_t numSplit_ = 0;

    int64_t nUbFactor_ = 0;     // Ub切分n轴主块大小
    int64_t nUbFactorTail_ = 0; // Ub切分n轴尾块大小
    int64_t nSplitUbTimes_ = 0; // Ub切分n轴处理次数
    int64_t nFactor_ = 0;       // block切分n轴大小

    int64_t mUbFactor_ = 0;     // Ub切分m轴主块大小
    int64_t mUbFactorTail_ = 0; // Ub切分m轴尾块大小
    int64_t mSplitUbTimes_ = 0; // Ub切分m轴处理次数
    int64_t mFactor_ = 0;       // block切分m轴大小

    int64_t ubSizeNum_ = 0; // Ub可以放个数
    int64_t startOffset_ = 0;
    int64_t splitNumPerCore_ = 0;
    int64_t srcOffset_ = 0;
    int64_t dstOffset_ = 0;
    int64_t curDstOffset_ = 0;

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};

    int32_t isNBlockMain_ = 0;
    int32_t isMBlockMain_ = 0;
    int32_t nIdx_ = 0;
    int32_t mIdx_ = 0;
    int64_t prefixBlock_ = 0;
    int64_t prefixSplitI_ = 0;
    int64_t prefixBlockBefore_ = 0;
};

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR sizeSplits,
                                                      const SplitVTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    ubSize_ = tilingData_->ubSize;
    realCoreNum_ = tilingData_->realCoreNum; // 真实使用的核数
    numSplit_ = tilingData->gSize;
    int32_t splitBufSize = Ops::Base::CeilAlign(static_cast<int32_t>(numSplit_ * sizeof(S)),
                                                static_cast<int32_t>(BLOCK_SIZE));
    pipe_.InitBuffer(splitBuf_, splitBufSize);
    int32_t ubTotal = ubSize_ - splitBufSize;
    // double buffer: 整块 UB 切成 BUFFER_NUM 份, 单份按 block 对齐; ubSizeNum_ 既是单份可放元素数,
    // 也用作 ping-pong 子块偏移 inTensorX_[(pingPong_ & 1) * ubSizeNum_]
    int32_t ubPerBuf = ubTotal / BUFFER_NUM / BLOCK_SIZE * BLOCK_SIZE;
    ubSizeNum_ = ubPerBuf / dtypeSize_;
    pipe_.InitBuffer(inXBuf_, ubPerBuf * BUFFER_NUM);
    inTensorX_ = inXBuf_.template Get<T>();
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    sizeSplitsGm_.SetGlobalBuffer((__gm__ S*)sizeSplits);
    splitOffsetLocal_ = splitBuf_.template Get<S>();
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

    nIdx_ = blockIdx_ % tilingData_->nBlockCount;
    mIdx_ = blockIdx_ / tilingData_->nBlockCount;
    isNBlockMain_ = (nIdx_ < tilingData_->nBlockFactorNum) ? 1 : 0;
    isMBlockMain_ = (mIdx_ < tilingData_->mBlockFactorNum) ? 1 : 0;
    nBlockOffset_ = isNBlockMain_ == 1 ? nIdx_ * tilingData_->nBlockFactor :
                                         tilingData_->nBlockFactorNum * tilingData_->nBlockFactor +
                                             (nIdx_ - tilingData_->nBlockFactorNum) * tilingData_->nBlockFactorTail;
    mBlockOffset_ = isMBlockMain_ == 1 ? mIdx_ * tilingData_->mBlockFactor :
                                         tilingData_->mBlockFactorNum * tilingData_->mBlockFactor +
                                             (mIdx_ - tilingData_->mBlockFactorNum) * tilingData_->mBlockFactorTail;
    splitNumPerCore_ = tilingData_->nBlockSplitOffsetEnd[blockIdx_] - tilingData_->nBlockSplitOffset[blockIdx_];
    blockOffset_ = mBlockOffset_ * tilingData_->nSize + nBlockOffset_;
    mFactor_ = isMBlockMain_ == 1 ? tilingData_->mBlockFactor : tilingData_->mBlockFactorTail;
    nFactor_ = isNBlockMain_ == 1 ? tilingData_->nBlockFactor : tilingData_->nBlockFactorTail;
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<S> padParamsIdx = {false, 0, 0, 0};
    copyParams.blockCount = 1;
    copyParams.blockLen = numSplit_ * sizeof(S);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(splitOffsetLocal_, sizeSplitsGm_, copyParams, padParamsIdx);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
    AscendC::Muls(splitOffsetLocal_, splitOffsetLocal_, tilingData_->sizeAfterSplitDim, numSplit_);
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID1);
    WaitFlag<HardEvent::V_S>(eventID1);
    prefixBlock_ = tilingData_->nBlockSplitPrefixStart[blockIdx_];
    prefixBlockBefore_ = tilingData_->nBlockSplitOffset[blockIdx_] > 0 ?
                             tilingData_->nBlockSplitPrefixStart[blockIdx_] -
                                 CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_]) :
                             0;
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    for (int32_t i = 0; i < splitNumPerCore_; i++) {
        yGm_.SetGlobalBuffer(GetTensorAddr(tilingData_->nBlockSplitOffset[blockIdx_] + i));
        prefixSplitI_ = (tilingData_->nBlockSplitOffset[blockIdx_] + i) > 0 ?
                            SplitPrefix(tilingData_->nBlockSplitOffset[blockIdx_] + i - 1) :
                            0;
        CalcNSplitSize(i);
        // 计算n轴ub切分参数
        nSplitUbTimes_ = (nSplitSize_ + ubSizeNum_ - 1) / ubSizeNum_;
        nUbFactor_ = nSplitSize_ >= ubSizeNum_ ? ubSizeNum_ : 0;
        nUbFactorTail_ = nSplitSize_ - nUbFactor_ * (nSplitUbTimes_ - 1);
        // 计算m轴ub切分参数
        mUbFactor_ = nSplitSize_ >= ubSizeNum_ ?
                         1 :
                         min(mFactor_,
                             ubSizeNum_ / (((nUbFactorTail_ + BLOCK_ELENUM - 1) / BLOCK_ELENUM) * BLOCK_ELENUM));
        mSplitUbTimes_ = (mFactor_ + mUbFactor_ - 1) / mUbFactor_;
        mUbFactorTail_ = mFactor_ - mUbFactor_ * (mSplitUbTimes_ - 1);
        CalcUbAndProcessCopy(i);
    }
    // 收尾: 清空最后 BUFFER_NUM 个仍在途、未被后续复用 wait 消费的 out event, 避免 event 悬挂
    int64_t drain = pingPong_ < BUFFER_NUM ? pingPong_ : BUFFER_NUM;
    for (int64_t k = 0; k < drain; k++) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(EVENT_ID2 + ((pingPong_ - drain + k) & 1)));
    }
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CalcNSplitSize(int32_t i)
{
    if (i == 0) {
        startOffset_ = blockOffset_;
        nSplitSize_ = min(prefixBlock_ - nBlockOffset_, nFactor_);
    } else if (i < splitNumPerCore_ - 1) {
        startOffset_ = mBlockOffset_ * tilingData_->nSize + prefixSplitI_;
        nSplitSize_ = CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_] + i);
    } else {
        startOffset_ = mBlockOffset_ * tilingData_->nSize + prefixSplitI_;
        nSplitSize_ = nBlockOffset_ + nFactor_ - prefixSplitI_;
    }
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CalcUbAndProcessCopy(int32_t i)
{
    srcOffset_ = startOffset_; // 更新Split切分块初始srcOffset_
    CalcCurDstOffset(i);
    dstOffset_ = curDstOffset_;
    // double buffer 后 tile 间依赖由 CopyOneTile 内的 ping-pong 复用同步统一接管, 此处不再插串行栅栏
    for (int32_t mIdx = 0; mIdx < mSplitUbTimes_ - 1; mIdx++) {
        // ub主块搬运
        srcOffset_ = startOffset_ + mUbFactor_ * tilingData_->nSize * mIdx;
        dstOffset_ = curDstOffset_ + mUbFactor_ * CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_] + i) * mIdx;
        CopyOfM(mUbFactor_, i);
    }
    // ub尾块搬运
    srcOffset_ = startOffset_ + mUbFactor_ * tilingData_->nSize * (mSplitUbTimes_ - 1);
    dstOffset_ = curDstOffset_ +
                 mUbFactor_ * CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_] + i) * (mSplitUbTimes_ - 1);
    CopyOfM(mUbFactorTail_, i);
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CopyOneTile(int64_t blockCount, int64_t blockLen, int64_t inSrcStride,
                                                             int64_t outDstStride)
{
    // buffer 复用依赖: 两个 tile 前用同一 buffer 的 out 必须已完成才能覆盖写入
    event_t reuseEvt = static_cast<event_t>(EVENT_ID2 + (pingPong_ & 1));
    if (pingPong_ >= BUFFER_NUM) {
        WaitFlag<HardEvent::MTE3_MTE2>(reuseEvt);
    }
    CopyInToUb(blockCount, blockLen, srcOffset_, inSrcStride, 0);
    // 同 buffer in->out 硬依赖
    event_t ioEvt = static_cast<event_t>(EVENT_ID0 + (pingPong_ & 1));
    SetFlag<HardEvent::MTE2_MTE3>(ioEvt);
    WaitFlag<HardEvent::MTE2_MTE3>(ioEvt);
    CopyOutToGm(blockCount, blockLen, dstOffset_, 0, outDstStride);
    // 只 set 不立即 wait: 延迟到下一个复用同一 buffer 的 tile 前, 让相邻 tile 的 in/out 跨 buffer 重叠
    SetFlag<HardEvent::MTE3_MTE2>(reuseEvt);
    pingPong_++;
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CopyOfM(int64_t factor, int32_t i)
{
    int64_t curSplitSize = CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_] + i);

    for (int32_t nIdx = 0; nIdx < nSplitUbTimes_ - 1; nIdx++) {
        CopyOneTile(factor, nUbFactor_, tilingData_->nSize - nUbFactor_, curSplitSize - nUbFactor_);
        srcOffset_ = srcOffset_ + nUbFactor_;
        dstOffset_ = dstOffset_ + nUbFactor_;
    }
    CopyOneTile(factor, nUbFactorTail_, tilingData_->nSize - nUbFactorTail_, curSplitSize - nUbFactorTail_);
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CalcCurDstOffset(int32_t i)
{
    int64_t curCoreNDstOffset = 0;
    int64_t curCoreMDstOffset = mBlockOffset_;
    if (i == 0) {
        curCoreNDstOffset = nBlockOffset_ - prefixBlockBefore_;
    } else {
        curCoreNDstOffset = 0;
    }
    curDstOffset_ = curCoreMDstOffset * CurSplitSize(tilingData_->nBlockSplitOffset[blockIdx_] + i) + curCoreNDstOffset;
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CopyInToUb(int64_t blockCount, int64_t blockLen, int64_t srcOffset_,
                                                            int64_t srcStride, int64_t dstStride)
{
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = 0;
    DataCopyPad(inTensorX_[(pingPong_ & 1) * ubSizeNum_], xGm_[srcOffset_], copyInParam_, padParam_);
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CopyOutToGm(int64_t blockCount, int64_t blockLen, int64_t dstOffset,
                                                             int64_t srcStride, int64_t dstStride)
{
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = 0;
    copyOutParam_.dstStride = dstStride * dtypeSize_;
    DataCopyPad(yGm_[dstOffset], inTensorX_[(pingPong_ & 1) * ubSizeNum_], copyOutParam_);
}

template <typename T, typename S>
__aicore__ inline __gm__ T* SplitVPureCopyMode<T, S>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

template <typename T, typename S>
__aicore__ inline int64_t SplitVPureCopyMode<T, S>::SplitPrefix(int64_t index)
{
    int64_t tensorSize = 0;
    for (int64_t jj = 0; jj < index + 1; jj++) {
        if (jj >= 0 && jj < numSplit_) {
            tensorSize = tensorSize + splitOffsetLocal_.GetValue(jj);
        }
    }
    return tensorSize;
}

template <typename T, typename S>
__aicore__ inline int64_t SplitVPureCopyMode<T, S>::CurSplitSize(int64_t index)
{
    int64_t tensorSize = 0;
    if (index >= 0 && index < numSplit_) {
        tensorSize = splitOffsetLocal_.GetValue(index);
    }

    return tensorSize;
}

template <typename T, typename S>
__aicore__ inline void SplitVPureCopyMode<T, S>::CopyArray(const int64_t* src, int64_t* dst, int32_t size)
{
    for (int32_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template <typename T, typename S>
__aicore__ inline int32_t SplitVPureCopyMode<T, S>::min(int32_t a, int32_t b)
{
    return a > b ? b : a;
}
} // namespace SplitV
#endif
