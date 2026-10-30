/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_PURE_COPY_SPECIAL_H
#define SPLIT_V_PURE_COPY_SPECIAL_H
#include "op_kernel/platform_util.h"

namespace SplitV {
using namespace AscendC;

template <typename T> class SplitVPureCopySpecialMode {
public:
    __aicore__ inline SplitVPureCopySpecialMode(TPipe &pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index);
    __aicore__ inline int64_t SplitPrefix(int64_t index);
    __aicore__ inline int64_t CurSplitSize(int64_t index);
    __aicore__ inline void ProcessSpecialCopy();
    __aicore__ inline void CopyOutToGm(int64_t blockCount, int64_t blockLen, int64_t dstOffset);
    __aicore__ inline void CopyInToUb(int64_t blockCount, int64_t blockLen, int64_t srcOffset);

private:
    constexpr static int32_t bufferNum = 1;
    TPipe &pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, bufferNum> inQueueX_;
    ListTensorDesc inputList_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    const SplitVTilingData *tilingData_;
    int64_t ubSize_ = 0;
    int32_t blockIdx_ = 0;
    int64_t splitDim_ = 0;          // split的轴
    int64_t realCoreNum_ = 0;       // 真实使用的核数
    int64_t sizeAfterSplitDim_ = 0; // split轴之后相同shape的乘积
    int64_t dtypeSize_ = sizeof(T); // 输入dtype所占的字节数
    int64_t nSize_ = 0;             // 当前分块n轴的大小

    int64_t nBlockFactorNum_ = 0;  // n轴主块的个数
    int64_t nBlockFactor_ = 0;     // n轴切分factor
    int64_t nBlockFactorTail_ = 0; // n轴切分factor尾块；
    int64_t nBlockCount_ = 1;      // n轴切分块数
    int64_t nBlockOffset_ = 0;

    int64_t nUbFactor_ = 1;     // Ub切分n轴主块大小
    int64_t nUbFactorTail_ = 0; // Ub切分n轴尾块大小
    int64_t nGmToUbTimes_ = 0;  // Ub切分n轴处理次数

    int64_t mUbFactor_ = 0;     // Ub切分m轴主块大小
    int64_t mUbFactorTail_ = 0; // Ub切分m轴尾块大小
    int64_t mGmToUbTimes_ = 0;  // Ub切分m轴处理次数

    int64_t ubSizeNum_ = 1; // Ub可以放个数
    int64_t ubCutNumPerCore_ = 0;
    int64_t curSrcOffset_ = 0;
    int64_t pureOutIdx_ = 0;

    DataCopyExtParams copyInParam_{ 0, 0, 0, 0, 0 };
    DataCopyPadExtParams<T> padParam_{ false, 0, 0, 0 };
    DataCopyExtParams copyOutParam_{ 0, 0, 0, 0, 0 };

    int32_t isNBlockMain_ = 0;
    int32_t nIdx_ = 0;
};

template <typename T>
__aicore__ inline void SplitVPureCopySpecialMode<T>::Init(GM_ADDR x, GM_ADDR y, const SplitVTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    ubSize_ = tilingData_->ubSize;
    nBlockFactorNum_ = tilingData_->nBlockFactorNum;
    splitDim_ = tilingData_->splitDim;                   // split的轴
    sizeAfterSplitDim_ = tilingData_->sizeAfterSplitDim; // split轴之后相同shape的乘积
    nBlockFactor_ = tilingData_->nBlockFactor;           // n轴切分factor
    nBlockFactorTail_ = tilingData_->nBlockFactorTail;   // n轴切分factor尾快
    nBlockCount_ = tilingData_->nBlockCount;             // n轴切分块数
    pureOutIdx_ = tilingData_->pureOutIdx;               // 纯搬运特殊模板输出索引
    realCoreNum_ = tilingData_->realCoreNum;             // 真实使用的核数
    ubSizeNum_ = ubSize_ / dtypeSize_;
    pipe_.InitBuffer(inQueueX_, bufferNum, ubSize_);
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void *>(y));

    nIdx_ = blockIdx_ % nBlockCount_;
    isNBlockMain_ = (nIdx_ < nBlockFactorNum_) ? 1 : 0;
}

template <typename T>
__aicore__ inline void SplitVPureCopySpecialMode<T>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    if (isNBlockMain_ == 1) {
        curSrcOffset_ = blockIdx_ * nBlockFactor_;
        nSize_ = nBlockFactor_;
    } else {
        curSrcOffset_ = nBlockFactorNum_ * nBlockFactor_ + (blockIdx_ - nBlockFactorNum_) * nBlockFactorTail_;
        nSize_ = nBlockFactorTail_;
    }
    nGmToUbTimes_ = ubSizeNum_ >= nSize_ ? 1 : (nSize_ + ubSizeNum_ - 1) / ubSizeNum_;
    ProcessSpecialCopy();
}

template <typename T>
__aicore__ inline void SplitVPureCopySpecialMode<T>::ProcessSpecialCopy()
{
    nUbFactor_ = ubSizeNum_ >= nSize_ ? nSize_ : ubSizeNum_;
    nUbFactorTail_ = nSize_ % nUbFactor_;
    yGm_.SetGlobalBuffer(GetTensorAddr(pureOutIdx_));

    // 纯搬运场景中的特殊场景, 展成1维n轴方向搬运
    for (int32_t j = 0; j < nGmToUbTimes_; j++) {
        curSrcOffset_ += ((j == 0) ? 0 : nUbFactor_);
        if (nUbFactorTail_ != 0 && j == nGmToUbTimes_ - 1) {
            // 处理Ub切分Tail数据
            CopyInToUb(1, nUbFactorTail_, curSrcOffset_);
            CopyOutToGm(1, nUbFactorTail_, curSrcOffset_);
        } else {
            CopyInToUb(1, nUbFactor_, curSrcOffset_);
            CopyOutToGm(1, nUbFactor_, curSrcOffset_);
        }
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopySpecialMode<T>::CopyInToUb(int64_t blockCount, int64_t blockLen, int64_t srcOffset)
{
    LocalTensor<T> srcLocal = inQueueX_.AllocTensor<T>();
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = 0;
    copyInParam_.dstStride = 0;
    DataCopyPad(srcLocal, xGm_[srcOffset], copyInParam_, padParam_);
    inQueueX_.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void SplitVPureCopySpecialMode<T>::CopyOutToGm(int64_t blockCount, int64_t blockLen,
                                            int64_t dstOffset)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = 0;
    copyOutParam_.dstStride = 0;
    DataCopyPad(yGm_[dstOffset], xLocal, copyOutParam_);
    inQueueX_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline __gm__ T *SplitVPureCopySpecialMode<T>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

template <typename T>
__aicore__ inline int64_t SplitVPureCopySpecialMode<T>::SplitPrefix(int64_t index)
{
    TensorDesc<T> desc;
    uint64_t buf[10] = {0};
    int64_t tensorSize = 0;
    desc.SetShapeAddr(&buf[0]);
    for (int64_t k = 0; k < index + 1; k++) {
        inputList_.GetDesc(desc, k);
        tensorSize = tensorSize + sizeAfterSplitDim_ * desc.GetShape(splitDim_);
    }
    return tensorSize;
}

template <typename T>
__aicore__ inline int64_t SplitVPureCopySpecialMode<T>::CurSplitSize(int64_t index)
{
    TensorDesc<T> descSpecial;
    uint64_t buf[10] = {0};
    descSpecial.SetShapeAddr(&buf[0]);
    inputList_.GetDesc(descSpecial, index);
    int64_t tensorSize = sizeAfterSplitDim_ * descSpecial.GetShape(splitDim_);
    return tensorSize;
}

} // namespace Split V
#endif