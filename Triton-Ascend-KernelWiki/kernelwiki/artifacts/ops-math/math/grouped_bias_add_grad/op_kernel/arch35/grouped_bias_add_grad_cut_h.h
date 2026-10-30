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
 * \file grouped_bias_add_grad_cut_h.h
 * \brief grouped_bias_add_grad_cut_h
 */

#ifndef OPS_MATH_GROUPED_BIAS_ADD_GRAD_CUT_H
#define OPS_MATH_GROUPED_BIAS_ADD_GRAD_CUT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "grouped_bias_add_grad_struct.h"
#include "group_utils.h"

namespace GroupedBiasAddGrad {
using namespace AscendC;

// T: grad_y data type, U: group_idx type
template <typename T, typename U>
class GroupedBiasAddGradSplitH {
    using PromoteDataT = float; // DataType in compute

    constexpr static int64_t BLOCK_BYTE = 512; // Processed block: 512B
    constexpr static uint64_t ALIGN_BYTE = 32;
    constexpr static int64_t BLOCK_SIZE = BLOCK_BYTE / sizeof(T);     // Processed block length: 512B
    constexpr static uint64_t BLOCK_UB_SIZE = ALIGN_BYTE / sizeof(T); // Aligned block length
    constexpr static float ZERO_VALUE = 0.0;
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = Ops::Base::GetVRegSize() / sizeof(PromoteDataT);
    constexpr static int64_t BUFFER_NUM = 4;

#ifdef __CCE_AICORE__
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
#endif

public:
    __aicore__ inline GroupedBiasAddGradSplitH(){};
    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,
                                const GroupedBiasAddGradCutHTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn(int64_t offset, int64_t curDimH, int64_t blkCount);
    __aicore__ inline void CopyDataInForOne(int64_t offset, int64_t curDimH, int64_t blkCount);
    __aicore__ inline void CopyInGroup();
    __aicore__ inline void CopyDataOut(int64_t outputDataOffset, int64_t dimH);
    __aicore__ inline void CopyDataOutForBuf(int64_t outputDataOffset, LocalTensor<PromoteDataT>& ubRes, int64_t dimH);
    __aicore__ inline void CopyDataOutUnCast(int64_t outputDataOffset, int64_t dimH);
    __aicore__ inline void BinaryReduceSum(int64_t loop, int64_t singleH);
    __aicore__ inline void ProcessZeroG(int64_t curH, int64_t startIndex, int64_t gId);
    __aicore__ inline void ProcessOneG(int64_t curH, int64_t startIndex, int64_t gId);
    __aicore__ inline void ProcessSplitG(int64_t curH);
    __aicore__ inline void ProcessSplitH(int64_t curH, int64_t g, int64_t gId);
    __aicore__ inline void SetGParam(int64_t singleH);
    __aicore__ inline void UpdateCacheAux(const int64_t cacheID, const int64_t stride, const int64_t count);

private:
    const GroupedBiasAddGradCutHTilingData* tilingData_;
    TPipe pipe_;

    // grad_y：input，grad_idx：group idx info，grad_bias：bias grad
    GlobalTensor<T> gradYGm_;    // grad_y
    GlobalTensor<U> gradIdxGm_;  // grad_idx
    GlobalTensor<T> gradBiasGm_; // output

    // basic param
    int64_t blockIdx_; // core id
    int64_t curH_;
    int64_t groupIdxDim_;    // group num
    int64_t inputShape_[2];  // input shape
    int64_t hMainFactor_;    // H main block length
    int64_t hTailFactor_;    // H tail block length
    int64_t usedUbSize_;     // used ub size
    int64_t useTempBufSize_; // tempBuf size
    int64_t groupedIdxSize_;
    int64_t outputSize_;     // output size
    int64_t maxPerHLen_ = 0; // single max h
    int64_t srcIndex_;       // CopyIn start addr
    int64_t startIndex_;     // core start addr
    bool groupIdxType_;
    int64_t cacheStride_ = 0;

    // G and H Que
    TQue<TPosition::VECIN, 1> groupIdxQue_;
    TQue<TPosition::VECIN, 1> xQue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> outputQue_; // output
    TQue<TPosition::VECOUT, 1> zeroOutQue_;                      // dup zero

    // cut G param
    int64_t g_;            // cur G length
    int64_t gId_;          // cur G id
    int64_t blockGNum_;    // cut cur G num
    int64_t mainG_;        // main G length
    int64_t tailG_;        // tail G length
    int64_t bisectionPos_; // nearest pow 2
    int64_t cacheCount_;
    int64_t bisectionTail_; // blockGNum_ - bisectionPos_

    TBuf<> computeResBuf_;
    TBuf<> tempResBuf_;
    TBuf<> tempBufBuf_;

    LocalTensor<U> groupIdx_;
    LocalTensor<PromoteDataT> xTensor_;
    LocalTensor<T> xTensorT_;

    // BinaryReduceSum space
    LocalTensor<PromoteDataT> computeRes_; // reduceSum compute result
    LocalTensor<PromoteDataT> tempRes_;    // temp result
    LocalTensor<PromoteDataT> tempBuf_;
};

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::Init(GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,
                                                            const GroupedBiasAddGradCutHTilingData* tiling)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tiling;

    // init
    groupIdxType_ = tilingData_->groupIdxType;
    hMainFactor_ = tilingData_->blockFactor * BLOCK_BYTE / sizeof(T);
    hTailFactor_ = tilingData_->hTailFactor;
    curH_ = hMainFactor_;

    if (blockIdx_ == GetBlockNum() - 1) {
        // tail core H
        curH_ = hTailFactor_;
        if (tilingData_->blockTailFactor > 1) {
            curH_ += (tilingData_->blockTailFactor - 1) * BLOCK_BYTE / sizeof(T);
        }
    }

    groupIdxDim_ = tilingData_->groupIdxDim;
    inputShape_[0] = tilingData_->inputShape[0];
    inputShape_[1] = tilingData_->inputShape[1];

    maxPerHLen_ = tilingData_->maxOutputElements;
    usedUbSize_ = tilingData_->useUbSize;
    groupedIdxSize_ = tilingData_->groupedIdxSize;
    outputSize_ = tilingData_->outputSize;
    useTempBufSize_ = tilingData_->useTempBuf;

    gradYGm_.SetGlobalBuffer((__gm__ T*)grad_y);
    gradIdxGm_.SetGlobalBuffer((__gm__ U*)group_idx);
    gradBiasGm_.SetGlobalBuffer((__gm__ T*)grad_bias);

    pipe_.InitBuffer(groupIdxQue_, 1, groupedIdxSize_);
    pipe_.InitBuffer(xQue_, BUFFER_NUM, usedUbSize_);
    pipe_.InitBuffer(tempResBuf_, BLOCK_BYTE * 2);
    pipe_.InitBuffer(computeResBuf_, BLOCK_BYTE * 2);
    pipe_.InitBuffer(tempBufBuf_, useTempBufSize_);
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputSize_);
    pipe_.InitBuffer(zeroOutQue_, BUFFER_NUM, outputSize_);

    computeRes_ = computeResBuf_.template Get<PromoteDataT>();
    tempBuf_ = tempBufBuf_.template Get<PromoteDataT>();
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyInGroup()
{
    // CopyIn group info
    groupIdx_ = groupIdxQue_.AllocTensor<U>();
    DataCopyPadExtParams<U> dataCopyPadParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyInParams{1, static_cast<uint32_t>(groupIdxDim_ * sizeof(U)), 0, 0, 0};
    DataCopyPad(groupIdx_, gradIdxGm_, dataCopyInParams, dataCopyPadParams);

    groupIdxQue_.EnQue<U>(groupIdx_);
    groupIdx_ = groupIdxQue_.DeQue<U>();
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyDataInForOne(int64_t offset, int64_t curDimH,
                                                                        int64_t blkCount)
{
    xTensorT_ = outputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> copyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCopyParams{0, 0, 0, 0, 0};
    dataCopyParams.blockCount = blkCount;
    dataCopyParams.blockLen = curDimH * sizeof(T);
    dataCopyParams.srcStride = (inputShape_[1] - curDimH) * sizeof(T);
    dataCopyParams.dstStride = 0;

    DataCopyPad(xTensorT_, this->gradYGm_[offset], dataCopyParams, copyPadExtParams);
    outputQue_.EnQue<T>(xTensorT_);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyDataIn(int64_t offset, int64_t curDimH, int64_t blkCount)
{
    xTensor_ = xQue_.AllocTensor<PromoteDataT>();

    DataCopyPadExtParams<T> copyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCopyParams{0, 0, 0, 0, 0};
    dataCopyParams.blockCount = blkCount;
    dataCopyParams.blockLen = curDimH * sizeof(T);
    dataCopyParams.srcStride = (inputShape_[1] - curDimH) * sizeof(T);
    dataCopyParams.dstStride = 0;

    if constexpr (IsSameType<PromoteDataT, T>::value) {
        // T == PromoteDataT
        DataCopyPad(xTensor_, this->gradYGm_[offset], dataCopyParams, copyPadExtParams);
        xQue_.EnQue<T>(xTensor_);
    } else {
        // T != PromoteDataT
        LocalTensor<T> inputLocal = xTensor_.template ReinterpretCast<T>();
        int64_t inputOffset = blkCount * utils::CeilAlign<int64_t>(curDimH, BLOCK_UB_SIZE);
        DataCopyPad(inputLocal[inputOffset], this->gradYGm_[offset], dataCopyParams, copyPadExtParams);

        // wait mte2
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // VF: T->PromoteDataT
        int64_t count = blkCount * utils::CeilAlign<int64_t>(curDimH, BLOCK_UB_SIZE);
        uint16_t loops = utils::CeilDiv<int64_t>(static_cast<int64_t>(count * sizeof(PromoteDataT)),
                                                 static_cast<int64_t>(Ops::Base::GetVRegSize()));
        uint32_t loopsStride = Ops::Base::GetVRegSize() / sizeof(PromoteDataT);

        __VEC_SCOPE__
        {
            uint32_t inputOffsetReg = inputOffset;

            __local_mem__ PromoteDataT* dst = (__local_mem__ PromoteDataT*)xTensor_.GetPhyAddr();
            __local_mem__ T* src = (__local_mem__ T*)inputLocal.GetPhyAddr() + inputOffsetReg;

            uint32_t sreg = static_cast<uint32_t>(count);

            AscendC::Reg::MaskReg mask;
            AscendC::Reg::RegTensor<T> aReg;
            AscendC::Reg::RegTensor<PromoteDataT> bReg;

            for (uint16_t i = 0; i < loops; i++) {
                mask = AscendC::Reg::UpdateMask<PromoteDataT>(sreg);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    aReg, (__local_mem__ T*)src + i * loopsStride);
                AscendC::Reg::Cast<PromoteDataT, T, castTrait0>(bReg, aReg, mask);
                AscendC::Reg::DataCopy(dst + i * loopsStride, bReg, mask);
            }
        }
        xQue_.EnQue<PromoteDataT>(xTensor_);
    }
    xTensor_ = xQue_.DeQue<PromoteDataT>();
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::SetGParam(int64_t singleH)
{
    int64_t singleG = usedUbSize_ /
                      (utils::CeilAlign<int64_t>(static_cast<int64_t>(singleH), static_cast<int64_t>(BLOCK_UB_SIZE)) *
                       sizeof(PromoteDataT));
    blockGNum_ = utils::CeilDiv<int64_t>(g_, singleG);
    mainG_ = singleG;
    tailG_ = g_ - (blockGNum_ - 1) * mainG_;
    if (blockGNum_ == 1) {
        mainG_ = tailG_;
    }

    bisectionPos_ = utils::FindNearestPower2(blockGNum_);
    cacheCount_ = utils::CalLog2(bisectionPos_) + 1;
    bisectionTail_ = blockGNum_ - bisectionPos_;
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::Process()
{
    // cal init index
    srcIndex_ = blockIdx_ * hMainFactor_; // mte start addr, maybe change
    startIndex_ = srcIndex_;              // keep no change

    // mte2 group
    CopyInGroup();

    // start process
    for (int64_t i = 0; i < groupIdxDim_; i++) {
        if (groupIdxType_ == 0 && i > 0) {
            g_ = groupIdx_.GetValue(i) - groupIdx_.GetValue(i - 1);
        } else {
            g_ = groupIdx_.GetValue(i);
        }

        gId_ = i;

        if (g_ == 0) {
            // null row, =========path1==========
            ProcessZeroG(curH_, startIndex_, gId_);
        } else if (g_ == 1) {
            ProcessOneG(curH_, startIndex_, gId_);
        } else if (g_ * BLOCK_SIZE * sizeof(PromoteDataT) > usedUbSize_) {
            // need cut G，and do reduceSum, =========path2==========
            ProcessSplitG(curH_);
        } else {
            // do not cut G, cut H, =========path3==========
            ProcessSplitH(curH_, g_, gId_);
        }

        // update addr param
        srcIndex_ += g_ * inputShape_[1];
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::ProcessOneG(int64_t curH, int64_t startIndex, int64_t gId)
{
    int64_t loops = utils::CeilDiv<int64_t>(curH, static_cast<int64_t>(outputSize_ / sizeof(T)));
    int64_t mainH = outputSize_ / sizeof(T);
    int64_t tailH = curH - (loops - 1) * mainH;

    for (int64_t loop = 0; loop < loops - 1; loop++) {
        int64_t offset = srcIndex_ + loop * mainH;
        CopyDataInForOne(offset, mainH, 1);

        int64_t outputDataOffset = startIndex + gId * inputShape_[1] + loop * mainH;
        CopyDataOutUnCast(outputDataOffset, mainH);
    }
    // tail
    int64_t offset = srcIndex_ + (loops - 1) * mainH;
    CopyDataInForOne(offset, tailH, 1);

    int64_t outputDataOffset = startIndex + gId * inputShape_[1] + (loops - 1) * mainH;
    CopyDataOutUnCast(outputDataOffset, tailH);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::ProcessZeroG(int64_t curH, int64_t startIndex, int64_t gId)
{
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};

    int64_t loops = utils::CeilDiv<int64_t>(curH, static_cast<int64_t>(outputSize_ / sizeof(PromoteDataT)));
    int64_t mainH = outputSize_ / sizeof(PromoteDataT);
    int64_t tailH = curH - (loops - 1) * mainH;

    LocalTensor<T> outputRes = zeroOutQue_.AllocTensor<T>();
    Duplicate<T>(outputRes, static_cast<T>(ZERO_VALUE), mainH); // broadcast
    zeroOutQue_.EnQue<T>(outputRes);

    outputRes = zeroOutQue_.DeQue<T>();
    copyOutParams.blockLen = mainH * sizeof(T);
    for (int64_t loop = 0; loop < loops - 1; loop++) {
        int64_t outputDataOffset = startIndex + gId * inputShape_[1] + loop * mainH;
        DataCopyPad(gradBiasGm_[outputDataOffset], outputRes, copyOutParams);
    }
    // tail
    copyOutParams.blockLen = tailH * sizeof(T);
    int64_t outputDataOffset = startIndex + gId * inputShape_[1] + (loops - 1) * mainH;
    DataCopyPad(gradBiasGm_[outputDataOffset], outputRes, copyOutParams);
    zeroOutQue_.FreeTensor(outputRes);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::ProcessSplitG(int64_t curH)
{
    int64_t hLoops = utils::CeilDiv<int64_t>(curH, BLOCK_SIZE);
    int64_t mainH = BLOCK_SIZE;
    int64_t tailH = curH - (hLoops - 1) * mainH;
    for (int64_t loop = 0; loop < hLoops; loop++) {
        int64_t singleH = mainH;
        if (loop == hLoops - 1) {
            singleH = tailH;
        }
        // set G param
        SetGParam(singleH);
        BinaryReduceSum(loop, singleH);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::ProcessSplitH(int64_t curH, int64_t g, int64_t gId)
{
    // cut H param
    int64_t hMainLen = 0;
    int64_t hTailLen = 0;
    int64_t loops = 0;
    utils::FindProcessedHLen<T, PromoteDataT>(maxPerHLen_, usedUbSize_, g, curH, hMainLen, hTailLen, loops);

    for (int64_t j = 0; j < loops - 1; j++) {
        int64_t offset = srcIndex_ + j * hMainLen;
        int64_t hMainAlign = utils::CeilAlign<int64_t>(hMainLen, BLOCK_UB_SIZE);

        CopyDataIn(offset, hMainLen, g);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);

        LocalTensor<PromoteDataT> outputRes = outputQue_.AllocTensor<PromoteDataT>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(g), static_cast<uint32_t>(hMainAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(outputRes, xTensor_, srcShape, false);
        outputQue_.EnQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>(outputRes);
        xQue_.FreeTensor(xTensor_);

        int64_t outputDataOffset = startIndex_ + gId * inputShape_[1] + j * hMainLen;
        CopyDataOut(outputDataOffset, hMainLen);
    }

    if (hTailLen > 0) {
        // process tail block
        int64_t offset = srcIndex_ + (loops - 1) * hMainLen;
        int64_t hTailAlign = utils::CeilAlign<int64_t>(hTailLen, BLOCK_UB_SIZE);

        CopyDataIn(offset, hTailLen, g);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);

        LocalTensor<PromoteDataT> outputRes = outputQue_.AllocTensor<PromoteDataT>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(g), static_cast<uint32_t>(hTailAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(outputRes, xTensor_, srcShape, false);
        outputQue_.EnQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>(outputRes);
        xQue_.FreeTensor(xTensor_);

        int64_t outputDataOffset = startIndex_ + gId * inputShape_[1] + (loops - 1) * hMainLen;
        CopyDataOut(outputDataOffset, hTailLen);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::BinaryReduceSum(int64_t loop, int64_t singleH)
{
    int64_t singleHAlign = utils::CeilAlign<int64_t>(singleH, BLOCK_UB_SIZE); // 32 byte Align
    int64_t i = 0;

    for (; i < bisectionTail_; i++) {
        int64_t offsetLeft = srcIndex_ + (loop * BLOCK_SIZE) + (i + bisectionPos_) * mainG_ * inputShape_[1];
        if (i == bisectionTail_ - 1) {
            CopyDataIn(offsetLeft, singleH, tailG_);
        } else {
            CopyDataIn(offsetLeft, singleH, mainG_);
        }
        uint32_t srcShape[2] = {
            (i == bisectionTail_ - 1) ? static_cast<uint32_t>(tailG_) : static_cast<uint32_t>(mainG_),
            static_cast<uint32_t>(singleHAlign)};
        tempRes_ = tempResBuf_.template Get<PromoteDataT>();
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(tempRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);
        int64_t offsetRight = srcIndex_ + (loop * BLOCK_SIZE) + i * mainG_ * inputShape_[1];

        CopyDataIn(offsetRight, singleH, mainG_);
        srcShape[0] = static_cast<uint32_t>(mainG_);
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);
        Add(computeRes_, computeRes_, tempRes_, 1 * singleH);
        tempResBuf_.FreeTensor(tempRes_);
        int64_t cacheID = utils::GetCacheID(i);
        cacheStride_ = utils::CeilDiv<int64_t>(static_cast<int64_t>(singleH),
                                               static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                       ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride_, singleH);
    }
    for (; i < bisectionPos_; i++) {
        int64_t offset = srcIndex_ + (loop * BLOCK_SIZE) + i * mainG_ * inputShape_[1];
        CopyDataIn(offset, singleH, mainG_);
        uint32_t srcShape[2] = {static_cast<uint32_t>(mainG_), static_cast<uint32_t>(singleHAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);
        int64_t cacheID = utils::GetCacheID(i);
        cacheStride_ = utils::CeilDiv<int64_t>(static_cast<int64_t>(singleH),
                                               static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                       ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride_, singleH);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    int64_t tmpBufOffest = (cacheCount_ - 1) * cacheStride_;
    LocalTensor<PromoteDataT> ubRes = tempBuf_[tmpBufOffest];
    int64_t outputDataOffset = startIndex_ + (loop * BLOCK_SIZE) + gId_ * inputShape_[1];
    CopyDataOutForBuf(outputDataOffset, ubRes, singleH);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyDataOutUnCast(int64_t outputDataOffset, int64_t dimH)
{
    // copyOutParams
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);

    // DataCopy
    LocalTensor<T> outputRes = outputQue_.DeQue<T>();
    DataCopyPad(gradBiasGm_[outputDataOffset], outputRes, copyOutParams);
    outputQue_.FreeTensor(outputRes);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyDataOutForBuf(int64_t outputDataOffset,
                                                                         LocalTensor<PromoteDataT>& ubRes, int64_t dimH)
{
    // if T == b16，b32->b16
    // copyOutParams
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);

    // DataCopy
    if constexpr (IsSameType<PromoteDataT, T>::value) {
        DataCopyPad(gradBiasGm_[outputDataOffset], ubRes, copyOutParams);
    } else {
        LocalTensor<T> outputLocal = ubRes.template ReinterpretCast<T>();
        Cast(outputLocal, ubRes, RoundMode::CAST_RINT, dimH);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad(gradBiasGm_[outputDataOffset], outputLocal, copyOutParams);
    }

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::CopyDataOut(int64_t outputDataOffset, int64_t dimH)
{
    // if T == b16，b32->b16
    // copyOutParams
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);

    LocalTensor<PromoteDataT> outputRes = outputQue_.DeQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>();
    // DataCopy
    if constexpr (IsSameType<PromoteDataT, T>::value) {
        DataCopyPad(gradBiasGm_[outputDataOffset], outputRes, copyOutParams);
    } else {
        LocalTensor<T> outputLocal = outputRes.template ReinterpretCast<T>();
        Cast(outputLocal, outputRes, RoundMode::CAST_RINT, dimH);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad(gradBiasGm_[outputDataOffset], outputLocal, copyOutParams);
    }
    outputQue_.FreeTensor(outputRes);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradSplitH<T, U>::UpdateCacheAux(const int64_t cacheID, const int64_t stride,
                                                                      const int64_t count)
{
    // count H size * VL
    uint16_t outerLoopTimes = utils::CeilDiv<int64_t>(static_cast<int64_t>(count * sizeof(PromoteDataT)),
                                                      static_cast<int64_t>(Ops::Base::GetVRegSize()));
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = ELEMENT_ONE_REPEAT_COMPUTE;
    uint32_t innerLoopStride = stride;

    LocalTensor<PromoteDataT> dstTensor = tempBuf_;
    LocalTensor<PromoteDataT> srcTensor = computeRes_;

    __VEC_SCOPE__
    {
        __local_mem__ PromoteDataT* dst = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr();
        __local_mem__ PromoteDataT* cah = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ PromoteDataT* src = (__local_mem__ PromoteDataT*)srcTensor.GetPhyAddr();

        uint32_t sreg = static_cast<uint32_t>(count);

        AscendC::Reg::RegTensor<PromoteDataT> aReg, bReg;
        AscendC::Reg::MaskReg pMask;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) { // outerLoopTimes is dimH size
            pMask = AscendC::Reg::UpdateMask<PromoteDataT>(sreg);
            DataCopy(aReg, (__local_mem__ PromoteDataT*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, (__local_mem__ PromoteDataT*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<PromoteDataT, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            DataCopy((__local_mem__ PromoteDataT*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}

} // namespace GroupedBiasAddGrad

#endif
